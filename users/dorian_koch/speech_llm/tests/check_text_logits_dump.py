"""Guard: the PersonaPlex text-logit dump (``--text_logits_out``) records the monologue without changing it.

``TextLogitRecorder`` (moshi_family/personaplex/text_logits.py) sits in ``LMGen``'s ``text_sample_fn`` seam and
re-implements the released text sampling call. Two ways it could silently go wrong: it consumes the RNG or samples
differently (the probe's replies would then no longer be the scored benchmark replies its logits are meant to
explain), or its rows are misaligned with the frames (a logit row labelled frame t would describe another frame).
Through the REAL ``engine.run_pairs`` (B=1 serial and the B=2 batched path, 3 clips of different lengths), tiny
random PersonaPlex LM (real LMModel + LMGen, sampling on, temp 0.7 / top-k 25), stub voice/system components and
a stateless stub Mimi:

  1. UNCHANGED. Same seed with and without the recorder -> byte-identical reply wavs.
  2. SHAPES. One ``.npy``/``.json`` pair; per clip ``n_frames == ceil(padded samples / frame)``; rows = sum.
  3. ALIGNED. Row t's sampled id is the text token the LM emitted for frame t: the text tokens ``lm_gen.step``
     returns during the user phase, shifted by ``max_delay``, equal the recorded ids; every recorded id is in
     the top-k of its own logit row.
  4. MONOLOGUE. The ``<i>.txt`` sidecar == the recorded ids decoded (specials dropped).
  5. CAN FAIL. A recorder that samples at another temperature breaks 1; one whose ids are off by a frame
     breaks 3.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_text_logits_dump.py'
"""

import contextlib
import copy
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import sphn  # noqa: E402
import torch  # noqa: E402

import moshi_family.personaplex.components as components  # noqa: E402
from moshi_family.personaplex import engine  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402
from moshi_family.personaplex.text_logits import TextLogitRecorder, decode_monologue  # noqa: E402

torch.set_default_dtype(torch.float64)
SR, FR = 24000, 12.5
FS = int(SR / FR)
LEAD, CAP = 0.16, 0.4
TMP = Path(tempfile.mkdtemp(prefix="check_text_logits_dump_"))
torch.manual_seed(0)
KW = dict(loaders._lm_kwargs)
KW.update(
    dim=64,
    num_heads=4,
    num_layers=2,
    hidden_scale=2.0,
    text_card=100,
    context=256,
    depformer_dim=32,
    depformer_dim_feedforward=64,
    depformer_num_heads=2,
    depformer_num_layers=1,
    dep_q=16,
)
BASE_LM = LMModel(device="cpu", dtype=torch.float64, **KW).eval()


class StubMimi(torch.nn.Module):
    """Stateless codec: codes are a pure function of the frame; decoding writes the codes into the PCM."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self._streaming_state = None
        self.frame_rate, self.sample_rate = FR, SR

    def encode(self, x):
        x = x.to(torch.float64)
        base = (x.abs().mean(-1) * 99991).long()  # [B, 1]
        return ((base + torch.arange(8) * 13) % 2048)[..., None]  # [B, 8, 1]

    def decode(self, codes):
        v = codes.to(torch.float32).mean(1, keepdim=True) / 2048.0  # [B, 1, 1]
        return v.expand(-1, -1, FS).contiguous()

    def reset_streaming(self):
        pass

    def streaming_forever(self, _b):
        pass

    def _stop_streaming(self):
        pass


class StubTok:
    def id_to_piece(self, i):
        return f"▁w{i}"


class StubVoice:
    def __init__(self, *_a, **_k):
        pass

    def load(self, model, hf_repo):
        pass


class StubSystem:
    def __init__(self, *_a, **_k):
        pass

    def configure_generation(self, lm_gen, *, tokenizer, clip_key=None):
        lm_gen.text_prompt_tokens = [5, 6, 7]


components.VoicePrompt, components.SystemPrompt = StubVoice, StubSystem


def make_model():
    g = LMGen(
        copy.deepcopy(BASE_LM),
        device="cpu",
        use_sampling=True,
        audio_silence_frame_cnt=3,
        sample_rate=SR,
        frame_rate=FR,
    )
    g.streaming_forever(1)
    return types.SimpleNamespace(
        lm_gen=g, mimi=StubMimi(), other_mimi=StubMimi(), text_tokenizer=StubTok(), sample_rate=SR, device="cpu"
    )


def inputs():
    r = np.random.default_rng(3)
    d = TMP / "in"
    d.mkdir(exist_ok=True)
    paths = []
    for i, sec in enumerate((0.7, 1.13, 0.52)):
        p = d / f"{i}.wav"
        sphn.write_wav(str(p), (0.3 * r.standard_normal(int(sec * SR))).astype(np.float32), SR)
        paths.append(p)
    return paths


class ShiftedIds(TextLogitRecorder):
    """Mutant: ids recorded one frame late (the alignment bug class)."""

    def stop(self):
        logits, ids = super().stop()
        return logits, np.roll(ids, 1, axis=1)


class OtherTemp(TextLogitRecorder):
    """Mutant: samples at temperature 1.0 instead of the generator's."""

    def sample(self, text_logits):
        return self._sample_token(text_logits, True, 1.0, self.lm_gen.top_k_text)


def run(B, name, recorder_cls=None):
    m = make_model()
    out = TMP / name
    pairs = [(str(p), str(out / p.name)) for p in inputs()]
    rec, emitted = None, []
    if recorder_cls is not None:
        rec = recorder_cls(m.lm_gen, m.text_tokenizer)
        step, start = m.lm_gen.step, rec.start

        def spy(*a, **k):
            r = step(*a, **k)
            if rec.active and r is not None:
                emitted[-1].append(r[:, 0].reshape(-1).tolist())  # the text token each row emitted this step
            return r

        def start_group():
            emitted.append([])  # one list per recorded group (B=1: per clip)
            start()

        m.lm_gen.step, rec.start = spy, start_group
    torch.manual_seed(1234)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        engine.run_pairs(m, pairs, hf_repo="stub", lead_in_s=LEAD, capture_s=CAP, batch_size=B, recorder=rec)
        if rec is not None:
            rec.save(str(out / "dump"), "text_logits.test", frame_rate=FR)
    wavs = [sphn.read(o)[0] for _, o in pairs]
    return m, pairs, rec, emitted, wavs, out


def check(B, recorder_cls=TextLogitRecorder):
    """-> dict of check name -> bool for one batch size and recorder class."""
    _, _, _, _, ref, _ = run(B, f"b{B}_off")
    m, pairs, rec, emitted, wavs, out = run(B, f"b{B}_{recorder_cls.__name__}", recorder_cls)
    ok = {"unchanged": all(np.array_equal(a, b) for a, b in zip(ref, wavs))}
    meta = json.loads((out / "dump" / "text_logits.test.json").read_text())
    L = np.load(out / "dump" / "text_logits.test.npy", mmap_mode="r")
    V = m.lm_gen.lm_model.text_linear.out_features
    shapes = L.shape == (sum(c["n_frames"] for c in meta["clips"]), V) and len(meta["clips"]) == len(pairs)
    D, aligned, mono = m.lm_gen.max_delay, True, True
    for k, ((in_wav, out_wav), c) in enumerate(zip(pairs, meta["clips"])):
        n = math.ceil((sphn.read(in_wav)[0].shape[-1] + round((LEAD + CAP) * SR)) / FS)
        shapes &= c["n_frames"] == n and c["key"] == Path(in_wav).stem
        rows = np.asarray(L[c["row_start"] : c["row_start"] + c["n_frames"]], dtype=np.float32)
        ids = np.asarray(c["sampled"])
        topk = np.argsort(-rows, axis=-1)[:, : m.lm_gen.top_k_text]
        aligned &= bool(all(i in t for i, t in zip(ids, topk)))
        # Clip k is row k % B of recorded group k // B. The step returning at user step t emits the text token
        # sampled max_delay steps earlier (acoustic delay), so emitted[t] == ids[t - D] for t >= D.
        g, b = divmod(k, B)
        per = [e[b] for e in emitted[g]][:n]
        aligned &= len(per) == n and per[D:] == [int(x) for x in ids[: n - D]]
        mono &= Path(out_wav).with_suffix(".txt").read_text() == decode_monologue(StubTok(), ids) == c["monologue"]
    ok.update(shapes=bool(shapes), aligned=bool(aligned), monologue=bool(mono))
    return ok


def main():
    failures = []
    for B in (1, 2):
        ok = check(B)
        for name, v in ok.items():
            print(f"[{'ok' if v else 'FAIL'}] B={B} {name}")
            if not v:
                failures.append(f"B={B} {name}")
        bad_t = check(B, OtherTemp)
        if bad_t["unchanged"]:
            failures.append(f"B={B}: check 1 cannot fail (a temperature-1.0 recorder passes)")
        bad_s = check(B, ShiftedIds)
        if bad_s["aligned"]:
            failures.append(f"B={B}: check 3 cannot fail (ids off by one frame pass)")
        if not bad_t["unchanged"] and not bad_s["aligned"]:
            print(f"[ok] B={B} the mutants fail (other temperature breaks 1, shifted ids break 3)")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
