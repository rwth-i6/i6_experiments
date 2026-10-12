"""Guard: PersonaPlex's batched MFDB path (``run_pairs(batch_size=B)``) produces, per clip, exactly the B=1 output.

2026-09-25. Batched MFDB/MTR rows came out shifted vs B=1 (backchannel TOR). The GPU probes
(analysis/ppx_batch_determinism.py, ppx_mimi_batch_encode.py) found no logic bug: with greedy decoding the
LM rows are bit-identical to B=1 and the only B-dependent computation is Mimi's cuDNN TF32 convolutions.
This guard pins the batched path's logic with a deterministic model so a future change cannot introduce a
real one (cross-row leakage, clip order, per-row time alignment, prompt broadcast, padding of short clips,
per-row streaming state, a partial last group).

  1. PARITY. A tiny random float64 PersonaPlex LM (real LMModel + LMGen, greedy) with a stateful, causal,
     per-row stub Mimi, driven through the REAL ``engine.run_pairs``: 5 clips of different lengths (a partial
     final frame, one clip of 2 frames) at B=1 and at B=3 (two groups, the second one partial) -> every output
     wav is byte-identical; and the batched outputs do not depend on which clips share a group (B=5 in reverse
     order).
  2. CAN FAIL. Each mutant must break check 1: (a) cross-row leakage in the encoder, (b) a batched group
     returning its outputs in the wrong order, (c) a batch-size-dependent encoder rounding (what cuDNN TF32 does
     to the real Mimi), (d) the batched rows fed one frame late.

moshi_family venv, CPU, seconds:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_batch_parity.py'
"""

import contextlib
import copy
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import sphn  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex import engine  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
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
SR, FS = 24000, 1920


class StubMimi:
    """Causal, stateful per row (codes depend on this frame and the previous one of the SAME row), no weights.

    ``mutant``: None, "leak" (codes also see the neighbouring row), "bvar" (rounding depends on the batch size,
    like TF32 kernel choice does), "late" (batched rows see the previous frame instead of the current one).
    """

    def __init__(self, mutant=None):
        self.mutant = mutant
        self._streaming_state = None
        self._param = torch.zeros(1)

    def parameters(self):
        return iter([self._param])

    def streaming_forever(self, b):
        self._streaming_state = types.SimpleNamespace(prev=torch.zeros(b), last=torch.zeros(b, 1, FS), b=b)
        self._streaming_state.__exit__ = lambda *a: None

    def _stop_streaming(self):
        self._streaming_state = None

    def reset_streaming(self):
        s = self._streaming_state
        s.prev.zero_()
        s.last.zero_()

    def encode(self, x):
        s = self._streaming_state
        x = x.to(torch.float64)
        if self.mutant == "late" and x.shape[0] > 1:
            x, s.last = s.last.clone(), x.clone()
        e = x.abs().mean(dim=(1, 2)) * 1000.0
        if self.mutant == "leak" and x.shape[0] > 1:
            e = e + 0.5 * e.roll(1, 0)
        if self.mutant == "bvar":
            e = e + 0.37 * (x.shape[0] > 1)
        base = torch.floor(e + 3.0 * s.prev).long()
        s.prev = e.clone()
        k = torch.arange(8).view(1, 8)
        return ((base.view(-1, 1) + 37 * k) % 2048).view(-1, 8, 1)

    def decode(self, tokens):
        v = tokens.to(torch.float64).mean(dim=1, keepdim=True) / 2048.0  # [B, 1, 1]
        return (v * torch.linspace(-1.0, 1.0, FS).view(1, 1, FS)).to(torch.float32)


class Tok:
    def encode(self, s):
        return [4 + (ord(c) % 90) for c in s[:12]]


def make_model(voice_dir, mutant=None):
    g = LMGen(
        copy.deepcopy(BASE_LM),
        device="cpu",
        use_sampling=False,
        audio_silence_frame_cnt=3,
        sample_rate=SR,
        frame_rate=12.5,
    )
    g.streaming_forever(1)
    m = types.SimpleNamespace(
        lm_gen=g,
        mimi=StubMimi(mutant),
        other_mimi=StubMimi(),
        text_tokenizer=Tok(),
        sample_rate=SR,
        device="cpu",
    )
    m.set_voice = lambda p: g.load_voice_prompt_embeddings(p)
    m.mimi.streaming_forever(1)
    m.other_mimi.streaming_forever(1)
    return m


def write_voice(d):
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(4, 1, 1, KW["dim"], generator=g) * 0.1  # 4 voice frames, batch 1
    ct = max(KW["delays"]) + 3
    cache = torch.randint(0, 100, (1, KW["n_q"] + 1, ct), generator=g)
    torch.save({"embeddings": emb, "cache": cache}, os.path.join(d, "stub.pt"))


def write_clips(d):
    r = np.random.default_rng(3)
    paths = []
    for i, n in enumerate((7 * FS + 500, 2 * FS, 11 * FS, 5 * FS + 1234, 9 * FS)):
        p = os.path.join(d, f"in{i}.wav")
        sphn.write_wav(p, (0.3 * r.standard_normal(n)).astype(np.float32), SR)
        paths.append(p)
    return paths


def run(d, clips, B, tag, mutant=None, order=None):
    m = make_model(d, mutant)
    idx = list(range(len(clips))) if order is None else order
    pairs = [(clips[i], os.path.join(d, tag, f"{i}.wav")) for i in idx]
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        engine.run_pairs(
            m, pairs, hf_repo="unused", voice="stub.pt", voice_prompt_dir=d, system_prompt="be nice", batch_size=B
        )
    return [open(os.path.join(d, tag, f"{i}.wav"), "rb").read() for i in range(len(clips))]


def main():
    with tempfile.TemporaryDirectory() as d:
        write_voice(d)
        clips = write_clips(d)
        ref = run(d, clips, 1, "b1")
        ref_again = run(d, clips, 1, "b1_again")
        assert ref == ref_again, "B=1 itself is not deterministic"
        got = run(d, clips, 3, "b3")
        bad = [i for i in range(len(clips)) if got[i] != ref[i]]
        assert not bad, f"batched B=3 differs from B=1 for clips {bad}"
        got5 = run(d, clips, 5, "b5rev", order=[4, 3, 2, 1, 0])
        bad = [i for i in range(len(clips)) if got5[i] != ref[i]]
        assert not bad, f"batched B=5 (reverse order) differs from B=1 for clips {bad}"
        print("[ok] 1 batched run_pairs (B=3 two groups incl. a partial one; B=5 reversed) == B=1, byte for byte")

        orig = engine._stream_group
        for mutant in ("leak", "order", "bvar", "late"):
            try:
                if mutant == "order":
                    engine._stream_group = lambda m, a, B, **kw: orig(m, a, B, **kw)[::-1]
                    got = run(d, clips, 3, f"mut_{mutant}")
                    r1 = ref
                else:
                    got = run(d, clips, 3, f"mut_{mutant}", mutant=mutant)
                    r1 = run(d, clips, 1, f"mut_{mutant}_b1", mutant=mutant)
            finally:
                engine._stream_group = orig
            assert got != r1, f"check 1 cannot fail: mutant {mutant!r} passes"
            print(f"[ok] 2 mutant {mutant!r} breaks the parity (the guard can fail)")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
