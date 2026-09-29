"""Guard: PersonaPlex GRPO log-probs (``rl.ppx_policy``) are the sampling distribution, frame for frame.

The knowledge GRPO computes pi_old / pi_theta / pi_ref by ONE teacher-forced pass over what the streaming engine fed
the backbone (recorded prompt-phase embeddings -- incl. the ``.pt`` voice-embedding replay -- then the recorded user-
phase input frames), reading log pi(y_t) at position P + t. If that sequence differs from what sampling saw (a missing
or shifted prompt, a frame off by one), the importance ratio is not pi_theta / pi_old of the sampled tokens and the
update is silently wrong. Through the REAL ``PpxRolloutGenerator`` on a tiny random PersonaPlex LM (real LMModel +
LMGen, float64, sampling on, voice prompt given as stored EMBEDDINGS + cache like NATM1.pt, a 3-token text prompt, a
stateless stub Mimi), 2 questions x G=2 per engine call of 4 rows, questions of different lengths:

  1. PARITY. The teacher-forced log-prob of every sampled non-PAD text token == the log-prob the sampler saw
     (log_softmax of the streaming step's logits at T=1), max |diff| < 1e-4 (the sampler seam casts to float32).
  2. TEXT ROW. The input frame of step t+1 carries the text token sampled at step t (text delay 0).
  3. SHAPES. Each rollout's codes are [18, ceil((question + capture) / frame)] -- the shorter question is cut to
     its own length -- and the probe path yields every question's n rollouts.
  4. GRAD. pi_theta (grad on) backpropagates into the LM.
  5. CAN FAIL. Without the prompt-phase prefix, or with the targets shifted by one frame, check 1 fails.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_rl_logprob.py'
"""

import contextlib
import math
import os
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402
from moshi_family.rl.ppx_policy import TARGET_ROW, PpxRolloutGenerator  # noqa: E402
from moshi_family.rl.rollout import AXIS_KNOWLEDGE, Segment  # noqa: E402
from moshi_family.train_data_common import TEXT_PADDING_ID  # noqa: E402

torch.set_default_dtype(torch.float64)
SR, FR = 24000, 12.5
FS = int(SR / FR)
CAP = 0.4
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


class StubMimi(torch.nn.Module):
    """Stateless codec: codes are a pure function of the frame; decoding writes the codes into the PCM."""

    def __init__(self):
        super().__init__()
        self._streaming_state = None
        self.frame_rate, self.sample_rate = FR, SR

    def encode(self, x):
        base = (x.to(torch.float64).abs().mean(-1) * 99991).long()  # [B, 1]
        return ((base + torch.arange(8) * 13) % 2048)[..., None]  # [B, 8, 1]

    def decode(self, codes):
        v = codes.to(torch.float32).mean(1, keepdim=True) / 2048.0
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


def make_generator():
    lm = LMModel(device="cpu", dtype=torch.float64, **KW).eval()
    for n, p in lm.named_parameters():  # as in the launcher: only LoRA / the backbone train, never embeddings
        p.requires_grad = not n.split(".")[0].endswith("emb")
    g = LMGen(lm, device="cpu", use_sampling=True, audio_silence_frame_cnt=3, sample_rate=SR, frame_rate=FR)
    g.streaming_forever(1)
    # a stored voice prompt, as the released .pt voices are: per-frame input embeddings + the token cache
    rng = torch.Generator().manual_seed(7)
    frames = []
    for _ in range(5):
        c = torch.randint(0, 100, (1, 17, 1), generator=rng)
        frames.append(lm.embed_codes(c).detach())  # [1, 1, D]
    g.voice_prompt_embeddings = torch.stack(frames)  # [V, 1, 1, D]
    g.voice_prompt_cache = torch.randint(0, 100, (1, 17, g.max_delay + 3), generator=rng)
    g.text_prompt_tokens = [5, 6, 7]
    model = types.SimpleNamespace(
        lm_gen=g, mimi=StubMimi(), other_mimi=StubMimi(), text_tokenizer=StubTok(), sample_rate=SR, device="cpu"
    )
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return PpxRolloutGenerator(model, num_samples=2, rows_per_call=4, capture_s=CAP, decode_audio=True)


def segments():
    r = np.random.default_rng(3)
    out = []
    for i, sec in enumerate((0.7, 1.13, 0.52)):
        pcm = (0.3 * r.standard_normal(int(sec * SR))).astype(np.float32)
        out.append(
            Segment(
                segment_id=f"q{i}",
                axis=AXIS_KNOWLEDGE,
                user_pcm=pcm,
                sample_rate=SR,
                meta={"question_end_s": sec, "question_id": i},
            )
        )
    return out


def run(mutant: str | None = None):
    """-> (max parity diff over all batches, text-row ok, shapes ok, batches)."""
    gen = make_generator()
    segs = segments()
    torch.manual_seed(1234)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        batches = list(gen.iter_generate(segs))
    worst, textrow, shapes = 0.0, True, len(batches) == len(segs)
    for seg, b in zip(segs, batches):
        n = math.ceil((len(seg.user_pcm) + round(CAP * SR)) / FS)
        shapes &= tuple(b.codes.shape) == (2, 18, n)
        x, y = b.codes[:, :TARGET_ROW], b.codes[:, TARGET_ROW]
        textrow &= bool(torch.equal(x[:, 0, 1:], y[:, :-1]))
        codes = b.codes
        if mutant == "shift":
            codes = codes.clone()
            codes[:, TARGET_ROW] = torch.roll(codes[:, TARGET_ROW], 1, dims=1)
        saved = gen._prefix
        if mutant == "no_prefix":
            gen._prefix = saved[:, :0]
        with torch.no_grad():
            logp, mask = gen.text_logprobs(codes)
        gen._prefix = saved
        # the sampler's log-probs: recompute them for this batch's rows from the recorded sample (rows = q x G)
        slp = b.stats["_sample_logp"]
        mask = mask & (codes[:, TARGET_ROW] != TEXT_PADDING_ID)
        worst = max(worst, float(((logp - slp).abs() * mask).max()))
    return worst, textrow, shapes, gen, segs


def main():
    import moshi_family.rl.ppx_policy as pp

    # expose the per-row sampling log-probs on the batch for this guard (the real stats keep only the summary)
    orig = pp.PpxRolloutGenerator._batch

    def _batch(self, seg, rows, score=True):
        b = orig(self, seg, rows, score)
        b.stats["_sample_logp"] = torch.stack([r["sample_logp"] for r in rows])
        return b

    pp.PpxRolloutGenerator._batch = _batch
    failures = []
    worst, textrow, shapes, gen, segs = run()
    for name, ok, detail in (
        ("parity", worst < 1e-4, f"max |tf - sample| = {worst:.2e}"),
        ("text row", textrow, ""),
        ("shapes", shapes, ""),
    ):
        print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        probe = gen.probe_batches(segs, 3)
    ok = len(probe) == 3 and all(len(r) == 3 and all(x.model_pcm.size > 0 for x in r) for _, r in probe)
    print(f"[{'ok' if ok else 'FAIL'}] probe batches (3 questions x 3 rollouts, audio decoded)")
    if not ok:
        failures.append("probe")
    b = next(iter(gen.iter_generate(segs[:1])))
    logp, mask = gen.text_logprobs(b.codes)
    (logp * mask).sum().backward()
    ok = gen.core.text_linear.weight.grad is not None and float(gen.core.text_linear.weight.grad.abs().sum()) > 0
    print(f"[{'ok' if ok else 'FAIL'}] grad reaches the LM")
    if not ok:
        failures.append("grad")
    for mutant in ("no_prefix", "shift"):
        w, *_ = run(mutant)
        if w < 1e-4:
            failures.append(f"parity cannot fail: mutant {mutant} passes ({w:.2e})")
        else:
            print(f"[ok] mutant {mutant} breaks parity ({w:.2e})")
    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
