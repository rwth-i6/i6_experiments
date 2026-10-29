"""Guard: base-Moshi knowledge GRPO (``rl.moshi_policy``) samples like the benchmark and scores what it sampled.

Knowledge GRPO on base Moshi records what the streaming engine fed the
backbone and computes pi_old / pi_theta / pi_ref by ONE teacher-forced ``forward_text`` over that recording. Two
things must hold, or the run trains on a distribution the benchmark never sees, or on a wrong importance ratio:
through the REAL ``MoshiRolloutGenerator`` + ``run_inference.InferenceState`` on a tiny random Moshi LM (real
LMModel + LMGen, float64, sampling on, a stateless stub Mimi), 2 questions x G = 2 per engine call of 4 rows,
questions of different lengths:

  1. SAME LOOP AS THE BENCHMARK. At the same seed, the sampled text of every row == ``InferenceState.run``'s text
     output on the same padded clips (``moshi_engine.run_pairs``'s engine) -- shifted by LMGen's ``max_delay`` --
     and the recorded clip length is its full frames + 1 (frame 0 stepped twice).
  2. PARITY. The teacher-forced log-prob of every sampled non-PAD text token == the sampler's (log_softmax of the
     streaming step's logits at T = 1), max |diff| < 1e-5.
  3. TEXT ROW. The input frame of step t+1 carries the text token sampled at step t (text delay 0).
  4. GRAD. pi_theta (grad on) backpropagates into the LM.
  5. CAN FAIL. Without the warm-up step (frame 0 stepped once) check 1 fails; with the targets shifted by one frame
     check 2 fails.

moshi_family venv, CPU, seconds (no checkpoint download):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_moshi_rl_logprob.py'
"""

import contextlib
import os
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.models import loaders  # noqa: E402
from moshi_family.models.lm import LMModel  # noqa: E402
from moshi_family.rl import moshi_policy  # noqa: E402
from moshi_family.rl.moshi_policy import MoshiRolloutGenerator  # noqa: E402
from moshi_family.rl.ppx_policy import TARGET_ROW  # noqa: E402
from moshi_family.rl.rollout import AXIS_KNOWLEDGE, Segment  # noqa: E402
from moshi_family.run_inference import InferenceState  # noqa: E402
from moshi_family.train_data_common import TEXT_PADDING_ID  # noqa: E402

torch.set_default_dtype(torch.float64)
SR, FR = 24000, 12.5
FS = int(SR / FR)
CAP = 0.4
ROWS, G = 4, 2
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
    norm="rms_norm",
)


class StubMimi(torch.nn.Module):
    """Stateless codec: codes are a pure function of the frame; decoding writes the codes into the PCM."""

    def __init__(self):
        super().__init__()
        self.frame_rate, self.sample_rate, self.cardinality, self.num_codebooks, self.channels = FR, SR, 2048, 8, 1

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


class StubTok:
    def id_to_piece(self, i):
        return f"▁w{i}"

    def eos_id(self):
        return -1


def make_state():
    torch.manual_seed(0)
    lm = LMModel(device="cpu", dtype=torch.float64, **KW).eval()
    for n, p in lm.named_parameters():  # as in the launcher: the LM trains, never the input embeddings
        p.requires_grad = not n.split(".")[0].endswith("emb")
    ckpt = types.SimpleNamespace(model_type="moshi")
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return InferenceState(ckpt, StubMimi(), StubTok(), lm, ROWS, 1.0, "cpu")


def segments():
    r = np.random.default_rng(3)
    out = []
    for i, sec in enumerate((0.7, 1.13, 0.52, 0.9)):
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


def generate(gen, segs, seed):
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        return list(gen.iter_generate(segs))


def reference_text(state, segs, seed):
    """``InferenceState.run`` on the same rows (each question repeated G times), as run_pairs feeds it."""
    pcms = [np.concatenate([s.user_pcm, np.zeros(round(CAP * SR), np.float32)]) for s in segs for _ in range(G)]
    out = []
    for i in range(0, len(pcms), ROWS):
        chunk = pcms[i : i + ROWS]
        t = max(len(p) for p in chunk)
        arr = np.zeros((ROWS, 1, t), np.float32)
        for b, p in enumerate(chunk):
            arr[b, 0, : len(p)] = p
        state.mimi.reset_streaming()
        state.lm_gen.reset_streaming()
        torch.manual_seed(seed + i)
        with contextlib.redirect_stdout(open(os.devnull, "w")), torch.no_grad():
            items = state.run(torch.from_numpy(arr))
        out += [items[b][0] for b in range(len(chunk))]
    return out


def main():
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    segs = segments()
    # 1. the benchmark's loop: one engine call per 2 questions, seeded per call like the reference
    state = make_state()
    gen = MoshiRolloutGenerator(state, StubTok(), num_samples=G, rows_per_call=ROWS, capture_s=CAP)
    batches = []
    for i in range(0, len(segs), ROWS // G):
        batches += generate(gen, segs[i : i + ROWS // G], seed=100 + i * G)
    ref = reference_text(state, segs, seed=100)
    md = state.lm_gen.max_delay
    same, lens_ok = True, True
    for k, b in enumerate(batches):
        n_frames = (len(segs[k].user_pcm) + round(CAP * SR)) // FS
        lens_ok &= b.codes.shape[2] == n_frames + 1
        for g in range(G):
            y = b.codes[g, TARGET_ROW]
            r = ref[k * G + g]
            m = min(len(r), len(y) - md)
            same &= bool(torch.equal(y[:m], r[:m]))
    check("same tokens as InferenceState.run", same)
    check("clip length = full frames + 1", lens_ok)

    # 2./3. parity and text row
    worst, textrow = 0.0, True
    for b in batches:
        x, y = b.codes[:, :TARGET_ROW], b.codes[:, TARGET_ROW]
        textrow &= bool(torch.equal(x[:, 0, 1:], y[:, :-1]))
        with torch.no_grad():
            logp, mask = gen.text_logprobs(b.codes)
        slp = b.stats["_sample_logp"]
        worst = max(worst, float(((logp - slp).abs() * (mask & (y != TEXT_PADDING_ID))).max()))
    check("parity", worst < 1e-5, f"max |tf - sample| = {worst:.2e}")
    check("text row", textrow)

    # 4. grad
    logp, mask = gen.text_logprobs(batches[0].codes)
    (logp * mask).sum().backward()
    gr = state.lm_gen.lm_model.text_linear.weight.grad
    check("grad reaches the LM", gr is not None and float(gr.abs().sum()) > 0)

    # 5. mutants
    b = batches[0]
    codes = b.codes.clone()
    codes[:, TARGET_ROW] = torch.roll(codes[:, TARGET_ROW], 1, dims=1)
    with torch.no_grad():
        logp, mask = gen.text_logprobs(codes)
    w = float(((logp - b.stats["_sample_logp"]).abs() * mask).max())
    check("mutant shift breaks parity", w > 1e-3, f"({w:.2e})")
    state2 = make_state()
    gen2 = MoshiRolloutGenerator(state2, StubTok(), num_samples=G, rows_per_call=ROWS, capture_s=CAP)
    gen2.step_offset = 0
    orig_step = state2.lm_gen.step
    calls = {"n": 0}

    def no_warmup(codes):  # frame 0 stepped once: drop the first call of every engine run
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return orig_step(codes)

    gen2._run_orig = gen2._run

    def run_once(pcms):
        calls["n"] = 0
        state2.lm_gen.step = no_warmup
        try:
            o = gen2._run_orig(pcms)
        except AssertionError:  # the step count no longer matches: also a detected failure
            return None
        finally:
            state2.lm_gen.step = orig_step
        return o

    gen2._run = run_once
    try:
        mb = generate(gen2, segs[: ROWS // G], seed=100)
        mut_same = all(
            torch.equal(
                mb[k].codes[g, TARGET_ROW][: len(ref[k * G + g]) - 1], ref[k * G + g][: len(ref[k * G + g]) - 1]
            )
            for k in range(len(mb))
            for g in range(G)
        )
    except Exception:
        mut_same = False
    check("mutant no-warm-up breaks the benchmark match", not mut_same)

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


def _expose_sample_logp():
    """Keep the per-row sampling log-probs on the batch for this guard (the real stats keep only the summary)."""
    orig = moshi_policy.MoshiRolloutGenerator._batch

    def _batch(self, seg, rows, score=True):
        b = orig(self, seg, rows, score)
        b.stats["_sample_logp"] = torch.stack([r["sample_logp"] for r in rows])
        return b

    moshi_policy.MoshiRolloutGenerator._batch = _batch


if __name__ == "__main__":
    _expose_sample_logp()
    main()
