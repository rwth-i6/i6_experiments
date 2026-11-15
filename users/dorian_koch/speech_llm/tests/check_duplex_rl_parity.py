"""Guard: RL through the architecture seam on PersonaPlex and base Moshi == the family's batched RL generators.

``moshi_family.duplex.rollouts.DuplexRolloutGenerator`` drives any adapter with the whole contract (a recording
session, ``trace()``, ``text_logprobs(trace)``); the production generators of the Moshi family batch several segments
per engine call. Both must sample and score the same thing, or a result obtained through one says nothing about the
other. On tiny random models (real LMModel + LMGen, float64, CPU, sampling on, stub codec; the builders of
``check_ppx_rollout_parity`` / ``check_ppx_selfplay_rl`` / ``check_moshi_rl_logprob``):

  1. PERSONAPLEX KNOWLEDGE. One segment with the run's prompt and one with its own system prompt, G = 4, same seed:
     the seam's trace == ``PpxRolloutGenerator``'s batch -- codes (17 backbone input rows + the sampled text), the
     prompt prefix embeddings, pi_old (teacher-forced, the same chunk), the decoded audio; pi_theta on the trace
     backpropagates into the LM. CAN FAIL: text targets one frame late in the trace; a prefix one frame short.
  2. PERSONAPLEX SELF-PLAY. Two recording exchange sessions under ``duplex.selfplay.selfplay_native`` ==
     ``PpxSelfPlayGenerator`` (two players, no partner channel) on 6 conversations with prompts of different
     lengths: per side and conversation the codes up to its own end, the sampling log-probs, and the adapter's
     scoring == the generator's. CAN FAIL: the forced prompt's text targets left as sampled (not PAD).
  3. BASE MOSHI. The seam's trace == ``MoshiRolloutGenerator``'s batch (codes, pi_old), n + 1 frames for n user
     frames. CAN FAIL: the first frame stepped once; targets one frame late in the scoring forward.
  4. THE TRAINER. The real ``rl.trainer.RLTrainer.run`` on PersonaPlex through ``DuplexRolloutGenerator`` (3 steps x
     2 segments x 4 completions): finite losses, the first update on-policy (ratio 1, KL 0 to a reference with the
     start weights), and the first group's log-probs move with its advantages. CAN FAIL: a reference with other
     weights gives a first KL > 0.

moshi_family venv, CPU, about a minute:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'CUDA_VISIBLE_DEVICES= python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_duplex_rl_parity.py'
"""

import contextlib
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import check_moshi_rl_logprob as mg  # noqa: E402
import check_ppx_selfplay_rl as sg  # noqa: E402
from moshi_family import moshi_duplex as md  # noqa: E402
from moshi_family.duplex.rollouts import DuplexRolloutGenerator  # noqa: E402
from moshi_family.duplex.selfplay import selfplay_native  # noqa: E402
from moshi_family.personaplex import duplex as pd  # noqa: E402
from moshi_family.personaplex.duplex import PersonaPlexDuplex  # noqa: E402
from moshi_family.rl import ppx_rollout as pr  # noqa: E402
from moshi_family.rl.moshi_policy import MoshiRolloutGenerator  # noqa: E402
from moshi_family.rl.ppx_rollout import TARGET_ROW  # noqa: E402
from moshi_family.rl.rollout import AXES, AXIS_KNOWLEDGE, Segment  # noqa: E402
from moshi_family.rl.tests import check_ppx_rollout_parity as kg  # noqa: E402
from moshi_family.train_data_common import TEXT_PADDING_ID  # noqa: E402

torch.set_default_dtype(torch.float64)
G = 4
FAIL: list = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}".rstrip(), flush=True)
    if not ok:
        FAIL.append(name)


@contextlib.contextmanager
def patched(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


def late_targets(trace_of):
    """Mutant: the trace's text targets one frame late."""

    def f(*a, **k):
        tr = trace_of(*a, **k)
        tr.prompt["codes"][:, TARGET_ROW] = torch.roll(tr.prompt["codes"][:, TARGET_ROW], 1, dims=1)
        return tr

    return f


# ---- 1. PersonaPlex knowledge ---------------------------------------------------------------------------------------
def knowledge(mutant=None):
    """-> per segment (codes equal, prefix equal, max |pi_old diff|, audio equal), and the seam generator + a batch."""
    segs = kg.segments((0.7, 1.13), prompts=(None, "be brief"))
    gen = kg.knowledge_gen(kg.make_model(), G=G, rows=G, logp_chunk=2)
    adapter = PersonaPlexDuplex(kg.make_model(), batch_size=G)
    seam = DuplexRolloutGenerator(
        adapter, adapter.frozen_reference(), group_size=G, role=kg.ROLE, seed=5, capture_s=kg.CAP, logp_chunk=2
    )
    ctx = contextlib.ExitStack()
    if mutant == "late_targets":
        ctx.enter_context(patched(pd, "_trace_of", late_targets(pd._trace_of)))
    if mutant == "short_prefix":
        head = pr.Recorder.head_embeddings
        ctx.enter_context(patched(pr.Recorder, "head_embeddings", lambda self, core: head(self, core)[:, :-1]))
    refs = []
    for k, seg in enumerate(segs):
        torch.manual_seed(5 + k * G)
        with kg.quiet():
            refs.append(gen.generate(seg))
    with ctx, kg.quiet():
        gots = [seam.generate(seg) for seg in segs]
    out = []
    for seg, ref, got in zip(segs, refs, gots):
        tr = got.codes
        n = len(ref.rollouts[0].model_pcm)
        out.append(
            (
                tr.prompt["codes"].shape == ref.codes.shape and torch.equal(tr.prompt["codes"], ref.codes),
                tr.prompt["prefix"].shape[1:] == gen._prefix_for(gen.prompt_key(seg)).shape[1:]
                and torch.equal(tr.prompt["prefix"][:1], gen._prefix_for(gen.prompt_key(seg))),
                float((got.old_logprobs - ref.old_logprobs).abs().max())
                if got.old_logprobs.shape == ref.old_logprobs.shape
                else float("inf"),
                all(np.array_equal(a.model_pcm[:n], b.model_pcm) for a, b in zip(got.rollouts, ref.rollouts)),
            )
        )
    return out, seam, got, adapter


def check_knowledge():
    out, seam, batch, adapter = knowledge()
    for k, (c, p, w, a) in enumerate(out):
        label = "the run's prompt" if k == 0 else "its own system prompt"
        check(f"1 PersonaPlex knowledge ({label}): codes == the batched generator's", c)
        check(f"1 PersonaPlex knowledge ({label}): prompt prefix", p)
        check(f"1 PersonaPlex knowledge ({label}): pi_old", w < 1e-12, f"(max |diff| {w:.1e})")
        check(f"1 PersonaPlex knowledge ({label}): decoded audio", a)
    core = adapter.model.lm_gen.lm_model
    core.zero_grad()
    with torch.enable_grad():
        lp, mask, _ = seam.logprobs(batch.codes)
        (lp * mask).sum().backward()
    gsum = sum(float(q.grad.abs().sum()) for q in core.parameters() if q.grad is not None)
    check("1 pi_theta on the trace backpropagates into the LM", gsum > 0, f"(sum |grad| {gsum:.2e})")
    core.zero_grad()
    out = knowledge("late_targets")[0]
    check("1 knowledge parity can fail: mutant late_targets", not any(c for c, *_ in out))
    out = knowledge("short_prefix")[0]
    check("1 knowledge parity can fail: mutant short_prefix", all(w > 1e-6 for _, _, w, _ in out), f"({out[0][2]:.1e})")


# ---- 2. PersonaPlex self-play ---------------------------------------------------------------------------------------
def selfplay(mutant=None):
    segs = sg.scenario_pairs()
    n = int(round(sg.CONV_S * sg.FR))
    players = [
        sg.make_model(sampling=True),
        sg.make_model(sampling=True),
    ]  # built before the seed: the first build draws
    torch.manual_seed(31)
    with sg.quiet():
        gen, o, convs = sg.gen_outputs(players, segs)
    C = len(convs)
    a, b = (
        PersonaPlexDuplex(sg.make_model(sampling=True), batch_size=C).open_exchange(
            C, silence_frames=sg.SIL, record=True
        )
        for _ in range(2)
    )
    start = [{"voice": [c["voices"][s] for c in convs], "role": [c["prompts"][s] for c in convs]} for s in (0, 1)]
    ctx = contextlib.ExitStack()
    if mutant == "sampled_prompt":
        orig = pd._trace_of
        ctx.enter_context(
            patched(pd, "_trace_of", lambda o, lanes, **k: orig(o, lanes, **{**k, "prompt_frames": None}))
        )
    torch.manual_seed(31)
    with ctx, sg.quiet():
        selfplay_native(a, b, n, start_a=start[0], start_b=start[1])
        traces = [a.trace(), b.trace()]
    codes_ok, slp, score = True, 0.0, 0.0
    for s, (sess, tr) in enumerate(zip((a, b), traces)):
        for c in range(C):
            row = gen._rows(o, s, c)
            t = row["codes"].shape[1]
            mine = tr.prompt["codes"][c, :, :t]
            codes_ok &= mine.shape == row["codes"].shape and bool(torch.equal(mine, row["codes"]))
            codes_ok &= int(tr.prompt["prompt_frames"][c]) == row["L"]
            slp = max(slp, float((tr.text_logprobs[c, :t] - row["sample_logp"]).abs().max()))
            with torch.no_grad():
                lp, m = gen.text_logprobs(row["codes"][None], side=s)
                got = sess.duplex.text_logprobs(tr[c])[:, :t]
            score = max(score, float(((lp - got).abs() * m).max()))
    return codes_ok, slp, score


def check_selfplay():
    c, slp, score = selfplay()
    check("2 PersonaPlex self-play: each side's trace == the generator's rows (codes, prompt frames)", c)
    check("2 PersonaPlex self-play: sampling log-probs", slp < 1e-12, f"(max |diff| {slp:.1e})")
    check(
        "2 PersonaPlex self-play: the adapter's scoring == the generator's", score < 1e-9, f"(max |diff| {score:.1e})"
    )
    c, _, _ = selfplay("sampled_prompt")
    check("2 self-play parity can fail: mutant sampled_prompt", not c)


# ---- 3. base Moshi --------------------------------------------------------------------------------------------------
def moshi(mutant=None):
    r = np.random.default_rng(9)
    pcm = (0.3 * r.standard_normal(int(10.4 * mg.FS))).astype(np.float32)  # Moshi's loop drops the partial frame
    seg = Segment(segment_id="m0", axis=AXIS_KNOWLEDGE, user_pcm=pcm, sample_rate=mg.SR, meta={"question_end_s": 0.8})
    tok = mg.StubTok()
    with sg.quiet():
        gen = MoshiRolloutGenerator(mg.make_state(), tok, num_samples=G, rows_per_call=mg.ROWS, capture_s=mg.CAP)
        torch.manual_seed(7)
        ref = gen.generate(seg)
    adapter = md.MoshiDuplex(
        types.SimpleNamespace(state=mg.make_state(), sample_rate=mg.SR, device="cpu"),
        repeat_first_frame=mutant != "one_first_step",
    )
    seam = DuplexRolloutGenerator(
        adapter, adapter.frozen_reference(), group_size=G, seed=7, capture_s=mg.CAP, partial_frame="drop"
    )
    ctx = contextlib.ExitStack()
    if mutant == "late_targets":
        fwd = md.moshi_text_logprobs

        def late(core, codes, full=False):
            codes = codes.clone()
            codes[:, TARGET_ROW] = torch.roll(codes[:, TARGET_ROW], 1, dims=1)
            return fwd(core, codes, full)

        ctx.enter_context(patched(md, "moshi_text_logprobs", late))
    with ctx, sg.quiet():
        got = seam.generate(seg)
    codes = got.codes.prompt["codes"]
    same = codes.shape == ref.codes.shape and bool(torch.equal(codes, ref.codes))
    w = (
        float((got.old_logprobs - ref.old_logprobs).abs().max())
        if got.old_logprobs.shape == ref.old_logprobs.shape
        else float("inf")
    )
    n = (len(pcm) + round(mg.CAP * mg.SR)) // mg.FS
    return same, w, got.codes.frames == n + 1


def check_moshi():
    same, w, frames = moshi()
    check("3 base Moshi: codes == the batched generator's", same)
    check("3 base Moshi: pi_old", w < 1e-12, f"(max |diff| {w:.1e})")
    check("3 base Moshi: n + 1 frames for n user frames", frames)
    same, _, _ = moshi("one_first_step")
    check("3 Moshi parity can fail: mutant one_first_step", not same)
    _, w, _ = moshi("late_targets")
    check("3 Moshi parity can fail: mutant late_targets", w > 1e-6, f"({w:.1e})")


# ---- 4. the trainer -------------------------------------------------------------------------------------------------
def reward(segment, rollouts):
    """More words of the upper half of the vocabulary = better: a reward the text stream alone decides."""
    return [
        {"high": float(np.mean(r.text_tokens[r.text_token_mask] > 50)) if r.text_token_mask.any() else 0.0}
        for r in rollouts
    ]


def train(other_reference=False):
    from moshi_family.rl.grpo import GRPOConfig
    from moshi_family.rl.trainer import RLTrainer, RLTrainerConfig

    model = kg.make_model()
    core = model.lm_gen.lm_model
    policy = PersonaPlexDuplex(model, batch_size=G)
    ref_model = kg.make_model()  # the start weights, never trained
    if other_reference:
        with torch.no_grad():
            w = ref_model.lm_gen.lm_model.text_linear.weight
            w.add_(0.05 * torch.randn(w.shape, generator=torch.Generator().manual_seed(1)))
    reference = PersonaPlexDuplex(ref_model, batch_size=G)
    gen = DuplexRolloutGenerator(policy, reference, group_size=G, role=kg.ROLE, seed=11, capture_s=kg.CAP)
    params = [q for q in core.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=0.05)
    rng = np.random.default_rng(3)
    n = [0]

    def sample(axis, context_s):
        n[0] += 1
        pcm = (0.3 * rng.standard_normal(int(0.6 * kg.SR))).astype(np.float32)
        return Segment(segment_id=f"seg{n[0]}", axis=axis, user_pcm=pcm, sample_rate=kg.SR, frame_rate=kg.FR)

    out = tempfile.mkdtemp()
    tr = RLTrainer(
        core=core,
        generator=gen,
        reward_registry={a: reward for a in AXES},
        ref_logprobs=gen.ref_logprobs,
        logprob_fn=gen.logprobs,
        optimizer=opt,
        base_lrs=[0.05],
        trainable_params=params,
        segment_source=type("Src", (), {"sample": staticmethod(sample)})(),
        save_fn=lambda step, final: None,
        cfg=RLTrainerConfig(
            max_steps=3,
            grad_accum=2,
            warmup_steps=1,
            log_every=1,
            save_every=0,
            overlap_scoring=True,
            lr_schedule="constant",
        ),  # fmt: skip
        grpo_cfg=GRPOConfig(kl_beta=0.01),
        out_dir=out,
    )
    updates, upd = [], tr._grpo_update

    def spy(axis, batch, comps):
        stats = upd(axis, batch, comps)
        updates.append((batch, comps, stats))
        return stats

    tr._grpo_update = spy
    with kg.quiet(), torch.enable_grad():
        tr.run()
    recs = [json.loads(line) for line in open(os.path.join(out, "metrics.train.jsonl"))]
    return recs, updates, gen


def check_trainer():
    recs, updates, gen = train()
    losses = [r.get("loss") for r in recs if "loss" in r]
    check(
        "4 RLTrainer.run on PersonaPlex through the seam",
        len(updates) == 6 and len(losses) >= 3 and all(np.isfinite(x) for x in losses)
        and not any(u[2].get("skipped") for u in updates),
        f"({len(updates)} updates, losses {[round(x, 4) for x in losses]})",
    )  # fmt: skip
    first = updates[0][2]
    check(
        "4 the first update is on-policy",
        abs(first["ratio"] - 1.0) < 1e-6 and abs(first["kl"]) < 1e-8,
        f"(ratio {first['ratio']:.8f}, kl {first['kl']:.1e})",
    )
    batch, _, stats = updates[0]
    with torch.no_grad():
        new = gen.logprobs(batch.codes)[0]
    gain = float((torch.as_tensor(stats["adv"]) * ((new - batch.old_logprobs) * batch.token_mask).sum(1)).sum())
    check("4 the first group's log-probs moved with its advantages", gain > 0, f"(sum adv * dlogp = {gain:.4f})")
    _, bad, _ = train(other_reference=True)
    check("4 can fail: a reference with other weights", abs(bad[0][2]["kl"]) > 1e-6, f"(kl {bad[0][2]['kl']:.1e})")


def main():
    assert TEXT_PADDING_ID == pd.PAD_ID == md.PAD_ID
    torch.set_grad_enabled(False)
    check_knowledge()
    check_selfplay()
    check_moshi()
    check_trainer()
    if FAIL:
        print("\nFAILED:\n  " + "\n  ".join(FAIL))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
