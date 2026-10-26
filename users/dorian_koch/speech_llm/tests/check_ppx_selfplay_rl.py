"""Guard: self-play GRPO on PersonaPlex (``rl.ppx_selfplay`` + ``rl.selfplay_reward``) samples the conversations the
self-play benchmark samples, scores them under the distribution it sampled from, and routes each side's update to that
side's policy.

Through the REAL ``PpxSelfPlayGenerator`` on tiny random PersonaPlex LMs (real LMModel + LMGen, float64, stub Mimi,
a stub tokenizer through the real ``encode_system_prompt``), conversations whose two sides have prompts of DIFFERENT
lengths (and different lengths across rows):

  1. REFERENCE PARITY. Greedy (audio argmax, text top-k 1), the generator's per-side audio output over the
     conversation equals ``personaplex.selfplay.converse_batched`` (the benchmark's worker) token for token, and its
     sampled text equals the reference's text shifted by ``max_delay`` -- for the SHARED policy (one LMGen, 2C rows)
     and for TWO players. CAN FAIL: without the prompt forcing (``_force_rows`` a no-op), or with each side hearing
     itself (the heard rows swapped), parity breaks.
  2. LOG-PROB PARITY. Sampling on: the teacher-forced log-prob (``text_logprobs`` over the row's recorded inputs from
     the first backbone call) of every trained token == the sampler's, max |diff| < 1e-4; the input text row at t+1
     == the token sampled at t over the trained span; the trained span starts at the prompt's length L (a step's
     token is the next step's input, so samples before L are overwritten) and the targets before it are PAD. CAN
     FAIL: targets shifted by a frame, or the prompt frames dropped from the forward.
  3. ROUTING (two players). Through the REAL ``RLTrainer._grpo_update``: a side-B group's loss puts gradient on
     player B only (exactly zero on A), and a side-A group's on A only. CAN FAIL: every side routed to player 0.
  4. REWARD. ``SelfPlayJudgeReward`` with a stub judge: one call per CONVERSATION (not per side); side A gets
     adherence_a, side B adherence_b, both the shared coherence / responsiveness; a failed call gets the group mean
     (zero advantage); ``render_message`` / ``parse_reply`` / ``two_sided_transcript`` equal the i6 originals
     (``selfplay_bench`` / ``selfplay_judge``, cut out by AST: those modules import sisyphus) on real inputs.
  5. SOURCE. A then B of one scenario per two draws; the length curriculum is constant within a step, rises
     linearly to the end length; the probe gets the end length; ``seek`` resumes at the same draw.
  6. EARLY STOP (``stop_silent_s``, 2026-10-26). All-silent calls stop early, rows cut there keep log-prob parity,
     the probe runs the whole slot. CAN FAIL: stop off, or speech, -> no stop.

moshi_family venv, CPU, ~1 min:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_selfplay_rl.py'
"""

import ast
import contextlib
import copy
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex import selfplay as sp  # noqa: E402
from moshi_family.personaplex.components import encode_system_prompt  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMGen, LMModel  # noqa: E402
from moshi_family.rl import ppx_selfplay as ps  # noqa: E402
from moshi_family.rl.ppx_policy import TARGET_ROW  # noqa: E402
from moshi_family.rl.rollout import AXIS_SELFPLAY, Segment  # noqa: E402
from moshi_family.rl.selfplay_reward import (  # noqa: E402
    SelfPlayJudgeReward,
    parse_reply,
    render_message,
    two_sided_transcript,
)
from moshi_family.train_data_common import TEXT_PADDING_ID  # noqa: E402

torch.set_default_dtype(torch.float64)
SR, FR = 24000, 12.5
FS = int(SR / FR)
SIL = 2
CONV_S = 1.2  # 15 frames
KW = dict(loaders._lm_kwargs)
KW.update(
    dim=64, num_heads=4, num_layers=2, hidden_scale=2.0, text_card=100, context=256, depformer_dim=32,
    depformer_dim_feedforward=64, depformer_num_heads=2, depformer_num_layers=1, dep_q=16,
)  # fmt: skip
I6 = os.path.join(os.getcwd(), "recipe", "i6_experiments", "users", "dorian_koch", "speech_llm")
FAIL: list = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}".rstrip(), flush=True)
    if not ok:
        FAIL.append(name)


class StubMimi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._streaming_state = None
        self.frame_rate, self.sample_rate = FR, SR

    def encode(self, x):
        raise AssertionError("self-play never encodes audio")

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
    """Ids 4..99 (never PAD 3 / EPAD 0): a word's id from its characters; pieces "▁w<i>"."""

    def encode(self, text):
        return [4 + (sum(map(ord, w)) % 90) for w in text.split()]

    def id_to_piece(self, i):
        return f"▁w{i}"


_BASE = {}


def make_model(seed=0, sampling=False, perturb=0.0):
    if seed not in _BASE:
        torch.manual_seed(seed)
        _BASE[seed] = LMModel(device="cpu", dtype=torch.float64, **KW).eval()
    lm = copy.deepcopy(_BASE[seed])
    if perturb:
        with torch.no_grad():
            lm.text_linear.weight.add_(perturb)
    for n, p in lm.named_parameters():  # as the launcher: never the input embeddings
        p.requires_grad = not n.split(".")[0].endswith("emb")
    # greedy = sampling with top-k 1 for text AND audio (LMGen only calls the recording text sampler when sampling)
    g = LMGen(lm, device="cpu", use_sampling=True, audio_silence_frame_cnt=SIL, sample_rate=SR, frame_rate=FR)
    g.streaming_forever(1)
    if not sampling:
        g.top_k_text, g.top_k = 1, 1
    return types.SimpleNamespace(
        lm_gen=g, mimi=StubMimi(), other_mimi=StubMimi(), text_tokenizer=StubTok(), sample_rate=SR, device="cpu"
    )


def scenario_pairs():
    """Three scenarios: the two sides' prompts and voices differ in length, and across scenarios."""
    r = np.random.default_rng(5)
    specs = [("one two three", "alpha beta gamma delta epsilon zeta", 5, 4),
             ("a much longer prompt for side a here ok", "short b", 3, 6),
             ("mid size prompt text", "another mid size prompt", 4, 4)]  # fmt: skip
    out = []
    for i, (pa, pb, va, vb) in enumerate(specs):
        voices = (r.integers(0, 2048, (8, va)), r.integers(0, 2048, (8, vb)))
        meta = {"scenario_id": f"s{i}", "scenario": f"s{i}", "category": "topic", "prompts": (pa, pb), "voices": voices,
                "conversation_s": CONV_S}  # fmt: skip
        out += [Segment(segment_id=f"s{i}:{'ab'[s]}", axis=AXIS_SELFPLAY, meta={**meta, "side": s}) for s in (0, 1)]
    return out


def quiet():
    return contextlib.redirect_stdout(open(os.devnull, "w"))


# ---- 1. reference parity -------------------------------------------------------------------------------------------
def reference(models_ab, convs, n_frames):
    """``converse_batched`` (the benchmark worker) on the same conversations: per side codes / text [C, ...]."""
    tok = StubTok()
    with quiet():
        ((ca, ta), (cb, tb)), _ = sp.converse_batched(
            *models_ab,
            voices_a=[c["voices"][0] for c in convs], texts_a=[encode_system_prompt(tok, c["prompts"][0]) or [] for c in convs],
            voices_b=[c["voices"][1] for c in convs], texts_b=[encode_system_prompt(tok, c["prompts"][1]) or [] for c in convs],
            n_frames=n_frames, silence_frames=SIL,
        )  # fmt: skip
    return (ca, ta), (cb, tb)


def gen_outputs(players, segs, mutant=None):
    G, C = 2, 6
    gen = ps.PpxSelfPlayGenerator(players, num_samples=G, convs_per_call=C, conversation_s=CONV_S)
    pairs = gen._pairs(segs)
    convs = [{"prompts": a.meta["prompts"], "voices": a.meta["voices"]} for a, _ in pairs for _ in range(G)]
    saved = sp._force_rows
    if mutant == "no_force":
        sp._force_rows = lambda *a, **k: None
    if mutant == "self_hear":
        for p in gen.players:
            orig = p.lm_gen.step

            def step(heard, _orig=orig, _p=p):
                if len(_p.sides) == 2:
                    heard = torch.cat([heard[C:], heard[:C]])  # each side hears its own previous output
                return _orig(heard)

            p.lm_gen.step = step
    try:
        o = gen._run(convs, int(round(CONV_S * FR)))
    finally:
        sp._force_rows = saved
    return gen, o, convs


def parity(players_fn, label, mutant=None):
    segs = scenario_pairs()
    gen, o, convs = gen_outputs(players_fn(), segs, mutant)
    n = int(round(CONV_S * FR))
    (ca, ta), (cb, tb) = reference((make_model(), make_model()), convs, n)
    md = gen.players[0].lm_gen.max_delay
    audio_ok, text_ok = True, True
    for s, (rc, rt) in enumerate(((ca, ta), (cb, tb))):
        for c in range(len(convs)):
            st, e = int(o["start"][c]), int(o["end"][c])
            mine = o["audio"][s][c, :, st:e].numpy()
            audio_ok &= bool(np.array_equal(mine, rc[c].astype(np.int64)))
            y = o["sampled"][s][c].numpy()
            # reference text at step t is LMGen's delayed output: the merged token of step t - max_delay, the free
            # sample from the row's prompt length L on
            L = int(o["lens"][s][c])
            k0 = max(0, L - (st - md))
            text_ok &= bool(np.array_equal(y[st - md + k0 : e - md], rt[c][k0:])) and k0 < e - st
    return audio_ok, text_ok


def check_parity():
    shared = lambda: [make_model()]  # noqa: E731
    two = lambda: [make_model(), make_model()]  # noqa: E731
    for label, fn in (("shared", shared), ("two players", two)):
        a, t = parity(fn, label)
        check(f"reference parity ({label}): audio", a)
        check(f"reference parity ({label}): text (shifted by max_delay)", t)
    for mutant in ("no_force", "self_hear"):
        a, t = parity(shared, "shared", mutant)
        check(f"reference parity can fail: mutant {mutant}", not (a and t), f"(audio {a}, text {t})")


# ---- 2. log-prob parity --------------------------------------------------------------------------------------------
def check_logprob():
    segs = scenario_pairs()
    torch.manual_seed(11)
    with quiet():
        gen = ps.PpxSelfPlayGenerator(
            [make_model(sampling=True)], num_samples=2, convs_per_call=6, conversation_s=CONV_S
        )
        import moshi_family.rl.ppx_selfplay as mod

        orig = mod.PpxSelfPlayGenerator._batch
        keep = []

        def _batch(self, seg, side, conv, rows, score=True):
            b = orig(self, seg, side, conv, rows, score)
            b.stats["_slp"] = torch.stack([r["sample_logp"] for r in rows])
            b.stats["_L"] = rows[0]["L"]
            keep.append(b)
            return b

        mod.PpxSelfPlayGenerator._batch = _batch
        try:
            batches = list(gen.iter_generate(segs))
        finally:
            mod.PpxSelfPlayGenerator._batch = orig
    check(
        "one batch per (scenario, side), in order",
        [b.segment.segment_id for b in batches] == [s.segment_id for s in segs],
    )
    worst, textrow, span, worst_mut = 0.0, True, True, {"shift": 0.0, "no_prompt": 0.0}
    for b in batches:
        L, mask = b.stats["_L"], b.token_mask
        x, y = b.codes[:, :TARGET_ROW], b.codes[:, TARGET_ROW]
        span &= bool((y[:, :L] == TEXT_PADDING_ID).all()) and not bool(mask[:, :L].any()) and bool(mask[:, L:].any())
        textrow &= bool(torch.equal(x[:, 0, L + 1 :], y[:, L:-1]))
        side = b.segment.meta["side"]
        with torch.no_grad():
            lp, _ = gen.text_logprobs(b.codes, side=side)
            worst = max(worst, float(((lp - b.stats["_slp"]).abs() * mask).max()))
            sh = b.codes.clone()
            sh[:, TARGET_ROW] = torch.roll(sh[:, TARGET_ROW], 1, dims=1)
            lp2, _ = gen.text_logprobs(sh, side=side)
            worst_mut["shift"] = max(worst_mut["shift"], float(((lp2 - b.stats["_slp"]).abs() * mask).max()))
            cut = b.codes[:, :, L:]
            lp3, _ = gen.text_logprobs(cut, side=side)
            d = (lp3 - b.stats["_slp"][:, L:]).abs() * mask[:, L:]
            worst_mut["no_prompt"] = max(worst_mut["no_prompt"], float(d.max()))
        # the batch's own fns are the ones the trainer uses
        with torch.no_grad():
            lpb, _ = b.logprob_fn(b.codes)
        worst = max(worst, float(((lpb - b.stats["_slp"]).abs() * mask).max()))
    check("log-prob parity (teacher-forced == sampler)", worst < 1e-4, f"max |diff| = {worst:.2e}")
    check("text row: input t+1 == sampled t over the trained span", textrow)
    check("trained span starts at L (the prompt's length), PAD targets before", span)
    for k, v in worst_mut.items():
        check(f"log-prob parity can fail: mutant {k}", v > 1e-3, f"({v:.2e})")
    return gen, batches


# ---- 3. routing through the real trainer ---------------------------------------------------------------------------
def check_routing():
    from moshi_family.rl.grpo import GRPOConfig
    from moshi_family.rl.trainer import RLTrainer, RLTrainerConfig

    torch.manual_seed(12)
    pa, pb = make_model(sampling=True), make_model(sampling=True, perturb=0.01)
    results = {}
    for mutant in (None, "route_all_to_a"):
        with quiet():
            gen = ps.PpxSelfPlayGenerator([pa, pb], num_samples=2, convs_per_call=2, conversation_s=CONV_S)
            if mutant:
                gen.side_player = {0: (0, 0), 1: (0, 0)}
            batches = list(gen.iter_generate(scenario_pairs()[:2]))
        params = [p for m in (pa, pb) for p in m.lm_gen.lm_model.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=0.0)
        with tempfile.TemporaryDirectory() as d:
            tr = RLTrainer(
                core=pa.lm_gen.lm_model, generator=gen, reward_registry={}, ref_logprobs=None, optimizer=opt,
                base_lrs=[0.0], trainable_params=params, segment_source=None, save_fn=lambda *a: None,
                cfg=RLTrainerConfig(overlap_scoring=False, profile=False), grpo_cfg=GRPOConfig(), out_dir=d,
                log=lambda *_: None,
            )  # fmt: skip
            got = {}
            for b in batches:
                opt.zero_grad(set_to_none=True)
                comps = [{"adherence": float(k)} for k in range(len(b.rollouts))]
                tr._grpo_update("selfplay", b, comps)
                ga = sum(float(p.grad.abs().sum()) for p in pa.lm_gen.lm_model.parameters() if p.grad is not None)
                gb = sum(float(p.grad.abs().sum()) for p in pb.lm_gen.lm_model.parameters() if p.grad is not None)
                got[b.segment.meta["side"]] = (ga, gb)
        results[mutant] = got
    ok = results[None][0][0] > 0 and results[None][0][1] == 0 and results[None][1][0] == 0 and results[None][1][1] > 0
    check(
        "two players: each side's update reaches only its own player (real RLTrainer._grpo_update)",
        ok,
        str(results[None]),
    )
    m = results["route_all_to_a"]
    check("routing can fail: mutant route_all_to_a", not (m[1][0] == 0 and m[1][1] > 0), str(m))


# ---- 4. reward -----------------------------------------------------------------------------------------------------
def _i6_namespace():
    """The i6 originals, cut out by AST (``selfplay_bench`` / ``selfplay_judge`` import sisyphus)."""
    ns = {"json": json, "hashlib": __import__("hashlib")}
    for fname, names in (
        ("selfplay_judge.py", ("EMPTY_TRANSCRIPT", "two_sided_transcript")),
        ("selfplay_bench.py", ("SPJ2_METRICS", "SPJ2_REASON_CHARS", "SPJ2_RUBRICS", "_ITEM_SCHEMA", "SPJ2_JSON_SCHEMA",
                               "render_judge_message", "parse_judge_reply")),
    ):  # fmt: skip
        src = open(os.path.join(I6, fname)).read()
        tree = ast.parse(src)
        for node in tree.body:
            tgt = node.name if isinstance(node, ast.FunctionDef) else (
                node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else
                node.target.id if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else None)  # fmt: skip
            if tgt in names:
                exec(compile(ast.Module([node], []), fname, "exec"), ns)
    return ns


class StubJudge(SelfPlayJudgeReward):
    def __init__(self, rubric, fail_keys=()):
        super().__init__(judge_model="stub", judge_base_url="http://none", rubric=rubric)
        self.calls, self.fail_keys = [], set(fail_keys)

    def judge_text(self, prompt_a, prompt_b, transcript, dur, n_a, n_b):
        self.calls.append(transcript)
        if any(k in transcript for k in self.fail_keys):
            return {"judged": None, "judge_raw": None, "judge_retries": 5, "error": "stub failure"}
        h = sum(map(ord, transcript))
        sc = {"adherence_a": 1 + h % 5, "adherence_b": 1 + (h // 5) % 5, "coherence": 1 + (h // 25) % 5,
              "responsiveness": 1 + (h // 125) % 5}  # fmt: skip
        return {"judged": {m: {"score": v, "reason": "r"} for m, v in sc.items()}, "judge_raw": "{}", "judge_retries": 0,
                "error": ""}  # fmt: skip


def check_reward(gen, batches):
    ns = _i6_namespace()
    rubric = {"text": ns["SPJ2_RUBRICS"]["spj2_judge_v1"], "schema": ns["SPJ2_JSON_SCHEMA"], "metrics": list(ns["SPJ2_METRICS"]),
              "reason_chars": ns["SPJ2_REASON_CHARS"]}  # fmt: skip
    # copies == originals
    wa = [
        {"text": "hello", "start": 0.2, "end": 0.5},
        {"text": "there", "start": 0.6, "end": 0.9},
        {"text": "late", "start": 5.0, "end": 5.3},
        {"text": "out", "start": 61.0, "end": 61.2},
    ]
    wb = [{"text": "hi", "start": 0.4, "end": 0.6}, {"text": "{x}", "start": 3.0, "end": 3.2}]  # fmt: skip
    same = all(
        two_sided_transcript(a, b, 60.0) == ns["two_sided_transcript"](a, b, 60.0)
        for a, b in ((wa, wb), ([], []), (wb, wa))
    )
    check("two_sided_transcript == selfplay_judge's", same)
    kw = dict(
        prompt_a="Be {n_a} A.", prompt_b="B", duration=59.6, n_a=3, n_b=2, transcript=two_sided_transcript(wa, wb, 60.0)
    )
    check(
        "render_message == render_judge_message",
        render_message(rubric["text"], **kw) == ns["render_judge_message"]("spj2_judge_v1", **kw),
    )
    good = json.dumps({m: {"reason": " a  b ", "score": 3} for m in rubric["metrics"]})
    same = parse_reply("x" + good + "y", rubric["metrics"], rubric["reason_chars"]) == ns["parse_judge_reply"](
        "x" + good + "y"
    )
    bad_ok = True
    for bad in (
        good.replace('"score": 3', '"score": 6', 1),
        good.replace('"score": 3', '"score": true', 1),
        "{}",
        "no json",
    ):
        e1 = e2 = None
        try:
            parse_reply(bad, rubric["metrics"], rubric["reason_chars"])
        except ValueError as e:
            e1 = type(e)
        try:
            ns["parse_judge_reply"](bad)
        except ValueError as e:
            e2 = type(e)
        bad_ok &= e1 is not None and e1 == e2
    check("parse_reply == parse_judge_reply (good and bad replies)", same and bad_ok)

    # per-side components, one call per conversation
    rw = StubJudge(rubric)
    a, b = batches[0], batches[1]
    ca, cb = rw(a.segment, a.rollouts), rw(b.segment, b.rollouts)
    check(
        "one judge call per conversation (side B reads side A's)",
        len(rw.calls) == len(a.rollouts),
        f"{len(rw.calls)} calls",
    )
    ok = True
    for x, y, ra in zip(ca, cb, a.rollouts):
        j = ra.conv["judged"]["judged"]
        ok &= x == {"adherence": j["adherence_a"]["score"], "coherence": j["coherence"]["score"],
                    "responsiveness": j["responsiveness"]["score"]}  # fmt: skip
        ok &= y == {"adherence": j["adherence_b"]["score"], "coherence": j["coherence"]["score"],
                    "responsiveness": j["responsiveness"]["score"]}  # fmt: skip
    check("side A gets adherence_a, side B adherence_b, both the shared scores", ok)
    # a failed conversation -> the group mean
    for r in a.rollouts:
        r.conv["judged"] = None
    tr0 = a.rollouts[0].conv
    rw2 = StubJudge(rubric)
    rw2.fail_keys = {two_sided_transcript(*tr0["words"], tr0["dur"])} if tr0["words"][0] or tr0["words"][1] else set()
    if rw2.fail_keys:
        comps = rw2(a.segment, a.rollouts)
        okc = [c for r, c in zip(a.rollouts, comps) if not r.info["judge_failed"]]
        mean = {k: sum(c[k] for c in okc) / len(okc) for k in okc[0]} if okc else None
        fails = [c for r, c in zip(a.rollouts, comps) if r.info["judge_failed"]]
        check("failed judge call -> the group mean (zero advantage)", fails and all(f == mean for f in fails))
    else:
        print("[skip] failure-mean check: the first conversation is silent")


# ---- 5. source -----------------------------------------------------------------------------------------------------
def check_source():
    from datasets import Dataset

    with tempfile.TemporaryDirectory() as d:
        rows = [{"id": f"sptr:topic:n{i}", "scenario": f"n{i}", "category": "topic", "level": "general",
                 "context": [f"pa{i}"], "context_other": [f"pb{i}"], "voice_codes": np.zeros(16, np.int16),
                 "voice_codes_other": np.ones(16, np.int16)} for i in range(30)]  # fmt: skip
        Dataset.from_list(rows).save_to_disk(os.path.join(d, "ds"))
        man = os.path.join(d, "m.json")
        json.dump({"scenarios": os.path.join(d, "ds"), "n_probe": 4, "seed": 0}, open(man, "w"))
        src = ps.SelfPlayScenarioSource(man, length=(20.0, 60.0, 40), draws_per_step=4, log=lambda *_: None)
        draws = [src.sample(AXIS_SELFPLAY) for _ in range(44)]
        pairs_ok = all(a.meta["scenario_id"] == b.meta["scenario_id"] and (a.meta["side"], b.meta["side"]) == (0, 1)
                       for a, b in zip(draws[0::2], draws[1::2]))  # fmt: skip
        lens = [s.meta["conversation_s"] for s in draws]
        step_const = all(len(set(lens[i : i + 4])) == 1 for i in range(0, 44, 4))
        check("source: A then B of one scenario per two draws", pairs_ok)
        check("source: length constant within a step, 20 -> 60 s, capped",
              step_const and lens[0] == 20.0 and lens[40] == 60.0 and lens[20] == 40.0 and max(lens) == 60.0, str(lens[::4]))  # fmt: skip
        check("source: probe at the end length", all(s.meta["conversation_s"] == 60.0 for s in src.probe_segments()))
        src2 = ps.SelfPlayScenarioSource(man, length=(20.0, 60.0, 40), draws_per_step=4, log=lambda *_: None)
        src2.seek(20)
        nxt = src2.sample(AXIS_SELFPLAY)
        check("source: seek resumes at the same draw", nxt.segment_id == draws[20].segment_id
              and nxt.meta["conversation_s"] == draws[20].meta["conversation_s"])  # fmt: skip


# ---- 6. early stop ------------------------------------------------------------------------------------------------
def check_early_stop():
    """``stop_silent_s``: a call whose conversations are all silent stops early; the rows are cut there and still score
    under the distribution they were sampled from; the probe runs the whole slot. Silence is forced by counting every
    text token as silent (``TEXT_SPECIAL_IDS`` patched). CAN FAIL: stop off (0) or real speech -> no early stop."""
    import dataclasses

    slot = 4.0  # 50 frames: room for a 1 s stop after the prompts
    segs = [dataclasses.replace(s, meta={**s.meta, "conversation_s": slot}) for s in scenario_pairs()]

    def run(stop_s, all_silent):
        torch.manual_seed(3)
        with quiet():
            gen = ps.PpxSelfPlayGenerator(
                [make_model(sampling=True)], num_samples=2, convs_per_call=6, conversation_s=slot, stop_silent_s=stop_s
            )
        saved = ps.TEXT_SPECIAL_IDS
        if all_silent:
            ps.TEXT_SPECIAL_IDS = frozenset(range(KW["text_card"] + 8))
        orig, keep = ps.PpxSelfPlayGenerator._batch, []

        def _batch(self, seg, side, conv, rows, score=True):
            b = orig(self, seg, side, conv, rows, score)
            b.stats["_slp"] = torch.stack([r["sample_logp"] for r in rows])
            keep.append(b)
            return b

        ps.PpxSelfPlayGenerator._batch = _batch
        try:
            with quiet():
                batches = list(gen.iter_generate(segs))
                probe = gen.probe_conversations(segs, 2)
        finally:
            ps.TEXT_SPECIAL_IDS = saved
            ps.PpxSelfPlayGenerator._batch = orig
        return gen, batches, probe

    gen, batches, probe = run(1.0, True)
    convs = [r.conv for b in batches for r in b.rollouts]
    check("early stop: every all-silent call stopped", all(c["stopped"] for c in convs))
    check(
        "early stop: judged span cut, planned length kept",
        all(c["slot_s"] == slot and c["dur"] < slot - 1.0 for c in convs),
        f"dur {sorted({round(c['dur'], 2) for c in convs})}",
    )
    check("early stop: the batch reports it", all(b.stats["stopped_early"] == 1.0 for b in batches))
    worst = 0.0
    for b in batches:
        with torch.no_grad():
            lp, _ = gen.text_logprobs(b.codes, side=b.segment.meta["side"])
        worst = max(worst, float(((lp - b.stats["_slp"]).abs() * b.token_mask).max()))
    n_tok = sum(int(b.token_mask.sum()) for b in batches)
    check(
        "early stop: cut rows keep log-prob parity",
        worst < 1e-4 and n_tok > 0,
        f"max |diff| = {worst:.2e} over {n_tok} tokens",
    )
    pc = [r.conv for _, ra, _, rb in probe for r in ra + rb]
    check("early stop: the probe runs the whole slot", all(not c["stopped"] and c["dur"] == slot for c in pc))
    _, b_off, _ = run(0.0, True)
    check("early stop can fail: off (0) -> no stop", not any(r.conv["stopped"] for b in b_off for r in b.rollouts))
    _, b_talk, _ = run(1.0, False)
    check("early stop can fail: speech -> no stop", not any(r.conv["stopped"] for b in b_talk for r in b.rollouts))


def main():
    check_parity()
    gen, batches = check_logprob()
    check_routing()
    check_reward(gen, batches)
    check_source()
    check_early_stop()
    if FAIL:
        print("\nFAILED:\n  " + "\n  ".join(FAIL))
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
