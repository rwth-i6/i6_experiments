"""Guard: the 2026-09-24 corpus-build rules -- short-intruder slicing and window start jitter.

WHY. (a) DuplexChat's ``_two_speaker_runs`` CLOSES a run at ANY third label, so a 0.3 s "yeah" from a
third voice cuts a long two-person dialogue into pieces; ``podcast_slice_main.drop_short_intruders``
drops such a segment (< ``min_intruder_sec`` AND not one of the two locally dominant speakers) BEFORE
their unchanged extractor (``PodcastDialogueSlice(min_intruder_sec=)``). (b) ``select_windows`` cuts at
run starts, so t = 0 of every window is exactly someone's turn onset; ``start_jitter_sec`` moves each
start earlier by u <= the silence before the cut point, seeded by the row id. Checked:
  1. drop_short_intruders + THEIR extractor (pinned clone): a 0.6 s third-speaker blip used to split a
     dialogue and no longer does; a 2 s third-speaker segment still splits; a 0.5 s backchannel of a
     main speaker is never dropped; min_sec=0 returns the list unchanged.
  2. THE REAL SLICER on a one-episode part: off -> the blip splits the episode (2 rows); on -> 1 row with
     the blip recorded in channel_info["dropped_intruders"] and counted (segments, words) in
     summary.json; --min_intruder_sec 0 is bit-identical to no flag at all.
  3. select_windows jitter on random turn lists: jitter 0 == the old call; every jittered start lies in
     silence (the added span [start', start) overlaps no turn), u <= silence before the cut point, u <= start_jitter_sec, length <=
     max_sec, same windows otherwise, deterministic per seed_key, different across keys; the share of
     windows whose first turn onset is < 0.08 s drops.
  4. The REAL converter (PodcastCodesTrainData) with a jitter rule: codes are the frame slice from the
     JITTERED start and alignments are re-based to it.
  5. Wiring: the Job knob (default 0.0, hash-excluded only at 0.0, forwarded), WINDOW_SELECT_KEYS, and
     both select_windows callers pass seed_key=item_id (else PodcastWindowContext would recompute
     different windows than the converter wrote).
  6. Shown able to fail: a drop rule without the dominant-speaker test drops the main speaker's
     backchannel ([1] fails); a jitter without the silence cap adds speech ([3] fails); a pre-roll
     without the silence cap reaches into the previous segment ([7] fails).
  7. Row pre-roll (``PodcastDialogueSlice(pre_roll_sec=)``, ``pre_roll_start``): on random segment lists
     the new start only adds silence (>= PRE_ROLL_MARGIN_SEC after the latest earlier segment end, never
     < 0, u <= pre_roll_sec), no move when segments overlap the start, deterministic per key; the REAL
     slicer on an episode whose dialogues follow 3 s / 8 s of silence moves both starts back by
     0 < u <= 1 s, keeps the codes the exact frame slice, re-bases turns and words to the new start,
     records ``pre_roll_sec`` in channel_info and summary.json, and ``--pre_roll_sec 0`` is
     bit-identical to no flag.

Setup root, setup .venv, CPU, ~30 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_slicing_rules_2026_09_24.py
"""

import ast
import glob
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
from datasets import Dataset, Sequence, load_from_disk  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import (  # noqa: E402
    WINDOW_SELECT_KEYS,
    PodcastCodesTrainData,
    select_windows,
)
from moshi_family.podcast_mimi_encode import part_features  # noqa: E402
from moshi_family.podcast_slice_main import PRE_ROLL_MARGIN_SEC, drop_short_intruders, pre_roll_start  # noqa: E402

(DUPLEXCHAT,) = sorted(glob.glob("work/i6_core/tools/git/CloneGitRepositoryJob.*/output/DuplexChat/src"))[:1]
sys.path.insert(0, DUPLEXCHAT)
from duplexchat_pipe.dialogue import extract_valid_dialogues  # noqa: E402


def seg(spk, a, b):
    return {"speaker": spk, "start": float(a), "end": float(b)}


def alt(n, step, t0=0.0, gap=0.0, spk=("A", "B")):
    return [seg(spk[i % 2], t0 + i * step, t0 + (i + 1) * step - gap) for i in range(n)]


# A/B alternate 5 s turns over [0, 60); a C blip sits inside B's turn at 30.2.
BASE = alt(12, 5.0)
BLIP = sorted(BASE + [seg("C", 30.2, 30.8)], key=lambda s: s["start"])
LONG_C = sorted(BASE + [seg("C", 30.2, 32.2)], key=lambda s: s["start"])
BACKCH = sorted(BASE + [seg("A", 27.0, 27.5)], key=lambda s: s["start"])  # A's 0.5 s "mm" inside B's turn


def spans(dlgs):
    return [(round(d.start, 2), round(d.end, 2)) for d in dlgs]


def check_intruders(drop):
    """[1] on an implementation ``drop(segments, min_sec) -> (kept, dropped)``; raises on a wrong one."""
    assert spans(extract_valid_dialogues(BLIP)) != [(0.0, 60.0)], "fixture: the blip must split without the rule"
    kept, dropped = drop(BLIP, 1.0)
    assert [s["speaker"] for s in dropped] == ["C"], dropped
    assert spans(extract_valid_dialogues(kept)) == [(0.0, 60.0)], spans(extract_valid_dialogues(kept))
    kept, dropped = drop(LONG_C, 1.0)
    assert not dropped and spans(extract_valid_dialogues(kept)) != [(0.0, 60.0)], "a 2 s intruder must still split"
    kept, dropped = drop(BACKCH, 1.0)
    assert not dropped and len(kept) == len(BACKCH), f"a main speaker's backchannel was dropped: {dropped}"
    kept, dropped = drop(BLIP, 0.0)
    assert kept == BLIP and not dropped, "min_sec=0 must be a no-op"


check_intruders(drop_short_intruders)
print(
    f"[1] drop_short_intruders + DuplexChat's extractor: 0.6 s blip -> "
    f"{spans(extract_valid_dialogues(BLIP))} becomes [(0.0, 60.0)]; 2 s intruder still splits; "
    "0.5 s main-speaker backchannel kept; min_sec=0 no-op"
)

# ---- 2. the real slicer ---------------------------------------------------------------------------
FR, K, DUR = 12.5, 8, 60.0
F = int(DUR * FR)
rng = np.random.default_rng(0)


def words_in(sps):
    out = []
    for a, b in sps:
        t = a + 0.1
        while t + 0.3 <= b:
            out.append({"text": "w", "start": round(t, 3), "end": round(t + 0.3, 3)})
            t += 0.5
    return out


asr = {
    "backend": "test",
    "model": "test",
    "channels": [
        {
            "channel": 0,
            "speaker": "A",
            "words": words_in([(s["start"], s["end"]) for s in BASE if s["speaker"] == "A"]),
        },
        {
            "channel": 1,
            "speaker": "B",
            "words": words_in([(s["start"], s["end"]) for s in BASE if s["speaker"] == "B"] + [(30.2, 30.8)]),
        },
    ],
}
_defaults = {"string": "", "binary": b"", "float32": 0.0, "int32": 0, "int64": 0, "int16": 0}
feats = part_features("episode")
ep = {n: ([] if isinstance(t, Sequence) else _defaults[t.dtype]) for n, t in feats.items()}
codes = rng.integers(0, 2048, (2, K, F)).astype(np.int16)
ep.update(
    item_id="ep",
    episode_id="ep",
    source="rss",
    feed="feed",
    span_start_sec=0.0,
    duration_sec=DUR,
    n_codebooks=K,
    n_frames=F,
    frame_rate=FR,
    sample_rate=24000,
    asr_json=json.dumps(asr),
    codes_a=codes[0].reshape(-1),
    codes_b=codes[1].reshape(-1),
    diar_turns_json=json.dumps(BLIP),
)
tmp = tempfile.mkdtemp(prefix="check_slicing_rules_")
ep_dir = os.path.join(tmp, "episodes")
Dataset.from_dict({n: [v] for n, v in ep.items()}, features=feats).save_to_disk(
    os.path.join(ep_dir, "parts", "part_00000")
)


def run_slicer(tag, *extra):
    out = os.path.join(tmp, tag)
    os.makedirs(out)
    subprocess.run(
        [
            sys.executable,
            os.path.abspath("recipe/speech_llm/full_duplex/moshi_family/podcast_slice_main.py"),
            "--in_dir",
            ep_dir,
            "--out_dir",
            out,
            "--repo_src",
            os.path.abspath(DUPLEXCHAT),
            "--min_dialogues_per_episode",
            "0",
            *extra,
        ],
        check=True,
        cwd=out,
        env={**os.environ, "PYTHONPATH": os.path.abspath("recipe/speech_llm/full_duplex")},
        stdout=subprocess.DEVNULL,
    )
    rows = [r for p in sorted(glob.glob(os.path.join(out, "parts", "part_*"))) for r in load_from_disk(p)]
    return rows, json.load(open(os.path.join(out, "summary.json")))


rows_none, sum_none = run_slicer("none")
rows_zero, sum_zero = run_slicer("zero", "--min_intruder_sec", "0")
rows_on, sum_on = run_slicer("on", "--min_intruder_sec", "1.0")
assert len(rows_none) >= 2, f"fixture: the blip must split the episode, got {len(rows_none)} rows"
assert rows_zero == rows_none and sum_zero == sum_none, "--min_intruder_sec 0 must be bit-identical to no flag"
assert "short_intruders" not in sum_none and "min_intruder_sec" not in sum_none["filters"], sum_none
assert len(rows_on) == 1 and abs(rows_on[0]["duration_sec"] - 60.0) < 0.1, [r["duration_sec"] for r in rows_on]
ci = json.loads(rows_on[0]["channel_info"])
assert ci["dropped_intruders"] == [["C", 30.2, 30.8]], ci
assert json.loads(rows_on[0]["turns_json"]) and {t["speaker"] for t in json.loads(rows_on[0]["turns_json"])} == {
    "A",
    "B",
}
si = sum_on["short_intruders"]
all_words = [w for c in asr["channels"] for w in c["words"]]
want_inside = sum(30.2 <= 0.5 * (w["start"] + w["end"]) <= 30.8 for w in all_words)
assert want_inside >= 1, "fixture: some ASR words must sit inside the blip"
assert si["n_dropped_segments"] == 1 and si["n_rows_containing"] == 1 and si["words_inside"] == want_inside, si
assert sum_on["filters"]["min_intruder_sec"] == 1.0
print(
    f"[2] real slicer: no flag -> {len(rows_none)} rows (== --min_intruder_sec 0, bit-identical); 1.0 -> "
    f"{len(rows_on)} row of {rows_on[0]['duration_sec']:.1f} s, intruder recorded, summary {si}"
)

# ---- 3. select_windows start jitter ---------------------------------------------------------------
RULE = dict(min_sec=20.0, max_sec=60.0, min_turns=3, max_stretch_sec=30.0, turn_min_sec=1.0)
JIT = dict(RULE, start_jitter_sec=0.5)


def random_turns(r):
    ts, t = [], float(r.uniform(0, 1.0))
    for i in range(int(r.integers(20, 60))):
        d = float(r.uniform(0.3, 8.0))
        spk = "AB"[i % 2] if r.random() > 0.1 else "AB"[(i + 1) % 2]
        ts.append(seg(spk, t, t + d))
        t += d + float(r.choice([r.uniform(-0.5, 0.0), r.uniform(0.0, 0.2), r.uniform(0.2, 2.0)]))
    return ts, t + 0.5


def check_jitter(sel, n_rows=300):
    """[3] on an implementation ``sel(turns, dur, seed_key=, **rule)``; returns first-onset shares."""
    r = np.random.default_rng(1)
    first0, first1, n = 0, 0, 0
    for k in range(n_rows):
        turns, dur = random_turns(r)
        old = select_windows(turns, dur, **RULE)
        assert sel(turns, dur, seed_key=f"row{k}", **dict(RULE, start_jitter_sec=0.0)) == old, (
            "jitter 0 changed windows"
        )
        new = sel(turns, dur, seed_key=f"row{k}", **JIT)
        assert new == sel(turns, dur, seed_key=f"row{k}", **JIT), "not deterministic"
        assert len(new) == len(old) and [b for _, b in new] == [b for _, b in old], "ends/count changed"
        for (a, b), (a2, _b2) in zip(old, new):
            u = a - a2
            assert -1e-9 <= u <= 0.5 + 1e-9 and b - a2 <= RULE["max_sec"] + 1e-9 and a2 >= -1e-9, (a, a2, b)
            # The ADDED span [a2, a) must hold no speech. (The cut point a itself can sit inside ANOTHER
            # speaker's overlapping turn -- that is the old rule's start, and then u must be 0.)
            assert u <= 1e-12 or not any(t["start"] < a and t["end"] > a2 for t in turns), (
                f"jittered start {a2} (cut {a}) adds speech"
            )
            before = [t["end"] for t in turns if t["start"] < a]
            silence = a - max(before) if before else a
            assert u <= max(0.0, silence) + 1e-9, (u, silence)
            first = min((t["start"] for t in turns if t["start"] >= a - 1e-9), default=None)
            if first is not None:
                n += 1
                first0 += first - a < 0.08
                first1 += first - a2 < 0.08
    # different keys give different draws somewhere
    turns, dur = random_turns(np.random.default_rng(5))
    assert any(sel(turns, dur, seed_key="x", **JIT) != sel(turns, dur, seed_key=f"y{i}", **JIT) for i in range(5)), (
        "seed_key has no effect"
    )
    return first0 / n, first1 / n, n


s0, s1, nw = check_jitter(select_windows)
assert s1 < s0, (s0, s1)
print(
    f"[3] start jitter on 300 random rows ({nw} windows): starts in silence, u <= silence and <= 0.5 s, "
    f"deterministic per seed_key; first turn onset < 0.08 s: {s0:.0%} -> {s1:.0%}; jitter 0 == old"
)

# ---- 4. the real converter with a jitter rule -----------------------------------------------------
K2, FR2, DUR2 = 8, 12.5, 130.0
F2 = int(DUR2 * FR2)
codes2 = (np.arange(K2)[:, None] * 100 + np.arange(F2)[None, :]).astype(np.int16)
turns2 = alt(24, 5.0, t0=1.0, gap=1.0)  # 4 s turns with 1 s silences: room for the full 0.5 s
JIT4 = dict(JIT, max_sec=58.0)  # windows of 55 s, so max_sec does not cap the jitter
words2 = [
    {"text": f"a{i}", "start": t["start"] + 0.1, "end": t["start"] + 0.6, "speaker": "A"} for i, t in enumerate(turns2)
]
row2 = {
    "item_id": "ep#7",
    "episode_id": "ep",
    "duration_sec": DUR2,
    "n_codebooks": K2,
    "n_frames": F2,
    "frame_rate": FR2,
    "codes_a": codes2.reshape(-1).tolist(),
    "codes_b": (codes2 + 7).reshape(-1).tolist(),
    "words_a": json.dumps(words2),
    "words_b": json.dumps(words2),
    "turns_json": json.dumps(turns2),
}
d2 = tempfile.mkdtemp(prefix="check_jitter_conv_")
Dataset.from_list([row2]).save_to_disk(os.path.join(d2, "in", "parts", "part_000"))
job = PodcastCodesTrainData(
    shard_dirs=[tk.Path(os.path.join(d2, "in"))], assistant_channel="a", min_assistant_words=1, window_select=JIT4
)
job.out_dir = tk.Path(os.path.join(d2, "out"))
cwd = os.getcwd()
os.chdir(d2)
try:
    job.run()
finally:
    os.chdir(cwd)
out = load_from_disk(os.path.join(d2, "out"))
wins = select_windows(turns2, DUR2, **JIT4, seed_key="ep#7")
plain = select_windows(turns2, DUR2, **dict(RULE, max_sec=58.0))
assert len(out) == len(wins) >= 2 and any(abs(a - p) > 0.05 for (a, _), (p, _) in zip(wins, plain)), (wins, plain)
for r, (a, b) in zip(out, wins):
    f0, f1 = int(np.floor(a * FR2)), min(F2, int(np.ceil(b * FR2)))
    assert np.array_equal(np.asarray(r["codes_assistant"], dtype=np.int16), codes2[:, f0:f1].reshape(-1)), r["id"]
    for al in r["alignments"]:
        i = int(al["text"][1:])
        assert abs(al["start"] - (words2[i]["start"] - f0 / FR2)) < 1e-4 and al["start"] >= 0, (al, a)
print(
    f"[4] converter: {len(out)} windows cut from the JITTERED starts {[round(a, 2) for a, _ in wins]} (plain {[p for p, _ in plain]}), alignments re-based"
)

# ---- 5. wiring --------------------------------------------------------------------------------------
src = open("recipe/i6_experiments/users/dorian_koch/speech_llm/podcast_ingest.py").read()
cls = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "PodcastDialogueSlice")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
kw = {a.arg: d for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults)}
assert ast.literal_eval(kw["min_intruder_sec"]) == 0.0
ex = ast.literal_eval(
    next(
        n for n in cls.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__sis_hash_exclude__"
    ).value
)
assert ex.get("min_intruder_sec", "missing") == 0.0, ex
run_src = ast.get_source_segment(src, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run"))
assert '"--min_intruder_sec"' in run_src, "run() does not forward min_intruder_sec"
assert ast.literal_eval(kw["pre_roll_sec"]) == 0.0 and ex.get("pre_roll_sec", "missing") == 0.0, ex
assert '"--pre_roll_sec"' in run_src, "run() does not forward pre_roll_sec"
assert "start_jitter_sec" in WINDOW_SELECT_KEYS
pp = open("recipe/i6_experiments/users/dorian_koch/speech_llm/persona_prompts.py").read()
for name, text in (("podcast_ingest", src), ("persona_prompts", pp)):
    calls = [
        n
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "select_windows"
    ]
    assert calls and all(any(k.arg == "seed_key" for k in c.keywords) for c in calls), (
        f"{name}: select_windows without seed_key"
    )
print(
    "[5] wiring: min_intruder_sec default 0.0, excluded only at 0.0, forwarded; start_jitter_sec a window key; seed_key passed by both callers"
)


# ---- 6. shown able to fail ----------------------------------------------------------------------------
def drop_any_short(segments, min_sec):
    """MUTANT: drops every short segment, ignoring who is locally dominant."""
    if min_sec <= 0:
        return list(segments), []
    k = [s for s in segments if s["end"] - s["start"] >= min_sec]
    return k, [s for s in segments if s["end"] - s["start"] < min_sec]


def jitter_no_silence_cap(turns, dur, seed_key="", start_jitter_sec=0.0, **rule):
    """MUTANT: always moves the start back by start_jitter_sec, into the previous turn if need be."""
    w = select_windows(turns, dur, **rule)
    return [(max(0.0, a - start_jitter_sec), b) for a, b in w] if start_jitter_sec > 0 else w


for name, fn, arg in (
    ("drop_any_short", check_intruders, drop_any_short),
    ("jitter_no_silence_cap", check_jitter, jitter_no_silence_cap),
):
    try:
        fn(arg)
    except AssertionError as e:
        print(f"[6] mutant {name} caught: {str(e)[:80]}")
    else:
        raise AssertionError(f"guard is vacuous: mutant {name} passed")


# ---- 7. row pre-roll ------------------------------------------------------------------------------------
def check_preroll(fn):
    """[7] on an implementation ``fn(start, segments, pre_roll_sec, seed_key) -> new start``."""
    r = np.random.default_rng(3)
    moved = 0
    for k in range(2000):
        segs, t = [], float(r.uniform(0, 3))
        for _ in range(int(r.integers(1, 8))):
            d = float(r.uniform(0.2, 6))
            segs.append(seg("AB"[int(r.integers(2))], t, t + d))
            t += d + float(r.choice([r.uniform(-1, 0), r.uniform(0, 0.3), r.uniform(0.3, 3)]))
        start = max(0.0, t)
        new = fn(start, segs, 1.0, f"k{k}")
        assert new == fn(start, segs, 1.0, f"k{k}"), "not deterministic"
        assert fn(start, segs, 0.0, f"k{k}") == start, "pre_roll 0 must be a no-op"
        ends = [s["end"] for s in segs if s["start"] < start - 1e-9]
        prev_end = max(ends) if ends else 0.0
        assert -1e-9 <= start - new <= 1.0 + 1e-9 and new >= -1e-9, (start, new)
        if new < start:
            moved += 1
            assert new >= prev_end + PRE_ROLL_MARGIN_SEC - 1e-9 or not ends, (start, new, prev_end)
        if ends and prev_end >= start:
            assert new == start, "moved although a segment overlaps the start"
    assert moved > 200, moved
    return moved


moved = check_preroll(pre_roll_start)

# the real slicer: two dialogues after 3 s (episode start) and 8 s of silence
SHIFT, GAP2 = 3.0, 8.0
blockA = alt(12, 5.0, t0=SHIFT)
blockB = alt(12, 5.0, t0=SHIFT + 60 + GAP2)
DUR3 = SHIFT + 60 + GAP2 + 60 + 2
F3 = int(DUR3 * FR)
ep3 = dict(ep)
ep3.update(
    item_id="ep3",
    episode_id="ep3",
    duration_sec=DUR3,
    n_frames=F3,
    diar_turns_json=json.dumps(blockA + blockB),
    asr_json=json.dumps(
        {
            "backend": "test",
            "model": "test",
            "channels": [
                {
                    "channel": c,
                    "speaker": spk,
                    "words": words_in([(x["start"], x["end"]) for x in blockA + blockB if x["speaker"] == spk]),
                }
                for c, spk in ((0, "A"), (1, "B"))
            ],
        }
    ),
)
codes3 = rng.integers(0, 2048, (2, K, F3)).astype(np.int16)
ep3["codes_a"], ep3["codes_b"] = codes3[0].reshape(-1), codes3[1].reshape(-1)
ep3_dir = os.path.join(tmp, "episodes3")
Dataset.from_dict({n: [v] for n, v in ep3.items()}, features=feats).save_to_disk(
    os.path.join(ep3_dir, "parts", "part_00000")
)


def run_slicer_on(in_dir, tag, *extra):
    global ep_dir
    keep, ep_dir = ep_dir, in_dir
    try:
        return run_slicer(tag, *extra)
    finally:
        ep_dir = keep


r_none, s_none = run_slicer_on(ep3_dir, "pre_none")
r_zero, s_zero = run_slicer_on(ep3_dir, "pre_zero", "--pre_roll_sec", "0")
r_pre, s_pre = run_slicer_on(ep3_dir, "pre_on", "--pre_roll_sec", "1.0")
assert r_zero == r_none and s_zero == s_none, "--pre_roll_sec 0 must be bit-identical to no flag"
assert len(r_none) == len(r_pre) == 2, (len(r_none), len(r_pre))
for a0, b0 in zip(r_none, r_pre):
    ci = json.loads(b0["channel_info"])
    u = ci["pre_roll_sec"]
    dstart = ci["dialogue_start_sec"]
    assert 0 < u <= 1.0 and dstart in (SHIFT, SHIFT + 60 + GAP2), ci
    t0 = float(b0["span_start_sec"])
    assert t0 <= float(a0["span_start_sec"]) - 0.5 / FR or u < 1 / FR, (t0, a0["span_start_sec"])
    prev_end = 0.0 if dstart == SHIFT else SHIFT + 60
    assert t0 >= prev_end + PRE_ROLL_MARGIN_SEC - 1 / FR, (t0, prev_end)
    f0 = int(round(t0 * FR))
    assert np.array_equal(np.asarray(b0["codes_a"], dtype=np.int16), codes3[0][:, f0 : f0 + b0["n_frames"]].reshape(-1))
    tr = json.loads(b0["turns_json"])
    assert abs(tr[0]["start"] - (dstart - t0)) < 2e-3, (tr[0], dstart, t0)
    wa = json.loads(b0["words_a"])
    assert min(w["start"] for w in wa) >= dstart - t0 - 1e-3, "a word landed in the silence pre-roll"
pr = s_pre["pre_roll"]
assert s_pre["filters"]["pre_roll_sec"] == 1.0 and pr["n_extended"] == 2 and pr["words_in_pre_roll"] == 0, pr
print(
    f"[7] pre-roll: {moved}/2000 random starts moved, all into silence >= {PRE_ROLL_MARGIN_SEC} s after the "
    f"previous segment; real slicer: row starts {[r['span_start_sec'] for r in r_none]} -> "
    f"{[round(r['span_start_sec'], 2) for r in r_pre]}, codes/turns/words re-based, summary {pr}; 0 == no flag"
)


def preroll_no_silence_cap(start, segments, pre_roll_sec, seed_key):
    """MUTANT: moves the start back by pre_roll_sec regardless of the previous segment."""
    return max(0.0, start - pre_roll_sec) if pre_roll_sec > 0 else start


try:
    check_preroll(preroll_no_silence_cap)
except AssertionError as e:
    print(f"[6] mutant preroll_no_silence_cap caught: {str(e)[:80]}")
else:
    raise AssertionError("guard is vacuous: mutant preroll_no_silence_cap passed")

print("OK")
