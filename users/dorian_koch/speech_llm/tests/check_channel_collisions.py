"""Guard: the dialogue-row channel-collision filter (``PodcastDialogueSlice(drop_channel_collisions=True)``).

WHY. Separation + permutation repair can put BOTH diarized speakers of a dialogue on one channel (or a
third voice merged under one label lands there), so the other channel is not the other side of the
conversation. The 2026-09-22 review measured it with a word-placement detector (validated by ear on
9/9 blind pilot clips; 6.6% of judged dialogue time on jre_chunked): for each of a row's two labels,
count ASR words whose midpoint is inside that label's turns and outside the other's, per channel; both
labels' majority on the SAME channel = collision. Checked:
  1. PARITY. ``podcast_slice_main.channel_collision`` gives the reviewer's verdict (its code copied
     verbatim below from cluster /tmp/tsp_check.py) on 2,000 random rows, incl. ties and < 3 words.
  2. VERDICTS. Separated / collided / too few words / not two labels on hand-built rows.
  3. THE REAL SLICER on a one-episode part holding one separated and one collided dialogue: with the
     flag it drops exactly the collided one, records it in rejected.jsonl and counts rows + hours in
     summary.json; without the flag (the default) the collided row survives (non-vacuity), and the
     summary still reports the collision it did not drop.
  4. WIRING. The Job forwards the flag, and hash-excludes it only at its False default.
Shown able to fail (2026-09-23, mutated copies): ``channel_collision`` always "separated" -> [1] fails;
the slicer never dropping -> [3] fails.

moshi_family venv (datasets), CPU, ~20 s:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_channel_collisions.py'
"""

import ast
import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
from datasets import Dataset, Sequence, load_from_disk  # noqa: E402

from moshi_family.podcast_mimi_encode import part_features  # noqa: E402
from moshi_family.podcast_slice_main import channel_collision  # noqa: E402


# ---- the reviewer's detector, verbatim (tsp_check.py, 2026-09-22) ---------------------------------
def channel_of(label, other, turns, words):
    """words on each channel ('a','b') inside label's turns and outside other's turns."""
    mine = [(t["start"], t["end"]) for t in turns if t["speaker"] == label]
    theirs = [(t["start"], t["end"]) for t in turns if t["speaker"] == other]
    inside = lambda m, spans: any(s <= m <= e for s, e in spans)  # noqa: E731
    c = Counter()
    for ch, ws in words.items():
        for w in ws:
            m = 0.5 * (w["start"] + w["end"])
            if inside(m, mine) and not inside(m, theirs):
                c[ch] += 1
    return c


def reference_kind(turns, words):
    labs = sorted({t["speaker"] for t in turns})
    if len(labs) != 2:
        return "not_two_labels"
    c0 = channel_of(labs[0], labs[1], turns, words)
    c1 = channel_of(labs[1], labs[0], turns, words)
    n0, n1 = sum(c0.values()), sum(c1.values())
    if n0 < 3 or n1 < 3:
        return "too_few_words"
    ch0, ch1 = c0.most_common(1)[0][0], c1.most_common(1)[0][0]
    return "collision" if ch0 == ch1 else "separated"


# ---- 1. parity ------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
seen = Counter()
for _ in range(2000):
    labs = ["S0", "S1", "S2"][: rng.choice([1, 2, 2, 2, 3])]
    turns, t = [], 0.0
    for _k in range(rng.integers(2, 8)):
        d = float(rng.uniform(0.5, 6.0))
        turns.append({"speaker": str(rng.choice(labs)), "start": round(t, 3), "end": round(t + d, 3)})
        t += d - float(rng.uniform(0, 0.5 * d))  # overlapping turns, like pyannote's
    words = {}
    for ch in "ab":
        n = int(rng.integers(0, 12))
        st = np.sort(rng.uniform(0, t, n))
        words[ch] = [{"text": "w", "start": round(float(s), 3), "end": round(float(s) + 0.3, 3)} for s in st]
    got, _counts = channel_collision(turns, words)
    want = reference_kind(turns, words)
    assert got == want, (got, want, turns, words)
    seen[got] += 1
assert all(seen[k] > 20 for k in ("collision", "separated", "too_few_words", "not_two_labels")), seen
print(f"[1] parity with the validated detector on 2000 random rows: {dict(seen)}")


# ---- 2. verdicts on hand-built rows ---------------------------------------------------------------
def turns_of(spec):
    return [{"speaker": s, "start": a, "end": b} for s, a, b in spec]


def words_in(spans, per_sec=1.0):
    """one 0.4 s word per second inside each (start, end)."""
    out = []
    for a, b in spans:
        t = a + 0.1
        while t + 0.4 <= b:
            out.append({"text": "w", "start": round(t, 3), "end": round(t + 0.4, 3)})
            t += 1.0 / per_sec
    return out


TURNS = [("A", 0.0, 5.0), ("B", 5.0, 10.0), ("A", 10.0, 15.0), ("B", 15.0, 20.0)]
A_SP, B_SP = [(a, b) for s, a, b in TURNS if s == "A"], [(a, b) for s, a, b in TURNS if s == "B"]
k, c = channel_collision(turns_of(TURNS), {"a": words_in(A_SP), "b": words_in(B_SP)})
assert k == "separated", (k, c)
k, c = channel_collision(turns_of(TURNS), {"a": words_in(A_SP) + words_in(B_SP), "b": []})
assert k == "collision" and c == {"A": {"a": 10}, "B": {"a": 10}}, (k, c)
k, _ = channel_collision(turns_of(TURNS), {"a": words_in(A_SP) + words_in([(5.0, 7.0)]), "b": []})
assert k == "too_few_words", k  # B has 2 words: cannot judge -> kept
k, _ = channel_collision(turns_of(TURNS + [("C", 20.0, 25.0)]), {"a": words_in(A_SP), "b": words_in(B_SP)})
assert k == "not_two_labels", k
print("[2] separated / collision / too_few_words (< 3 words) / not_two_labels on hand-built rows")

# ---- 3. the real slicer ---------------------------------------------------------------------------
FR, K = 12.5, 8
DUR = 60.0
F = int(DUR * FR)
# Dialogue 1 [0, 20) separated (A on channel 0, B on channel 1); a 10 s gap (> DuplexChat's 5 s) splits
# it from dialogue 2 [30, 50), where both labels' words sit on channel 0.
second = [(s, a + 30.0, b + 30.0) for s, a, b in TURNS]
diar = turns_of(TURNS + second)
asr = {
    "backend": "test",
    "model": "test",
    "channels": [
        {"channel": 0, "speaker": "A", "words": words_in(A_SP) + words_in([(a, b) for _s, a, b in second])},
        {"channel": 1, "speaker": "B", "words": words_in(B_SP)},
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
    diar_turns_json=json.dumps(diar),
)
tmp = tempfile.mkdtemp(prefix="check_collisions_")
ep_dir = os.path.join(tmp, "episodes")
Dataset.from_dict({n: [v] for n, v in ep.items()}, features=feats).save_to_disk(
    os.path.join(ep_dir, "parts", "part_00000")
)
(duplexchat,) = sorted(glob.glob("work/i6_core/tools/git/CloneGitRepositoryJob.*/output/DuplexChat/src"))[:1]


def run_slicer(tag, *extra):
    out = os.path.join(tmp, tag)
    subprocess.run(
        [
            sys.executable,
            os.path.abspath("recipe/speech_llm/full_duplex/moshi_family/podcast_slice_main.py"),
            "--in_dir",
            ep_dir,
            "--out_dir",
            out,
            "--repo_src",
            os.path.abspath(duplexchat),
            "--min_dialogues_per_episode",
            "0",
            *extra,
        ],
        check=True,
        cwd=tmp,
        env={**os.environ, "PYTHONPATH": os.path.abspath("recipe/speech_llm/full_duplex")},
        stdout=subprocess.DEVNULL,
    )
    rows = [r for p in sorted(glob.glob(os.path.join(out, "parts", "part_*"))) for r in load_from_disk(p)]
    rej = os.path.join(out, "rejected.jsonl")
    rejected = [json.loads(line) for line in open(rej)] if os.path.exists(rej) else []
    return rows, rejected, json.load(open(os.path.join(out, "summary.json")))


rows_off, rej_off, sum_off = run_slicer("off")
starts_off = sorted(round(float(r["span_start_sec"])) for r in rows_off)
assert starts_off == [0, 30], f"fixture must give the two dialogues, got {starts_off}"
assert not rej_off and sum_off["channel_collisions"]["n"]["collision"] == 1, (rej_off, sum_off)
assert sum_off["channel_collisions"]["n_dropped"] == 0, sum_off
rows_on, rej_on, sum_on = run_slicer("on", "--drop_channel_collisions")
assert [round(float(r["span_start_sec"])) for r in rows_on] == [0], "the filter must keep exactly the separated row"
assert [r["reason"] for r in rej_on] == ["channel_collision"] and "dialogue 1" in rej_on[0]["detail"], rej_on
cc = sum_on["channel_collisions"]
dropped_sec = float(next(r for r in rows_off if round(float(r["span_start_sec"])) == 30)["duration_sec"])
assert cc["n_dropped"] == 1 and abs(cc["dropped_hours"] - round(dropped_sec / 3600, 3)) < 1e-9, cc
assert sum_on["filters"]["drop_channel_collisions"] is True and sum_on["n_dialogues"] == 1, sum_on
print(
    f"[3] real slicer: off -> {len(rows_off)} rows (collided one kept, reported {sum_off['channel_collisions']['n']}); "
    f"on -> {len(rows_on)} row, dropped {cc['n_dropped']} ({dropped_sec:.1f} s, share {cc['dropped_share_of_dialogue_time']:.0%}), "
    "recorded in rejected.jsonl"
)

# ---- 4. wiring ------------------------------------------------------------------------------------
jobsrc = open("recipe/i6_experiments/users/dorian_koch/speech_llm/podcast_ingest.py").read()
cls = next(n for n in ast.walk(ast.parse(jobsrc)) if isinstance(n, ast.ClassDef) and n.name == "PodcastDialogueSlice")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
kw = {a.arg: d for a, d in zip(init.args.kwonlyargs, init.args.kw_defaults)}
assert "drop_channel_collisions" in kw and ast.literal_eval(kw["drop_channel_collisions"]) is False
excl = next(
    n for n in cls.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__sis_hash_exclude__"
)
ex = ast.literal_eval(excl.value)
assert ex.get("drop_channel_collisions", "missing") is False, f"must be excluded only at False: {ex}"
run_src = ast.get_source_segment(
    jobsrc, next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
)
assert '"--drop_channel_collisions"' in run_src and "drop_channel_collisions" in run_src, (
    "run() does not forward the flag"
)
print(
    "[4] PodcastDialogueSlice: drop_channel_collisions=False default, hash-excluded only at False, forwarded to the worker"
)
print("OK")
