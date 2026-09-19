"""Guard: slicing whole-episode codes into dialogues is frame-correct and stays DuplexChat's own.

Two failure modes, both of which produce output that looks entirely valid:

  * **Wrong axis.** The stored code layout is CODEBOOK-MAJOR (`k * n_frames + f`), so taking frames
    means slicing the second axis after a reshape. A contiguous slice of the flat array -- the
    obvious thing to write -- takes a band of *codebooks* instead. Same dtype, plausible length,
    trains happily, produces noise. Asserted here against an independent reference, AND the naive
    version is asserted to differ, so the test cannot pass by coincidence.
  * **A re-implementation of their filter.** `extract_valid_dialogues` has four interacting rules
    that are easy to get subtly wrong (dominance applied AFTER long-dialogue chunking;
    single-speaker runs dropped by never being emitted rather than by a filter; a gap of exactly
    `gap_seconds` splits). The whole premise of this pipeline is "exactly what DuplexChat does", so
    an AST check asserts we import and call THEIR function and define no substitute.

Login node, no GPU, no `datasets` needed: `podcast_slice_main` has only light module-level imports.

Run: CUDA_HOME=/usr .venv/bin/python recipe/.../tests/check_episode_slice.py
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
LIB = SETUP / "recipe/speech_llm/full_duplex/moshi_family"
os.environ.setdefault("CUDA_HOME", "/usr")

fails = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name} {detail if not cond else ''}")
    if not cond:
        fails.append(name)


spec = importlib.util.spec_from_file_location("psm", LIB / "podcast_slice_main.py")
psm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psm)

print("[1] slice_codes takes FRAMES, not a band of codebooks")
K, F = 8, 100
# Distinct value per (codebook, frame) so any axis confusion is visible.
ref = (np.arange(K)[:, None] * 1000 + np.arange(F)[None, :]).astype(np.int16)
flat = ref.reshape(-1)
f0, f1 = 30, 55
got = psm.slice_codes(flat, K, F, f0, f1)
want = np.ascontiguousarray(ref[:, f0:f1]).reshape(-1)
check("matches an independent reshape-and-slice", np.array_equal(got, want))
check("length is K * (f1-f0)", got.size == K * (f1 - f0), f"{got.size} vs {K * (f1 - f0)}")
check("dtype stays int16", got.dtype == np.int16, str(got.dtype))
# Non-vacuity: the naive flat slice must NOT agree, or this test proves nothing.
naive = flat[f0 * K : f1 * K]
check(
    "the naive flat slice really is different (non-vacuity)", naive.size == got.size and not np.array_equal(naive, got)
)
check(
    "every codebook is represented after slicing",
    len({int(v) // 1000 for v in got}) == K,
    f"{len({int(v) // 1000 for v in got})} of {K}",
)
check("frame 0 of the slice is frame f0 of the source", int(got[0]) % 1000 == f0, str(int(got[0])))

print("[2] frame_span is a SUPERSET of the requested seconds")
fr = 12.5
n = 1000
for start, end in ((0.0, 10.0), (3.37, 9.91), (0.04, 0.09), (3.5, 9.79)):
    a, b = psm.frame_span(start, end, fr, n)
    check(
        f"[{start}, {end}) covers its seconds",
        a / fr <= start and b / fr >= end,
        f"frames {a}..{b} = {a / fr:.3f}..{b / fr:.3f} s",
    )
check("clamped at 0", psm.frame_span(-5.0, 2.0, fr, n)[0] == 0)
check("clamped at n_frames", psm.frame_span(0.0, 1e9, fr, n)[1] == n)
# The superset property holds UNLESS the row ends first -- frames that do not exist cannot be
# returned, so a request past the end is capped. Asserted explicitly rather than left as a hole.
a, b = psm.frame_span(79.999, 80.001, fr, n)
check(
    "a request past the end of the row is capped at n_frames, not extended",
    b == n and b / fr < 80.001,
    f"frames {a}..{b}",
)

# floor/ceil vs round is a CHOICE, so it must be one that visibly differs -- otherwise this
# assertion passes on a span where the two happen to coincide and proves nothing. At 12.5 Hz,
# 3.5 s -> 43.75 (floor 43, round 44) and 9.79 s -> 122.375 (ceil 123, round 122): both differ,
# and in both cases rounding would CLIP audio that floor/ceil keeps.
a, b = psm.frame_span(3.5, 9.79, fr, n)
check(
    "floor/ceil differs from round at BOTH ends of an off-grid span",
    a != round(3.5 * fr) and b != round(9.79 * fr),
    f"{(a, b)} vs {(round(3.5 * fr), round(9.79 * fr))}",
)
check("...and rounding would have clipped the dialogue", round(3.5 * fr) / fr > 3.5 and round(9.79 * fr) / fr < 9.79)

print("[3] the slice must be DuplexChat's own filter, not ours")
src = (LIB / "podcast_slice_main.py").read_text()
tree = ast.parse(src)
imported = {
    a.name
    for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("duplexchat_pipe")
    for a in node.names
}
check("imports extract_valid_dialogues from duplexchat_pipe", "extract_valid_dialogues" in imported, str(imported))
called = {getattr(n.func, "id", getattr(n.func, "attr", "")) for n in ast.walk(tree) if isinstance(n, ast.Call)}
check("and calls it", "extract_valid_dialogues" in called)
ours = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
forbidden = {
    "extract_valid_dialogues",
    "is_balanced_dialogue",
    "split_into_dialogues",
    "_two_speaker_runs",
    "_split_long_dialogue",
}
check("defines no substitute for their rules", not (ours & forbidden), str(ours & forbidden))
# All four thresholds must be forwarded, or a "policy change" would silently do nothing.
for kw in ("gap_seconds", "max_single_speaker_ratio", "min_duration_seconds", "max_duration_seconds"):
    check(f"forwards {kw}", f"{kw}=args.{kw}" in src.replace(" ", ""), "not passed through")

print("[4] the whole-episode assumption is asserted, not assumed")
check("refuses a row whose frame 0 is not episode t=0", "span_start_sec" in src and "!= 0.0" in src)
check("records rejections with a reason", "rejected.jsonl" in src and "def reject" in src)
check("refuses to register an empty corpus", "refusing to register an empty corpus" in src)

print("[5] the episode gate matches their pipeline")
check(
    "min_dialogues_per_episode defaults to 4 (their pipeline.py:301)",
    "default=4" in src.replace(" ", "").replace("default=4,", "default=4"),
    "their episode gate is `< 4`",
)

print("[6] the job class forwards every filter to the worker")
jobsrc = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/podcast_ingest.py").read_text()
jtree = ast.parse(jobsrc)
cls = next(n for n in ast.walk(jtree) if isinstance(n, ast.ClassDef) and n.name == "PodcastDialogueSlice")
init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
params = {a.arg for a in init.args.kwonlyargs}
for kw in (
    "gap_seconds",
    "max_single_speaker_ratio",
    "min_duration_seconds",
    "max_duration_seconds",
    "min_dialogues_per_episode",
):
    check(f"{kw} is a constructor arg", kw in params)
excl = next(
    (n for n in cls.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__sis_hash_exclude__"),
    None,
)
excluded = {k.value for k in excl.value.keys} if excl is not None else set()
# The filters MUST stay hashed: they are the retention policy, and the entire point of this job is
# that changing them produces a different corpus rather than silently reusing the old one.
check(
    "no filter threshold is hash-excluded",
    not (
        excluded
        & {
            "gap_seconds",
            "max_single_speaker_ratio",
            "min_duration_seconds",
            "max_duration_seconds",
            "min_dialogues_per_episode",
        }
    ),
    f"excluded {excluded}",
)
check(
    "it asks for NO GPU",
    "'gpu'" not in jobsrc.split("class PodcastDialogueSlice")[1].split("def tasks")[0]
    and '"gpu"' not in jobsrc.split("class PodcastDialogueSlice")[1].split("def tasks")[0],
)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
