"""Every metric any RL run has logged has a structured name (``rl_metric_names``), and the readers' translation of it
is unambiguous.

    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_metric_registry.py

Why: the readers (``evalstats.rl_metrics``, ``run_curves``) show legacy runs under the structured names, so a key the
registry misses or matches twice would silently drop or merge a curve. Reads the metrics streams of every
``RLFinetune`` / ``RLFinetuneV2`` job dir (first 400 rows of each stream), builds each run's layout from its own config
and translates every row with the strict ``Namer``: no row may raise (unknown key, a key two registry entries match,
a layout that cannot be built), every output name must be ``registered``, and no two keys of a row may land on one name.
The writer side (structured runs through the real trainer) is ``moshi_family/rl/tests/test_metric_names.py``.
Can fail: two mutants (a registry entry removed; an entry duplicated under another pattern) run in ``main`` and must
be caught. Standard library only (the login node's python3).
"""

import glob
import json
import os
import sys

SETUP = os.environ.get("SETUP_ROOT", os.getcwd())  # run from the setup root (run_guards does)
assert os.path.isdir(os.path.join(SETUP, "work")), f"run from the setup root (no work/ in {SETUP})"
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
import rl_metric_names as M  # noqa: E402

W = os.path.join(SETUP, "work")
RUN_DIRS = (
    "speech_llm/full_duplex/rl_spec/job/RLFinetuneV2.*/output/run_dir",
    "i6_experiments/users/dorian_koch/speech_llm/rl_finetune/RLFinetune.*/output/run_dir",
)
MAX_ROWS = 400


def check(max_rows=MAX_ROWS) -> dict:
    """-> {"runs", "rows", "names"}; raises AssertionError naming the first problems."""
    dirs = sorted(d for pat in RUN_DIRS for d in glob.glob(os.path.join(W, pat)))
    assert dirs, f"no RL run dirs under {W}"
    problems, n_rows, names = [], 0, set()
    for d in dirs:
        try:
            namer = M.Namer(M.Layout.from_run_dir(d), strict=True)
        except Exception as e:  # noqa: BLE001 -- reported, not raised
            problems.append(f"{d}: layout: {type(e).__name__}: {e}")
            continue
        for p in sorted(glob.glob(os.path.join(d, "metrics.*.jsonl"))):
            stream = os.path.basename(p)[len("metrics.") : -len(".jsonl")]
            with open(p, encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= max_rows:
                        break
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if "kind" in row:
                        continue
                    n_rows += 1
                    try:
                        out = namer.translate(row, stream)
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{os.path.relpath(p, W)}: {type(e).__name__}: {e}")
                        continue
                    dropped = len(row) - len(out)
                    if dropped < 0:
                        problems.append(f"{os.path.relpath(p, W)}: translation added keys")
                    bad = [k for k in out if not M.registered(k)]
                    if bad:
                        problems.append(f"{os.path.relpath(p, W)}: not registered {bad[:5]}")
                    names.update(out)
    uniq = list(dict.fromkeys(problems))
    assert not uniq, f"{len(uniq)} problems:\n  " + "\n  ".join(uniq[:20])
    return {"runs": len(dirs), "rows": n_rows, "names": len(names)}


def _collisions_checked():
    """Two legacy keys of one row on the same name would merge silently: ``Namer`` output keeps every input key
    except the documented drops, so compare the counts on a row that has both colliding candidates."""
    nm = M.Namer(M.Layout.from_v2_config({"tasks": [
        {"name": "knowledge", "rollout": {"kind": "monologue_rollout"}, "source": {"kind": "knowledge_questions"}},
        {"name": "selfplay", "rollout": {"kind": "selfplay_rollout"}, "source": {"kind": "selfplay_scenarios"}}]}),
        strict=True)  # fmt: skip
    row = {"step": 1, "correct_rate": 0.1, "selfplay_words_mean": 3.0, "words_mean": 4.0, "reward_mean": 0.5}
    out = nm.translate(row)
    assert len(out) == len(row), (row, out)


def _mutant(name, patch):
    saved = M.TASK
    try:
        M.TASK = patch(saved)
        M._REGISTERED_RE = None
        try:
            check(max_rows=50)
        except (AssertionError, ValueError):
            print(f"  mutant {name}: caught")
            return
        raise SystemExit(f"mutant {name} NOT caught")
    finally:
        M.TASK = saved
        M._REGISTERED_RE = None


def main():
    _collisions_checked()
    print("check_metric_registry:", check())
    _mutant("entry removed", lambda t: tuple(m for m in t if m.legacy != "correct_rate"))
    _mutant("entry duplicated", lambda t: t + (M.Metric(r"correct_rat(?P<x>e)", "dup.{x}", "frac", "mutant"),))
    print("check_metric_registry OK")


if __name__ == "__main__":
    main()
