#!/usr/bin/env python3
"""The n=1000 judged ledger, at a glance -- every benchmark tag, sorted by score.

Why this exists: `ft_v3` (QA-only, no Fisher) has scored 12.9% since long before the A-series
started, against every 50%-Fisher A-arm at 6.5-8.7%. That comparison was sitting in
`output/benchmark/*/summary` the whole time and nobody had read the ledger end to end -- it took an
explicit "why are we even training on Fisher" to surface it. `read_results.py` reads RESULTS.jsonl
(per-arm digests); this reads the BENCHMARK summaries, which is where the comparable numbers live.

Run from the setup root:  python3 read_benchmarks.py [substring]
Login-node python3, no venv, no sisyphus import -- deliberately, like read_results.py.
"""
import json, os, sys, glob

root = "output/benchmark"
want = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for path in sorted(glob.glob(f"{root}/*/summary")):
    tag = os.path.basename(os.path.dirname(path))
    if want and want not in tag:
        continue
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        rows.append((None, tag, f"unreadable: {e}", 0))
        continue
    o = d.get("overall") or {}
    rows.append((o.get("accuracy"), tag, o.get("avg_quality"), o.get("n")))

scored = sorted([r for r in rows if r[0] is not None], key=lambda r: -r[0])
other = [r for r in rows if r[0] is None]
print(f"{'tag':<42}{'acc':>8}{'quality':>9}{'n':>7}")
print("-" * 66)
for acc, tag, q, n in scored:
    qs = f"{q:.3f}" if isinstance(q, (int, float)) else "-"
    print(f"{tag:<42}{100*acc:>7.1f}%{qs:>9}{n:>7}")
for _, tag, msg, _ in other:
    print(f"{tag:<42}  {msg}")
print("-" * 66)
print(f"{len(scored)} scored tags. Graders differ: a judge-free alias score and an LLM-judged score")
print("are NOT comparable in absolute terms (measured gap 6.2 pts on base, 0.0 on terse arms).")
print("Compare within a grader, and always name the base reference alongside any arm.")
