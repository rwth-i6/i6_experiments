"""Read `RESULTS.jsonl` back out as step-ordered trajectories, grouped by arm.

Why this exists as a committed script rather than a throwaway: the never-miss-a-result sink appends
one JSON line per finished experiment, which is the right shape for *not losing* a result and the
wrong shape for *reading* one -- a checkpoint sweep lands as 12 unordered lines interleaved with
every other arm's. Twice now a readout that was sitting complete in this file went unread for weeks
(`a8_long`'s 12 re-probe points, 2026-08-05 -> read 2026-09-07). Grouping is the whole job.

Deliberately dependency-free and sisyphus-free, so it runs from a bare login-node shell:

    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/read_results.py            # every arm
    python3 .../read_results.py a8_long a11_full                                          # some arms
    python3 .../read_results.py --raw a17_diag                                            # full JSON

Accuracy is printed as a percentage next to the judge's mean quality, and n is printed once per arm
because it is what makes the numbers readable: the quick scorer's n=64 carries a standard error of
about 3.9 points, so a trajectory that moves by less than ~8 points between steps has not moved.
"""

# The login node's system `python3` predates PEP 604, and the whole point of this script is that it
# runs without the setup venv -- so keep the modern annotations lazy rather than dropping them.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

DEFAULT_LOG = "RESULTS.jsonl"
#: `quick_moshi_ft_<tag>_s<step>` / `moshi_ft_<tag>_s<step>` -> (tag, step). A tag may itself end in
#: digits, so anchor the step to the final `_s<digits>` rather than searching from the left.
_TAG_STEP = re.compile(r"^(?P<tag>.+?)_s(?P<step>\d+)$")


def _split_tag(tag: str) -> tuple[str, int | None]:
    m = _TAG_STEP.match(tag)
    if not m:
        return tag, None
    return m.group("tag"), int(m.group("step"))


def _overall(rec: dict) -> dict:
    """The scorer's `overall` block, wherever this record's writer happened to put it.

    `ResultNotify` has written three shapes over time (a bare summary, a nested
    ``summary.summary``, and a metrics digest), so read defensively rather than assuming the
    newest -- the old lines are exactly the ones worth going back to.
    """
    for path in (("summary", "summary", "overall"), ("summary", "overall"), ("overall",)):
        node = rec
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict) and "accuracy" in node:
            return node
    return {}


def load(path: str) -> dict[str, dict]:
    """Last line wins per tag -- a re-run appends rather than replacing."""
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a live writer is normal, not an error
            if isinstance(rec, dict) and rec.get("tag"):
                out[rec["tag"]] = rec
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="substrings; an arm is shown if any matches its tag")
    ap.add_argument("--log", default=DEFAULT_LOG, help=f"path to the results log (default {DEFAULT_LOG})")
    ap.add_argument("--raw", action="store_true", help="dump each matching record as JSON instead")
    args = ap.parse_args(argv)

    if not os.path.exists(args.log):
        print(f"no {args.log} -- run from the setup root", file=sys.stderr)
        return 2

    records = load(args.log)
    keep = lambda t: not args.filters or any(f in t for f in args.filters)  # noqa: E731

    if args.raw:
        for tag in sorted(records):
            if keep(tag):
                print(json.dumps(records[tag], indent=2))
        return 0

    arms: dict[str, list[tuple[int | None, dict]]] = defaultdict(list)
    for tag, rec in records.items():
        if keep(tag):
            arm, step = _split_tag(tag)
            arms[arm].append((step, rec))

    if not arms:
        print("no matching results", file=sys.stderr)
        return 3

    for arm in sorted(arms):
        points = sorted(arms[arm], key=lambda p: (p[0] is None, p[0] or 0))
        cells, ns, dates = [], set(), set()
        for step, rec in points:
            o = _overall(rec)
            dates.add((rec.get("at") or "")[:10])
            if o.get("n") is not None:
                ns.add(o["n"])
            label = str(step) if step is not None else "final"
            if o.get("accuracy") is None:
                cells.append(f"{label}:-")
            else:
                cells.append(f"{label}:{100 * o['accuracy']:.1f}/{o.get('avg_quality', float('nan')):.2f}")
        n = "/".join(str(x) for x in sorted(ns)) or "?"
        span = sorted(d for d in dates if d)
        when = span[0] if len(span) < 2 else f"{span[0]}..{span[-1]}"
        print(f"{arm}  [n={n}, {when}]")
        print("    " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
