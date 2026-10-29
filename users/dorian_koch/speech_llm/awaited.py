"""Results we are waiting on, and why: a watchlist over the result sink (``RESULTS.jsonl``).

Waiting on a result used to mean a background monitor or a note in a session's context, both lost when the session
ends. Mark it here instead; any later session (or the digest) asks which marked results have landed, with their numbers
and what they were for:

    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/awaited.py add \\
        --tag 'benchmark/llm_judge-parakeet_v3/n1000/*' --expect 5 \\
        --why "Parakeet v3 vs Whisper reply ASR: decides the knowledge default" --next "run analysis/asr_compare.py"
    python3 .../awaited.py              # every open item: READY (all its results landed, numbers shown) or waiting
    python3 .../awaited.py done 3 --note "read out 10-29: v3 = v2 within noise"
    python3 .../awaited.py list --all   # also the closed ones

A ``--tag`` is a result tag as ``read_results.py`` prints it (the output name; ``fnmatch`` patterns allowed);
``--expect N`` = how many tags the patterns must match in total (default: one per ``--tag``). An item is READY once that
many have landed. ``AWAITED.jsonl`` (setup root) is an append-only event log: ``add`` / ``done`` lines.
Dependency-free (the login node's python3, 3.9), like ``read_results.py``.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_results  # noqa: E402

DEFAULT_FILE = "AWAITED.jsonl"


def events(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def items(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for e in events(path):
        if e["event"] == "add":
            out[e["id"]] = {**e, "closed": None}
        elif e["event"] == "done" and e["id"] in out:
            out[e["id"]]["closed"] = e
    return out


def append(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def landed(item: dict, records: dict) -> list[str]:
    return sorted({t for t in records for p in item["tags"] if fnmatch.fnmatchcase(t, p)})


def show(item: dict, records: dict) -> bool:
    got = landed(item, records)
    ready = len(got) >= item["expect"]
    age = (time.time() - item["at_unix"]) / 86400
    state = "READY" if ready else f"waiting {len(got)}/{item['expect']}"
    if item["closed"]:
        state = f"closed {item['closed']['at'][:10]}: {item['closed'].get('note', '')}"
    print(f"[{item['id']}] {state}  (marked {item['at'][:16]}, {age:.1f} d ago)")
    print(f"    why:  {item['why']}")
    if item.get("next"):
        print(f"    next: {item['next']}")
    print(f"    tags: {', '.join(item['tags'])}")
    for t in got:
        print(f"      {t}: {read_results.render(records[t])[0]}")
    return ready


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--log", default=read_results.DEFAULT_LOG, help="the results log")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add", help="mark results as awaited")
    a.add_argument("--tag", action="append", required=True, help="result tag or fnmatch pattern (repeatable)")
    a.add_argument("--expect", type=int, help="tags that must land (default: one per --tag)")
    a.add_argument("--why", required=True, help="what the result decides or answers")
    a.add_argument("--next", default="", help="what to do once it has landed")
    d = sub.add_parser("done", help="close an item (after its read-out)")
    d.add_argument("id", type=int)
    d.add_argument("--note", default="")
    lst = sub.add_parser("list", help="show items")
    lst.add_argument("--all", action="store_true", help="include closed items")
    sub.add_parser("check", help="open items (the default)")
    args = ap.parse_args(argv)

    its = items(args.file)
    if args.cmd == "add":
        new_id = max(its, default=0) + 1
        now = time.time()
        append(args.file, {"event": "add", "id": new_id, "tags": args.tag, "expect": args.expect or len(args.tag),
                           "why": args.why, "next": args.next, "by": getpass.getuser(), "at_unix": now,
                           "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))})  # fmt: skip
        print(f"awaited [{new_id}] added")
        return 0
    if args.cmd == "done":
        if args.id not in its or its[args.id]["closed"]:
            print(f"no open item {args.id}", file=sys.stderr)
            return 2
        append(args.file, {"event": "done", "id": args.id, "note": args.note,
                           "at": time.strftime("%Y-%m-%d %H:%M:%S")})  # fmt: skip
        print(f"awaited [{args.id}] closed")
        return 0

    records = read_results.load(args.log) if os.path.exists(args.log) else {}
    sel = [i for i in its.values() if (args.cmd == "list" and args.all) or not i["closed"]]
    if not sel:
        print("nothing awaited")
        return 0
    ready = sum(show(i, records) for i in sorted(sel, key=lambda i: i["id"]))
    print(f"{ready} of {len(sel)} ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
