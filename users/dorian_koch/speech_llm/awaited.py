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
``add`` checks the patterns against ``RESULTS_SINKED.json`` (every sink tag in the graph, written at each manager start):
a pattern no sink can produce is refused (``--unsinked`` to override: declared but the manager not restarted yet), and
sinked results of the same arm that no pattern covers are listed, so an item does not miss part of the arm's suite.
Dependency-free (the login node's python3, 3.9), like ``read_results.py``.
"""

from __future__ import annotations

import argparse
import fnmatch
import getpass
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import read_results  # noqa: E402

DEFAULT_FILE = "AWAITED.jsonl"
DEFAULT_SINKED = "RESULTS_SINKED.json"


def events(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for x in f:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except json.JSONDecodeError:  # a torn line (a session killed mid-write) must not break every call
                    print(f"awaited: skipping an unparsable line: {x[:80]!r}", file=sys.stderr)
    return out


def items(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for e in events(path):
        if e["event"] == "add":
            if e["id"] in out:  # two concurrent adds before the lock existed: keep both, never drop one
                print(f"awaited: duplicate id {e['id']} ({e['why'][:40]!r}); shown as {e['id']}.1", file=sys.stderr)
                out[f"{e['id']}.1"] = {**e, "id": f"{e['id']}.1", "closed": None}
                continue
            out[e["id"]] = {**e, "closed": None}
        elif e["event"] == "done" and e["id"] in out:
            out[e["id"]]["closed"] = e
    return out


def append(path: str, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def landed(item: dict, records: dict) -> list[str]:
    """Tags matching the item that were produced after it was marked: a re-run under the same tag (judge bump,
    hpc-rerun, code_version) must not read as READY on the old numbers. A record without a time counts."""
    since = item.get("since") or item.get("at", "")
    return [
        t
        for t in matching(item, records)
        if not read_results.when(records[t]) or read_results.when(records[t]) >= since
    ]


def matching(item: dict, records: dict) -> list[str]:
    """Every tag the item's patterns match, whenever produced."""
    pats = read_results.with_old_names(item["tags"])  # a renamed arm's results may carry its old name
    return sorted({t for t in records for p in pats if fnmatch.fnmatchcase(t, p)})


def _arm_key(pattern: str) -> str:
    """The longest path component of a pattern, cut at its first wildcard (the arm name, usually); "" if none is
    distinctive."""
    parts = [re.split(r"[*?\[]", p)[0] for p in pattern.split("/")]
    best = max(parts, key=len, default="")
    return best if len(best) >= 12 and best not in ("n1000", "eval", "summary") else ""


def sink_check(tags: list[str], sinked: list[str], records: dict) -> tuple[list[str], list[str]]:
    """``(patterns no sink or past record can match, sinked tags of the same arms that no pattern covers)``."""
    known = set(sinked) | set(records)
    pats = read_results.with_old_names(tags)
    dead = [
        t for t in tags if not any(fnmatch.fnmatchcase(k, p) for k in known for p in read_results.with_old_names([t]))
    ]
    keys = {k for k in map(_arm_key, tags) if k}
    covered = {k for k in known for p in pats if fnmatch.fnmatchcase(k, p)}
    missed = sorted(s for s in sinked if s not in covered and any(k in s for k in keys))
    return dead, missed


def near_tags(pattern: str, known, n: int = 3) -> list[str]:
    """Up to ``n`` known tags containing the pattern's longest literal piece: a pattern matches WHOLE tags, so a bare
    arm name (no ``benchmark/<grader>/n1000/`` prefix, no ``*``) matches nothing although the arm is sinked."""
    piece = max(re.split(r"[*?\[\]]", pattern), key=len)
    return sorted(k for k in known if piece and piece in k)[:n]


def refusal(dead: list[str], known: set, written: str) -> str:
    """The refusal for patterns that match no known tag: per pattern, either the likely typo (a piece of a tag given as
    if it were the whole tag) with the corrected ``--tag``, or the causes when nothing known even contains it."""
    out = ["refused: nothing was added, because these --tag patterns match no result tag, so the item could never "
           "turn READY. A --tag must match a WHOLE tag as read_results.py prints it (fnmatch: * = anything)."]  # fmt: skip
    for p in dead:
        near = near_tags(p, known, n=10**9)
        if near:
            fix = "*" + p.strip("*") + "*"
            n_fix = sum(fnmatch.fnmatchcase(k, fix) for k in known)
            out.append(f"\n  --tag {p!r}: {len(near)} known tag(s) contain this text, but none IS it, e.g.")
            out += [f"      {k}" for k in near[:3]]
            out.append(f"    did you mean  --tag '{fix}'  (matches {n_fix} tag(s); narrow it if that is too many)")
        else:
            out.append(f"\n  --tag {p!r}: no known tag contains it (sink manifest of {written}). Either")
            out.append(
                "    - the eval is declared but the manager has not restarted since: restart it, or add --unsinked;"
            )
            out.append(
                "    - its job kind writes no result: add the class to result_notify.SINK_KINDS or call notify_result;"
            )
            out.append("    - or the name is wrong: look it up with read_results.py <substring>.")
    return "\n".join(out)


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
    earlier = [t for t in matching(item, records) if t not in got]
    if earlier and not item["closed"]:
        print(f"    note: {len(earlier)} matching result(s) were produced before {item.get('since') or item['at'][:16]} "
              f"and do not count (re-add with --since to count them): {', '.join(earlier[:3])}")  # fmt: skip
    return ready


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--log", default=read_results.DEFAULT_LOG, help="the results log")
    ap.add_argument("--sinked", default=DEFAULT_SINKED, help="the sink manifest (written at each manager start)")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add", help="mark results as awaited")
    a.add_argument("--tag", action="append", required=True, help="result tag or fnmatch pattern (repeatable)")
    a.add_argument("--expect", type=int, help="tags that must land (default: one per --tag)")
    a.add_argument("--why", required=True, help="what the result decides or answers")
    a.add_argument("--next", default="", help="what to do once it has landed")
    a.add_argument("--since", help="count results produced from this time on (YYYY-MM-DD[ HH:MM]; default: now)")
    a.add_argument(
        "--unsinked", action="store_true", help="accept patterns no sink produces yet (manager not restarted)"
    )
    d = sub.add_parser("done", help="close an item (after its read-out)")
    d.add_argument("id", type=lambda x: int(x) if x.isdigit() else x, help="item id (N, or N.1 for a duplicate)")
    d.add_argument("--note", default="")
    lst = sub.add_parser("list", help="show items")
    lst.add_argument("--all", action="store_true", help="include closed items")
    sub.add_parser("check", help="open items (the default)")
    args = ap.parse_args(argv)

    its = items(args.file)
    if args.cmd == "add":
        import fcntl

        if os.path.exists(args.sinked):
            with open(args.sinked) as f:
                man = json.load(f)
            recs = read_results.load(args.log) if os.path.exists(args.log) else {}
            dead, missed = sink_check(args.tag, man["tags"], recs)
            if dead and not args.unsinked:
                print(refusal(dead, set(man["tags"]) | set(recs), man["written"]), file=sys.stderr)
                return 2
            if missed:
                print(f"note: {len(missed)} sinked result(s) of the same arm are not awaited (add a --tag if the "
                      f"decision needs them): {', '.join(missed[:8])}")  # fmt: skip
        else:
            print(f"awaited: no {args.sinked}, patterns not checked against the sinks", file=sys.stderr)

        # read + append under one lock: two sessions adding at once took the same id, and the later replaced the first
        with open(args.file, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            new_id = max((i for i in items(args.file) if isinstance(i, int)), default=0) + 1
            now = time.time()
            append(args.file, {"event": "add", "id": new_id, "tags": args.tag, "expect": args.expect or len(args.tag),
                               "why": args.why, "next": args.next, "by": getpass.getuser(), "at_unix": now,
                               **({"since": args.since} if args.since else {}),
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
    ready = sum(show(i, records) for i in sorted(sel, key=lambda i: float(i["id"])))
    print(f"{ready} of {len(sel)} ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
