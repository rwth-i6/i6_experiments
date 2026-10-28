"""Read `RESULTS.jsonl` back out: one line per result, every kind the sink writes, grouped by arm.

The never-miss-a-result sink appends one JSON line per finished experiment, which is the right shape for *not losing*
a result and the wrong shape for *reading* one -- a checkpoint sweep lands as 12 unordered lines interleaved with every
other arm's (`a8_long`'s 12 re-probe points sat unread 2026-08-05 -> 09-07), and a sink attached after the fact lands
years of old results at once. So: group by arm, order by step, and sort ``--recent`` by when the result was PRODUCED.

Dependency-free and sisyphus-free (the login node's system python3, 3.9):

    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/read_results.py --recent 40      # newest results
    python3 .../read_results.py a8_long ppx_rl_sp_mono                                         # some arms
    python3 .../read_results.py --raw a17_diag                                                 # full JSON

Kinds (shape of the line's ``summary``): knowledge (acc% / quality, n), MFDB (turn-take TOR / median latency,
backchannel JSD / TOR, pause TORs, interruption rating), MRQA (acc_upstream / acc_strict per subset), spj2 (overall /
adherence / coherence per label), coherence (mean [ci] per label), held-out loss (per run / step), training and RL runs
(last step + last value of each digested metric). A shape no reader knows prints its keys with ``?`` instead of
pretending to be empty. The quick scorer's n=64 carries ~3.9 points of standard error: a trajectory that moves by less
than ~8 points between steps has not moved.
"""

# The login node's system `python3` predates PEP 604, and the whole point of this script is that it runs without the
# setup venv -- so keep the modern annotations lazy.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

DEFAULT_LOG = "RESULTS.jsonl"
#: `quick_moshi_ft_<tag>_s<step>` / `.../<tag>_s<step>` -> (tag, step); anchored to the final `_s<digits>`
_TAG_STEP = re.compile(r"^(?P<tag>.+?)_s(?P<step>\d+)$")
MFDB_TASKS = (
    "candor_turn_taking",
    "icc_backchannel",
    "candor_pause_handling",
    "synthetic_pause_handling",
    "synthetic_user_interruption",
)


def _split_tag(tag: str) -> tuple[str, int | None]:
    m = _TAG_STEP.match(tag)
    if not m:
        return tag, None
    return m.group("tag"), int(m.group("step"))


def _num(x, fmt="{:.3g}") -> str:
    if isinstance(x, dict):
        x = x.get("mean")
    return "-" if not isinstance(x, (int, float)) or isinstance(x, bool) else fmt.format(x)


def _parts(rec: dict) -> list:
    """The record's ``(label, payload)`` pairs (``summary`` is ``{label: payload}``; the oldest lines are bare)."""
    s = rec.get("summary")
    if not isinstance(s, dict):
        return [("", rec)]
    if "overall" in s:  # the oldest lines: a bare grading summary
        return [("", s)]
    return list(s.items())


def _overall(p) -> dict:
    for node in (p, p.get("summary") if isinstance(p, dict) else None):
        if isinstance(node, dict) and isinstance(node.get("overall"), dict) and "accuracy" in node["overall"]:
            return node["overall"]
    return {}


def render(rec: dict) -> tuple[str, bool]:
    """``(one-line text, recognised)`` for a record of any kind."""
    parts = _parts(rec)
    tasks = {}
    out = []
    for lab0, p in parts:
        if not isinstance(p, dict):
            continue
        o = _overall(p)
        if o:
            out.append(f"acc {100 * o['accuracy']:.1f}% q {_num(o.get('avg_quality'), '{:.2f}')} n={o.get('n', '?')}")
        elif {"average_take_turn", "JSD", "average_rating"} & set(p):
            tasks[lab0] = p  # an MFDB task's eval.json, labelled by its task
        elif "mean" in p and "judge" in p:  # MRQA
            cells = []
            for sub in ("llama_questions", "web_questions", "trivia_qa", "halueval"):
                m = (p.get(sub) or {}).get("monologue") or (p.get(sub) or {}).get("asr") or {}
                if m:
                    cells.append(
                        f"{sub.split('_')[0]} {_num(m.get('acc_upstream'), '{:.3f}')}/{_num(m.get('acc_strict'), '{:.3f}')}"
                    )
            out.append("MRQA up/strict " + " ".join(cells))
        elif isinstance(p.get("result"), dict) and "evaluator" in p:  # one VoiceBench subset
            r = p["result"]
            v = r.get("acc", r.get("final"))
            out.append(f"{lab0} {_num(v, '{:.1f}' if 'acc' in r else '{:.3f}')} n={p.get('n', '?')}")
        elif isinstance(p.get("summary"), dict) and "success_rate" in p["summary"]:  # one MTR cell
            m = p["summary"]
            out.append(
                f"{m.get('scenario', m.get('dimension'))} success {_num(m['success_rate'], '{:.3f}')} lat {_num(m.get('avg_latency'), '{:.2f}')}"
            )
        elif isinstance(p.get("labels"), dict):
            for lab, v in p["labels"].items():
                if not isinstance(v, dict):
                    continue
                if isinstance(v.get("metrics"), dict):  # spj2
                    m = v["metrics"]
                    keys = [k for k in ("overall", "adherence", "coherence", "responsiveness") if k in m]
                    out.append(
                        f"{lab}: "
                        + " ".join(f"{k} {_num(m[k], '{:.2f}')}" for k in keys)
                        + f" n={v.get('n_conv', '?')}"
                    )
                elif "mean" in v:  # coherence
                    ci = v.get("ci95") or ["?", "?"]
                    out.append(
                        f"{lab}: {_num(v['mean'], '{:.2f}')} [{_num(ci[0], '{:.2f}')}, {_num(ci[1], '{:.2f}')}] n={v.get('n', '?')}"
                    )
        elif isinstance(p.get("rows"), list) and p["rows"] and isinstance(p["rows"][0], dict) and "run" in p["rows"][0]:
            for r in p["rows"]:
                loss = (r.get("metrics") or {}).get("loss")
                out.append(f"{r.get('run')}@{r.get('step')} {r.get('set', '')} loss {_num(loss, '{:.4f}')}")
        elif (
            "rows" in p
            and ("last_step" in p or p.get("rows") == 0)
            or any(isinstance(v, dict) and "rows" in v for v in p.values())
        ):
            streams = {"": p} if "rows" in p else {k: v for k, v in p.items() if isinstance(v, dict) and "rows" in v}
            for name, st in streams.items():
                last = " ".join(
                    f"{k} {_num(v.get('last'))}" for k, v in st.items() if isinstance(v, dict) and "last" in v
                )
                label = name.replace("metrics.", "").replace(".jsonl", "") or "metrics"
                out.append(f"{label}: step {st.get('last_step', '?')} {last}".rstrip())
    if tasks:
        by = tasks
        cells = []
        tt, bc = by.get("candor_turn_taking"), by.get("icc_backchannel")
        if tt:
            cells.append(
                f"turn TOR {_num(tt.get('average_take_turn'), '{:.2f}')} lat {_num(tt.get('median_latency'), '{:.2f}')}s"
            )
        if bc:
            cells.append(f"bc JSD {_num(bc.get('JSD'), '{:.3f}')} TOR {_num(bc.get('TOR'), '{:.2f}')}")
        for t, short in (("candor_pause_handling", "c-pause"), ("synthetic_pause_handling", "s-pause")):
            if by.get(t):
                cells.append(f"{short} TOR {_num(by[t].get('average_take_turn'), '{:.2f}')}")
        if by.get("synthetic_user_interruption"):
            cells.append(f"interrupt {_num(by['synthetic_user_interruption'].get('average_rating'), '{:.2f}')}")
        if not cells:  # one task's eval.json on its own (a per-task sink)
            p = next(iter(tasks.values()))
            cells.append(" ".join(f"{k} {_num(v, '{:.3f}')}" for k, v in p.items() if not isinstance(v, (list, str))))
        out.append("MFDB " + " | ".join(cells))
    if not out and parts and all(isinstance(p, str) for _, p in parts):
        out.append("artifacts: " + ", ".join(str(k) for k in (rec.get("summary") or {})))
    if out:
        return "; ".join(out), True
    keys = sorted({k for _, p in parts if isinstance(p, dict) for k in p})[:8]
    return f"? no reader for keys {keys} (--raw)", False


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


def when(rec: dict) -> str:
    return rec.get("produced_at") or rec.get("at") or ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("filters", nargs="*", help="substrings; an arm is shown if any matches its tag")
    ap.add_argument("--log", default=DEFAULT_LOG, help=f"path to the results log (default {DEFAULT_LOG})")
    ap.add_argument("--raw", action="store_true", help="dump each matching record as JSON instead")
    ap.add_argument("--recent", type=int, metavar="N", help="the N newest results (by produced_at), one line each")
    ap.add_argument("--check", action="store_true", help="exit 1 if any record has no reader (lists them)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.log):
        print(f"no {args.log} -- run from the setup root", file=sys.stderr)
        return 2

    records = load(args.log)
    keep = lambda t: not args.filters or any(f in t for f in args.filters)  # noqa: E731
    sel = {t: r for t, r in records.items() if keep(t)}
    if not sel:
        print("no matching results", file=sys.stderr)
        return 3

    if args.raw:
        for tag in sorted(sel):
            print(json.dumps(sel[tag], indent=2))
        return 0

    if args.check:
        bad = [t for t, r in sel.items() if not render(r)[1]]
        print(f"{len(sel) - len(bad)}/{len(sel)} records rendered")
        for t in bad:
            print("  no reader:", t, render(sel[t])[0])
        return 1 if bad else 0

    if args.recent:
        for tag, rec in sorted(sel.items(), key=lambda tr: when(tr[1]))[-args.recent :]:
            print(f"{when(rec)[:16]}  {tag}  [{rec.get('note', '')}]")
            print("    " + render(rec)[0])
        return 0

    arms: dict[str, list[tuple[int | None, dict]]] = defaultdict(list)
    for tag, rec in sel.items():
        arm, step = _split_tag(tag)
        arms[arm].append((step, rec))
    for arm in sorted(arms):
        points = sorted(arms[arm], key=lambda p: (p[0] is None, p[0] or 0))
        span = sorted({when(r)[:10] for _, r in points if when(r)})
        stamp = "" if not span else span[0] if len(span) < 2 else f"{span[0]}..{span[-1]}"
        print(f"{arm}  [{stamp}]")
        for step, rec in points:
            print(f"    {'final' if step is None else step}: {render(rec)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
