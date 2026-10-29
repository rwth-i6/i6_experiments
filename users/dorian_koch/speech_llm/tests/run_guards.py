"""Run every standalone guard (``check_*.py`` / ``test_*.py``) and report which pass, fail or time out.

The guards live in three places and run in three kinds of interpreter; ``guards_manifest.json`` (next to this file) says
for each guard how it runs:

* ``env``: ``login`` = the login node's ``python3``; ``setup`` = the setup's ``.venv`` (``CUDA_HOME=/usr``); ``module`` =
  the setup ``.venv`` with ``PYTHONPATH=recipe/speech_llm/full_duplex:recipe``, run as ``-m moshi_family.rl.tests.<name>``
  (a file without a ``__main__`` block under pytest instead); ``venv:<name>`` = the ``CreateVenv.<name>`` job venv (a
  packed one mounts itself), same ``PYTHONPATH``.
* ``args``: extra argv (e.g. ``["--graph"]``); ``graph``: loads the recipe config (at most ``MAX_GRAPH_PARALLEL`` at
  once); ``gpu``: needs a GPU (skipped unless ``--gpu``); ``timeout`` (s).

A guard with several modes (unit tests in a job venv, ``--graph`` in the setup venv) has a list of such specs.

A guard missing from the manifest is reported as ``unlisted`` (add it); a manifest entry without a file as ``missing``.
Run from the setup root, never for long on the login node: ``--sbatch`` submits the same command as a CPU job.

    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/run_guards.py --list
    python3 .../run_guards.py --sbatch [--jobs 16]          # all CPU guards in one SLURM job, report in OUT_ROOT
    python3 .../run_guards.py --only legacy_pins awaited     # name substrings, run here (short ones only)
    python3 .../run_guards.py --report                       # the newest report

Standard library only (the login node's python3, 3.9).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "guards_manifest.json")
#: Guard directories, relative to the setup root.
ROOTS = (
    "recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_*.py",
    "recipe/speech_llm/full_duplex/rl_spec/tests/check_*.py",
    "recipe/speech_llm/full_duplex/moshi_family/rl/tests/test_*.py",
)
OUT_ROOT = os.path.join("/hpcwork", os.environ.get("USER", "tt201262"), "guard_runs")
VENV_GLOB = "work/i6_experiments/users/dorian_koch/jobs/venv/CreateVenv.{name}/output/venv/bin/python"
DEFAULT_TIMEOUT = 900
#: Graph guards load the whole config (~9 s, a few GB each): at most this many at once.
MAX_GRAPH_PARALLEL = 4


def setup_root() -> str:
    d = HERE
    while d != "/":
        if os.path.isdir(os.path.join(d, "recipe")) and os.path.isfile(os.path.join(d, "settings.py")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("run_guards: no setup root above " + HERE)


def discover(root: str) -> list:
    out = []
    for pat in ROOTS:
        out += sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, pat)))
    return out


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def command(path: str, spec: dict, root: str):
    """(argv, env) of one guard."""
    env = dict(os.environ, CUDA_HOME="/usr", HF_HUB_OFFLINE="1")
    fd = os.path.join(root, "recipe/speech_llm/full_duplex")
    kind = spec["env"]
    args = list(spec.get("args", []))
    if kind == "login":
        return ["python3", path, *args], env
    setup_py = os.path.join(root, ".venv/bin/python")
    if kind == "setup":
        return [setup_py, path, *args], env
    env["PYTHONPATH"] = (
        fd + ":" + os.path.join(root, "recipe") + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    if kind == "module":
        with open(os.path.join(root, path)) as f:
            has_main = "__main__" in f.read()
        if not has_main:  # pytest-style: run as a module it would exit 0 without testing anything
            return [setup_py, "-m", "pytest", "-q", "-p", "no:cacheprovider", path, *args], env
        mod = "moshi_family.rl.tests." + os.path.splitext(os.path.basename(path))[0]
        return [setup_py, "-m", mod, *args], env
    if kind.startswith("venv:"):
        py = os.path.join(root, VENV_GLOB.format(name=kind.split(":", 1)[1]))
        return [py, path, *args], env
    raise ValueError(f"{path}: unknown env {kind!r}")


def run_one(path: str, spec: dict, root: str) -> dict:
    argv, env = command(path, spec, root)
    t0 = time.time()
    try:
        p = subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True,
                           timeout=spec.get("timeout", DEFAULT_TIMEOUT), errors="replace")  # fmt: skip
        status = "pass" if p.returncode == 0 else "fail"
        tail = (p.stdout + p.stderr)[-3000:]
    except subprocess.TimeoutExpired as e:
        status = "timeout"
        tail = ((e.stdout or "") if isinstance(e.stdout, str) else "")[-3000:]
    except OSError as e:  # e.g. a venv that was never built
        status, tail = "error", repr(e)
    label = " ".join([path, *spec.get("args", [])])
    return {"guard": label, "status": status, "seconds": round(time.time() - t0, 1), "env": spec["env"], "tail": tail}


def select(root: str, manifest: dict, only, gpu: bool):
    found = discover(root)
    unlisted = [p for p in found if p not in manifest]
    missing = [p for p in manifest if p not in found]
    todo = [
        (p, s)
        for p in found
        if p in manifest
        for s in (manifest[p] if isinstance(manifest[p], list) else [manifest[p]])
    ]
    if only:
        todo = [(p, s) for p, s in todo if any(o in p for o in only)]
    skipped = [p for p, s in todo if s.get("gpu") and not gpu]
    todo = [(p, s) for p, s in todo if gpu or not s.get("gpu")]
    return todo, skipped, unlisted, missing


def run(root: str, args) -> int:
    manifest = load_manifest()
    todo, skipped, unlisted, missing = select(root, manifest, args.only, args.gpu)
    graph = [t for t in todo if t[1].get("graph")]
    rest = [t for t in todo if t not in graph]
    print(f"run_guards: {len(todo)} to run ({len(graph)} graph), {len(skipped)} GPU skipped, {len(unlisted)} unlisted, "
          f"{len(missing)} missing; {args.jobs} workers", flush=True)  # fmt: skip
    results = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.jobs) as pool, cf.ThreadPoolExecutor(MAX_GRAPH_PARALLEL) as gpool:
        futs = [pool.submit(run_one, p, s, root) for p, s in rest] + [
            gpool.submit(run_one, p, s, root) for p, s in graph
        ]
        for f in cf.as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  {r['status']:7s} {r['seconds']:7.1f}s  {r['guard']}", flush=True)
    results.sort(key=lambda r: r["guard"])
    report = {
        "at": dt.datetime.now().isoformat(timespec="seconds"),
        "host": os.uname().nodename,
        "wall_s": round(time.time() - t0, 1),
        "results": results,
        "skipped_gpu": skipped,
        "unlisted": unlisted,
        "missing": missing,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"guards_{report['at'].replace(':', '')}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print_report(report, path)
    return 0 if all(r["status"] == "pass" for r in results) and not unlisted and not missing else 1


def print_report(report: dict, path: str = "") -> None:
    res = report["results"]
    by = {}
    for r in res:
        by.setdefault(r["status"], []).append(r)
    cpu = sum(r["seconds"] for r in res)
    print(f"\nguard run {report['at']} on {report['host']}: wall {report['wall_s']:.0f}s, summed {cpu:.0f}s  {path}")
    print("  " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by.items())) +
          f", GPU skipped {len(report['skipped_gpu'])}, unlisted {len(report['unlisted'])}, missing {len(report['missing'])}")  # fmt: skip
    for k in ("fail", "timeout", "error"):
        for r in by.get(k, []):
            last = [ln for ln in r["tail"].strip().splitlines() if ln.strip()][-1:] or [""]
            print(f"  {k.upper():7s} {r['guard']} ({r['env']}, {r['seconds']:.0f}s): {last[0][:160]}")
    for p in report["unlisted"]:
        print(f"  UNLISTED {p}: add it to guards_manifest.json")
    for p in report["missing"]:
        print(f"  MISSING  {p}: in the manifest, no such file")
    slow = sorted(res, key=lambda r: -r["seconds"])[:5]
    print("  slowest: " + ", ".join(f"{os.path.basename(r['guard'])} {r['seconds']:.0f}s" for r in slow))


def sbatch(root: str, args) -> int:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = os.path.join("/hpcwork", os.environ.get("USER", "tt201262"), "sbatch", "guards")
    os.makedirs(d, exist_ok=True)
    me = os.path.relpath(os.path.abspath(__file__), root)
    only = (" --only " + " ".join(args.only)) if args.only else ""
    script = os.path.join(d, f"guards_{stamp}.sh")
    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=run_guards
#SBATCH --cpus-per-task={args.jobs}
#SBATCH --mem={args.mem}G
#SBATCH --time={args.time}
#SBATCH --output={d}/guards_{stamp}.log
set -u
export HF_HOME=/hpcwork/p0023999/common_hf_home
cd {root}
python3 {me} --jobs {args.jobs} --out {args.out}{only}
""")
    out = subprocess.run(["sbatch", script], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip())
    print(f"log: {d}/guards_{stamp}.log; report: {args.out}/guards_<time>.json (--report)")
    return out.returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the guards and how each runs")
    ap.add_argument("--report", action="store_true", help="print the newest report")
    ap.add_argument("--sbatch", action="store_true", help="submit the run as a CPU SLURM job")
    ap.add_argument("--only", nargs="+", help="guards whose path contains any of these")
    ap.add_argument("--gpu", action="store_true", help="also run the guards marked gpu (needs a GPU)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--mem", type=int, default=96, help="GB, --sbatch")
    ap.add_argument("--time", default="3:00:00", help="--sbatch")
    ap.add_argument("--out", default=OUT_ROOT)
    args = ap.parse_args(argv)
    root = setup_root()
    if args.list:
        manifest = load_manifest()
        todo, skipped, unlisted, missing = select(root, manifest, args.only, True)
        for p, s in todo:
            print(f"{s['env']:30s} {'gpu ' if s.get('gpu') else '    '}{' '.join(s.get('args', [])):10s} {p}")
        for p in unlisted:
            print(f"{'UNLISTED':30s}     {'':10s} {p}")
        for p in missing:
            print(f"{'MISSING':30s}     {'':10s} {p}")
        return 0
    if args.report:
        runs = sorted(glob.glob(os.path.join(args.out, "guards_*.json")))
        if not runs:
            print("no guard run yet")
            return 1
        with open(runs[-1]) as f:
            print_report(json.load(f), runs[-1])
        return 0
    if args.sbatch:
        return sbatch(root, args)
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
