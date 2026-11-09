"""Guard: concurrent config loads may all write ``RESULTS_SINKED.json`` (``result_notify.write_sink_manifest``).

On 2026-11-09 a full guard run failed ``check_knowledge_default_asr`` with ``FileNotFoundError:
'RESULTS_SINKED.json.tmp' -> 'RESULTS_SINKED.json'``: two graph loads shared one temp name, one renamed it away under
the other. The manager, ``hash_snapshot.py`` and the graph guards all load the config, so this can hit any of them.
Drives the real function from 8 processes x 100 writes into one path; every write must succeed and the file must parse.
CAN FAIL: the shared temp name (``--mutant``) raises.
Run (setup root, ~10 s): ``CUDA_HOME=/usr .venv/bin/python <this file>`` [``--mutant``].
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

N_PROC, N_WRITES = 8, 100


def _writes(path: str, mutant: bool) -> str:
    import i6_experiments.users.dorian_koch.speech_llm.result_notify as rn

    if mutant:  # the pre-fix temp name, shared by every process
        real_getpid = os.getpid
        os.getpid = lambda: "shared"
    try:
        for _ in range(N_WRITES):
            rn.write_sink_manifest(path)
    except Exception as e:  # noqa: BLE001 -- reported to the parent
        return f"{type(e).__name__}: {e}"
    finally:
        if mutant:
            os.getpid = real_getpid
    return ""


def run(mutant: bool) -> list:
    with tempfile.TemporaryDirectory(dir="/tmp") as d:
        path = os.path.join(d, "RESULTS_SINKED.json")
        with mp.get_context("fork").Pool(N_PROC) as pool:
            errs = [e for e in pool.starmap(_writes, [(path, mutant)] * N_PROC) if e]
        json.load(open(path))
        left = [n for n in os.listdir(d) if n.endswith(".tmp")]
    return errs + [f"temp files left: {left}"] * bool(left)


if __name__ == "__main__":
    if "--mutant" in sys.argv:
        errs = run(True)
        print("mutant:", errs[:1] or "no error")
        sys.exit(0 if errs else 1)
    errs = run(False)
    mut = run(True)
    print(f"{N_PROC} processes x {N_WRITES} writes: {errs[:2] or 'all ok'}; mutant (shared temp name): "
          f"{'caught' if mut else 'NOT caught'} ({mut[:1]})")  # fmt: skip
    print("PASS" if not errs and mut else "FAIL")
    sys.exit(0 if not errs and mut else 1)
