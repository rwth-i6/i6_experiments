"""Guard: the PersonaPlex suite runs MFDB at batch 16 under ``<tag>_b16``, except tags that finished at batch 1.

Loads the real config with ``fdb_benchmark_py`` recorded, then checks every ``ppx_suite`` call:
[1] a tag with a finished batch-1 MFDB inference is in ``MFDB_B1_TAGS`` (else its hash moved and it re-runs);
[2] a frozen tag is declared at batch 1 under its own name; [3] every other tag at batch 16 under ``<tag>_b16``;
[4] mutant: dropping a finished tag from the frozen set is caught by [1].
Run (setup root, ~2 min): ``CUDA_HOME=/usr .venv/bin/python <this file>``.
"""

import inspect
import logging
import os
import sys

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

import i6_experiments.users.dorian_koch.speech_llm.fdb as fdbmod  # noqa: E402

calls = []
_real = fdbmod.fdb_benchmark_py


def _record(*a, **kw):
    out = _real(*a, **kw)
    if inspect.stack()[1].function == "ppx_suite":
        calls.append((kw["tag"], kw.get("batch_size"), out))
    return out


fdbmod.fdb_benchmark_py = _record

from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
from speech_llm.full_duplex.sis_recipe.doriank import mfdb_protocol as mp  # noqa: E402


def finished_b1(out) -> bool:
    for job in out.values():
        for inp in job._sis_inputs:
            c = getattr(inp, "creator", None)
            if c is not None and type(c).__name__ == "SpeechInference":
                p = c._sis_path()
                if os.path.exists(p + "/finished") or os.path.exists(p + "/finished.tar.gz"):
                    return True
    return False


def check(frozen) -> list:
    bad = []
    for tag, batch, out in calls:
        base = tag[: -len(f"_b{mp.MFDB_SUITE_BATCH}")] if batch else tag
        if batch is None and finished_b1(out) and base not in frozen:
            bad.append(f"[1] {tag}: finished at batch 1 but not frozen")
        if base in frozen and (batch is not None or tag != base):
            bad.append(f"[2] {tag}: frozen tag not at batch 1 under its own name")
        if base not in frozen and (batch != mp.MFDB_SUITE_BATCH or tag != f"{base}_b{mp.MFDB_SUITE_BATCH}"):
            bad.append(f"[3] {tag}: new tag not at batch {mp.MFDB_SUITE_BATCH} under _b{mp.MFDB_SUITE_BATCH}")
    return bad


assert calls, "no ppx_suite MFDB call recorded"
bad = check(mp.MFDB_B1_TAGS)
n16 = sum(b == mp.MFDB_SUITE_BATCH for _, b, _ in calls)
print(f"{len(calls)} suite MFDB calls: {len(calls) - n16} batch 1 (frozen), {n16} batch 16")
victim = next(t for t, b, o in calls if b is None and finished_b1(o))
mutant = check(mp.MFDB_B1_TAGS - {victim})
ok = not bad and any(m.startswith("[1]") for m in mutant)
print("\n".join(bad) or "[1]-[3] ok", "| [4] mutant caught" if mutant else "| [4] mutant NOT caught")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
