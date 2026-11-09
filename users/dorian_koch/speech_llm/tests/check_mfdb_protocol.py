"""Guard: the PersonaPlex suite runs MFDB at batch 16 under ``<tag>_b16``, except models that finished at batch 1.

Loads the real config with ``fdb_benchmark_py`` recorded, then checks every ``ppx_suite`` call:
[1] a model with a finished batch-1 MFDB inference is in ``MFDB_B1_KEYS`` (else its hash moved and it re-runs);
[2] a frozen model is declared at batch 1 under its own name; [3] every other model at batch 16 under ``<tag>_b16``;
[4] mutant: dropping a finished model from the frozen set is caught by [1];
[5] label-free: a frozen model under another name keeps batch 1 (``check_label_free_hash`` renames arms).
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
        key = mp.mfdb_suite_key({k: v for k, v in kw.items() if k not in ("tag", "batch_size")})
        calls.append((kw["tag"], kw.get("batch_size"), out, key, kw))
    return out


fdbmod.fdb_benchmark_py = _record

from sisyphus.loader import config_manager  # noqa: E402

from speech_llm.full_duplex.sis_recipe.doriank import mfdb_protocol as mp  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])


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
    for tag, batch, out, key, _ in calls:
        base = tag[: -len(f"_b{mp.MFDB_SUITE_BATCH}")] if batch else tag
        if batch is None and finished_b1(out) and key not in frozen:
            bad.append(f"[1] {tag}: finished at batch 1 but not frozen")
        if key in frozen and (batch is not None or tag != base):
            bad.append(f"[2] {tag}: frozen model not at batch 1 under its own name")
        if key not in frozen and (batch != mp.MFDB_SUITE_BATCH or tag != f"{base}_b{mp.MFDB_SUITE_BATCH}"):
            bad.append(f"[3] {tag}: new model not at batch {mp.MFDB_SUITE_BATCH} under _b{mp.MFDB_SUITE_BATCH}")
    return bad


assert calls, "no ppx_suite MFDB call recorded"
frozen = set(mp.MFDB_B1_KEYS)
bad = check(frozen)
n16 = sum(c[1] == mp.MFDB_SUITE_BATCH for c in calls)
print(f"{len(calls)} suite MFDB calls: {len(calls) - n16} batch 1 (frozen), {n16} batch 16")
victim = next(c for c in calls if c[1] is None and finished_b1(c[2]))
mutant = check(frozen - {victim[3]})
renamed = mp.mfdb_suite_tag("zzguard_renamed", {k: v for k, v in victim[4].items() if k != "tag"})
label_free = renamed == ("zzguard_renamed", {})
ok = not bad and any(m.startswith("[1]") for m in mutant) and label_free
print("\n".join(bad) or "[1]-[3] ok", "| [4] mutant caught" if mutant else "| [4] mutant NOT caught",
      "| [5] renamed model keeps batch 1" if label_free else f"| [5] renamed model moved: {renamed}")  # fmt: skip
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
