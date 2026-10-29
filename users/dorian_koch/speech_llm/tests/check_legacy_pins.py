"""Guard: no new job copies a legacy pin (``legacy_pins.LEGACY_PINS``).

Pinned call sites keep a known-wrong value so finished jobs keep their hash (Echo-gold questions, unclamped podcast
slices, windowed Fisher corpora, rehearsal filters v1, CTC sidecars that train PAD over failed turns, MTR pause
scoring without re-detection). ``legacy_pins.check_graph()`` runs at the end of the recipe config and refuses any
job carrying such a value that is not one of the frozen ids.

  [1] the real graph: no violation, and every frozen id is still in the graph (a pin that names nothing is stale);
  [2] CAN FAIL: a NEW ``LLMPreprocess(gold_from_source=False)`` (a copied call site) is flagged, the same job with the
      default is not.

Run from the setup root (~2 min, one config load):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_legacy_pins.py
"""

import logging
import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
os.chdir(SETUP)
sys.path[:0] = [str(SETUP / "recipe"), str(SETUP / "recipe" / "sisyphus")]
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)

from sisyphus import graph, tk  # noqa: E402
from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
from i6_experiments.users.dorian_koch.speech_llm import legacy_pins as lp  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import LLMPreprocess  # noqa: E402

FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


print("[1] the graph")
jobs = list(graph.graph.jobs())
check(not lp.violations(jobs), f"no job copies a legacy pin ({lp.violations(jobs)[:2]})")
present = {(type(j).__name__, j._sis_id().rsplit("/", 1)[-1].split(".", 1)[1]) for j in jobs}
for (cls, attr, val), (_why, ids) in lp.LEGACY_PINS.items():
    stale = sorted(i for i in ids if (cls, i) not in present)
    check(not stale, f"{cls}.{attr}={val!r}: all {len(ids)} pinned jobs are in the graph (stale: {stale[:3]})")

print("[2] CAN FAIL: a copied call site")
probe = tk.Path("/dev/null/_legacy_pin_probe")
bad = LLMPreprocess(in_hf=probe, gold_from_source=False)
good = LLMPreprocess(in_hf=probe)
check(len(lp.violations([bad])) == 1, "LLMPreprocess(gold_from_source=False) on a new input is flagged")
check(not lp.violations([good]), "the same job with the default is not")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
