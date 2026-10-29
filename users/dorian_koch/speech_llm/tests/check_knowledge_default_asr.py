"""Guard: new knowledge tags are scored with Parakeet v3; the tags declared before 2026-10-29 keep Whisper.

``knowledge_benchmark.default_knowledge_asr`` resolves ``asr="default"`` against the generated
``knowledge_whisper_tags.WHISPER_TAGS``. Graph mode (setup .venv, ~1 min):
  [1] every frozen tag is still declared with the default and resolves to Whisper (none stale), and no declared default
      call resolves to Parakeet unexpectedly (lists them: each is a NEW tag, scored in the v3 table);
  [2] can fail: a tag not in the list resolves to Parakeet v3, a frozen one to Whisper.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_knowledge_default_asr.py
"""

import logging
import os
import sys

FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


setup = next(p for p in (os.getcwd(), *map(str, __import__("pathlib").Path(__file__).absolute().parents))
             if os.path.isdir(os.path.join(p, "recipe")))  # fmt: skip
os.chdir(setup)
sys.path[:0] = ["recipe", "recipe/sisyphus"]
os.environ.setdefault("CUDA_HOME", "/usr")
logging.disable(logging.WARNING)
from sisyphus.loader import config_manager  # noqa: E402

config_manager.load_configs(["recipe/speech_llm/full_duplex/sis_recipe/doriank/synthetic_train_data.py"])
from i6_experiments.users.dorian_koch.speech_llm import knowledge_benchmark as kb  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.knowledge_whisper_tags import WHISPER_TAGS  # noqa: E402

print("[1] the graph's default resolutions")
calls = dict(kb.DEFAULT_ASR_CALLS)
stale = sorted(WHISPER_TAGS - set(calls))
check(not stale, f"every frozen tag is still declared ({len(stale)} stale: {stale[:3]})")
check(all(calls[k] is None for k in WHISPER_TAGS if k in calls), "every frozen tag resolves to Whisper")
new = sorted(k for k, a in calls.items() if a is not None)
print(f"       {len(new)} new tag(s) on {kb.KNOWLEDGE_DEFAULT_ASR}: {new[:5]}")

print("[2] CAN FAIL: a new tag gets Parakeet v3")
check(kb.default_knowledge_asr("guard_new_tag_x", 1000) == kb.KNOWLEDGE_DEFAULT_ASR, "a new tag -> Parakeet v3")
some = sorted(WHISPER_TAGS)[0]
n = int(some.split("/n")[1].split("/")[0]) if "/n" in some else None
check(kb.default_knowledge_asr(some.rsplit("/", 1)[1], n) is None, f"a frozen tag ({some}) -> Whisper")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
