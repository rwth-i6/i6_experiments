"""Guard: new knowledge replies are scored with Parakeet v3; the replies that existed on 2026-10-29 keep Whisper.

``knowledge_benchmark.default_knowledge_asr`` resolves ``asr="default"`` against the generated
``knowledge_whisper_tags.WHISPER_REPLIES``, keyed by the replies' inference job (``reply_key``), never the tag: renaming
an arm moves nothing (check_label_free_hash). Graph mode (setup .venv, ~1 min):
  [1] every frozen reply set is still scored with the default and resolves to Whisper (none stale); the declared default
      calls that resolve to Parakeet are listed (each a NEW reply set, scored in the v3 table);
  [2] can fail: replies of a job not in the list resolve to Parakeet v3, frozen ones to Whisper.

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
from i6_experiments.users.dorian_koch.speech_llm.knowledge_whisper_tags import WHISPER_REPLIES  # noqa: E402

print("[1] the graph's default resolutions")
calls = dict(kb.DEFAULT_ASR_CALLS)
stale = sorted(WHISPER_REPLIES - set(calls))
check(not stale, f"every frozen reply set is still scored ({len(stale)} stale: {stale[:3]})")
check(all(calls[k] is None for k in WHISPER_REPLIES if k in calls), "every frozen reply set resolves to Whisper")
new = sorted(k for k, a in calls.items() if a is not None)
print(f"       {len(new)} new reply set(s) on {kb.KNOWLEDGE_DEFAULT_ASR}: {new[:5]}")

print("[2] CAN FAIL: new replies get Parakeet v3")
from types import SimpleNamespace  # noqa: E402


def fake(key):
    job, out = key.rsplit("/", 1)
    return SimpleNamespace(creator=SimpleNamespace(_sis_id=lambda: job), path=out)


check(kb.default_knowledge_asr(fake("x/y/SpeechInference.guardNEWxxxx/out_dir")) == kb.KNOWLEDGE_DEFAULT_ASR,
      "replies of a new job -> Parakeet v3")  # fmt: skip
some = sorted(WHISPER_REPLIES)[0]
check(kb.default_knowledge_asr(fake(some)) is None, f"frozen replies ({some}) -> Whisper")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
