"""Guard: a finetune trains from the HF hub cache it was given, never silently from the shared one.

2026-10-17, merging upstream speech-llm: Robin Schmitt's ``model/schmitt/finetune.py`` subclasses ``SpeechFinetune`` and
swaps the cache by ``mock.patch.multiple(speech_llm.finetune, HF_HOME_DIR=..., HF_CACHE_DIR=...)``. Our env had moved
into ``common.worker_env`` (reads ``common``'s globals), so that patch would have been ignored -- or, with
``HF_HOME_DIR`` gone from the module, crash in ``mock``. Now: the official ``SpeechFinetune(hf_cache_dir=)`` (hash-excluded
at None), and a patched module global is refused with a hint (``finetune.training_hf_env``).

Drives Robin's real subclass ``run()`` with his exact patch (subprocess stubbed), and the env builder directly.
Run: ``CUDA_HOME=/usr .venv/bin/python projects/i6_experiments/users/dorian_koch/speech_llm/tests/check_hf_cache_dir.py``
"""

import os
import sys
from pathlib import Path
from unittest import mock

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm import common, finetune  # noqa: E402

FAILS = []


def check(ok, what):
    print(("[ok] " if ok else "[FAIL] ") + what)
    if not ok:
        FAILS.append(what)


OWN = "/hpcwork/p0023999/tt201262/check_hf_cache_dir/hub"


class _Job:
    def __init__(self, hf_cache_dir=None):
        if hf_cache_dir is not None:
            self.hf_cache_dir = hf_cache_dir


print("[1] the env builder")
env = finetune.training_hf_env(_Job())
check(env.get("HF_HUB_CACHE") == common.HF_CACHE_DIR.get(), "default: the shared hub cache")
check(env.get("HF_HOME") == common.HF_HOME_DIR.get(), "default: the shared HF_HOME")
env = finetune.training_hf_env(_Job(tk.Path(OWN)))
check(env.get("HF_HUB_CACHE") == OWN, f"hf_cache_dir: HF_HUB_CACHE is the job's own ({env.get('HF_HUB_CACHE')})")
check(env.get("HF_HOME") != common.HF_HOME_DIR.get(), "hf_cache_dir: HF_HOME is not forced to the shared one")
check(env.get("HF_HUB_OFFLINE") == "1", "hf_cache_dir: still offline")

print("[2] Robin's subclass with its own patch is refused, with the hint")
from speech_llm.full_duplex.model.schmitt.finetune import SpeechFinetune as RobinFinetune  # noqa: E402

job = RobinFinetune(
    hf_hub_cache=tk.Path(OWN),
    adapter=finetune.MOSHI_LIB_ADAPTER,
    venv_python_path=tk.Path("/nonexistent/python"),
    train_data=tk.Path("/nonexistent/train"),
)
launched = []
with mock.patch("subprocess.run", side_effect=lambda cmd, env=None, **kw: launched.append(env)):
    try:
        job.run()
        err = None
    except RuntimeError as e:
        err = str(e)
    except Exception as e:  # anything else: the guard's premise broke (e.g. setup before the env check)
        err = f"OTHER {e!r}"
check(err is not None and "hf_cache_dir=" in err, f"run() raises the hint (got {err and err[:90]!r})")
check(not launched, "nothing was launched with the shared cache")

print("[3] hashes")
check(
    finetune.SpeechFinetune.__sis_hash_exclude__.get("hf_cache_dir", "MISSING") is None, "hf_cache_dir excluded at None"
)

if FAILS:
    raise SystemExit(f"{len(FAILS)} check(s) failed")
print("all ok")
