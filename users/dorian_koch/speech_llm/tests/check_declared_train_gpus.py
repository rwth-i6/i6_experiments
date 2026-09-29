"""Guard: ``finetune.launch_training`` can size every job class that calls it.

``launch_training`` caps the torchrun ranks at the GPU count the run DECLARES. The 2026-09-23 fix read
``job.hparams`` unguarded, and ``RLFinetune`` has no ``hparams``. So the first RL run after it, the PersonaPlex
knowledge-GRPO smoke of 2026-09-29, crashed with ``AttributeError`` after its vLLM judge had loaded. Through the REAL
``declared_train_gpus`` and a REAL ``RLFinetune`` (and stand-ins for the SpeechFinetune shapes):

  1. RLFinetune (no hparams, no compute) -> 1.
  2. compute={"gpu": 4} -> 4; hparams={"gpu": 2} -> 2; compute wins over hparams.
  3. CAN FAIL: the pre-fix expression raises AttributeError on the RLFinetune.

Setup ``.venv``, CPU, seconds:
    .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_declared_train_gpus.py
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))

from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.finetune import declared_train_gpus  # noqa: E402
from i6_experiments.users.dorian_koch.speech_llm.rl_finetune import RLFinetune  # noqa: E402


def main():
    rl = RLFinetune(venv_python_path=tk.Path("/nonexistent/python"))
    checks = {
        "RLFinetune -> 1": declared_train_gpus(rl) == 1,
        "compute 4": declared_train_gpus(types.SimpleNamespace(compute={"gpu": 4}, hparams={})) == 4,
        "hparams 2": declared_train_gpus(types.SimpleNamespace(compute=None, hparams={"gpu": 2})) == 2,
        "compute wins": declared_train_gpus(types.SimpleNamespace(compute={"gpu": 3}, hparams={"gpu": 2})) == 3,
    }
    try:
        int((getattr(rl, "compute", None) or {}).get("gpu", rl.hparams.get("gpu", 1)))
        checks["pre-fix expression fails on RLFinetune"] = False
    except AttributeError:
        checks["pre-fix expression fails on RLFinetune"] = True
    for k, v in checks.items():
        print(f"[{'ok' if v else 'FAIL'}] {k}")
    if not all(checks.values()):
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
