"""Guard: re-running ``write_config`` never moves a run_dir that holds checkpoints.

``prepare_run_dir`` (every training job's ``write_config``: SpeechFinetune, RLFinetune, RLFinetuneV2, Audex) moves a
non-empty run_dir aside so a re-run starts clean. ``run`` resumes from ``run_dir/checkpoints``, so moving one that holds
checkpoints (``hpc-rerun.py --redo-task write_config`` on a run that already trained) restarted it from step 0 under the
same hash. CAN FAIL: the old code moved it with a warning.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_run_dir_redo.py
"""

import os
import sys
import tempfile

os.environ.setdefault("CUDA_HOME", "/usr")

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "sisyphus"))

from i6_experiments.users.dorian_koch.speech_llm.finetune import prepare_run_dir  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    run_dir = os.path.join(tmp, "output", "run_dir")
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    open(os.path.join(run_dir, "metrics.train.jsonl"), "w").close()
    prepare_run_dir(run_dir)  # no checkpoint yet: moved aside, the re-run starts clean
    assert not os.path.exists(run_dir) and os.path.isdir(os.path.join(tmp, "moshi_finetune_old_runs", "0001"))

    os.makedirs(os.path.join(run_dir, "checkpoints", "checkpoint_000200"))
    try:
        prepare_run_dir(run_dir)
    except RuntimeError as e:
        assert "checkpoint_000200" in str(e), e
    else:
        raise AssertionError("a run_dir holding checkpoints was moved aside: the run would restart from step 0")
    assert os.path.isdir(os.path.join(run_dir, "checkpoints", "checkpoint_000200"))
    assert not os.path.exists(os.path.join(tmp, "moshi_finetune_old_runs", "0002"))
print("check_run_dir_redo: a run_dir without checkpoints is moved aside, one with checkpoints is refused and kept")
