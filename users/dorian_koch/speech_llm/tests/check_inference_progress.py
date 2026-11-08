"""Guard: a storage="hf" SpeechInference reports progress from the scratch its run writes into.

``run`` (cwd = the job's ``work/`` dir) writes replies into ``CLIP_SCRATCH`` and packs them at the end; the progress
hooks (``completed_fraction`` / ``info``) counted ``<job>/clip_scratch`` instead, so every storage="hf" job showed 0
until done. CAN FAIL: the old path reads 0 of 3.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_inference_progress.py
"""

import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import global_settings as gs  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.speech_inference import CLIP_SCRATCH, SpeechInference  # noqa: E402

with tempfile.TemporaryDirectory() as job_dir:
    work = os.path.join(job_dir, gs.JOB_WORK_DIR)
    os.makedirs(work)
    cwd = os.getcwd()
    os.chdir(work)  # the task's cwd, as sisyphus runs it
    try:
        scratch = os.path.abspath(CLIP_SCRATCH)  # where run() writes
    finally:
        os.chdir(cwd)
    os.makedirs(scratch)
    for i in range(3):
        open(os.path.join(scratch, f"q{i}.wav"), "w").close()
    job = object.__new__(SpeechInference)  # Job.__new__ needs a graph; the hooks read only these
    job.mode, job.storage, job._total_cache = "knowledge", "hf", 4
    job._sis_path = lambda name=None: os.path.join(job_dir, name) if name else job_dir
    assert job._progress_dir() == scratch, (job._progress_dir(), scratch)
    assert job.completed_fraction() == 0.75 and job.info() == "3/4 clips", (job.completed_fraction(), job.info())
print("check_inference_progress: storage=hf progress counts the scratch run() writes (3/4)")
