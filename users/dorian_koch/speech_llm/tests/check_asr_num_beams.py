"""Guard: ``RLFinetune(quality_asr_num_beams=...)`` is hash-neutral at its default 5 and reaches the launcher config.

Every ASR-graded RL arm up to 2026-10-11 judged beam-5 Whisper text; greedy (1) is the faster option (asr_bench
2026-10-05). Can fail: the new knob re-hashes existing arms (default not excluded); a non-default value does NOT
re-hash (two arms judged on different text sharing one identity); the value never reaches the rendered config.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_asr_num_beams.py
"""

import os
import sys
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import sisyphus.job  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.rl_finetune import RLFinetune, _render_rl_config  # noqa: E402

VENV = tk.Path("/dev/null/venv/python", hash_overwrite="beams_guard_venv")


def rl(**kw):
    sisyphus.job.created_jobs.clear()
    return RLFinetune(venv_python_path=VENV, max_steps=100, **kw)


def main():
    base, five, greedy = rl(), rl(quality_asr_num_beams=5), rl(quality_asr_num_beams=1)
    assert base._sis_id() == five._sis_id(), "beam 5 (the default) must not move the hash"
    assert greedy._sis_id() != base._sis_id(), "greedy must re-hash (different judged text = a different run)"
    for job, n in ((base, 5), (greedy, 1)):
        text = _render_rl_config(job)
        assert f"\nquality_asr_num_beams: {n}\n" in text, f"beam {n} not rendered"
    print("check_asr_num_beams: ok")


if __name__ == "__main__":
    main()
