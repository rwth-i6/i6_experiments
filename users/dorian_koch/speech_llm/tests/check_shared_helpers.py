"""Guard: helpers that once existed as diverging copies stay single (code-duplication pass, 2026-11-04).

* ``completed_fraction``: every Job reads its progress markers through ``common.job_progress_fraction``. Seven copies
  read only ``<work>/progress.json`` and missed the per-GPU ``gpu<k>/progress.json`` files.
* torchcodec env: FFmpeg + CUDA NPP + libpython go on a worker's path only via ``common.add_torchcodec_to_env``.
  Three of six copies skipped a missing FFmpeg silently (the c25g libavutil crash).
* RL reward timing: rewards accumulate wall-clock only through ``judge_call.Timings`` (one subclass bypassed the
  parent's lock), and ``_ensure_client`` is defined once (``JsonJudge``).

Each rule is also run on a mutated copy that reintroduces the duplicate, to show it can fail.

Standard library only; run from the setup root:
    python3 recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_shared_helpers.py
"""

import ast
import sys
from pathlib import Path

SETUP = next((p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir()), Path.cwd())
I6 = SETUP / "recipe" / "i6_experiments" / "users" / "dorian_koch" / "speech_llm"
RL = SETUP / "recipe" / "speech_llm" / "full_duplex" / "moshi_family" / "rl"
FAILURES: list[str] = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


def progress_copies(sources: dict) -> list:
    """Modules whose ``completed_fraction`` opens a progress file itself instead of calling job_progress_fraction."""
    out = []
    for name, src in sources.items():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name == "completed_fraction":
                body = ast.get_source_segment(src, node)
                if "progress" in body and ("json.load" in body or "open(" in body):
                    out.append(f"{name}:{node.lineno}")
    return out


def env_copies(sources: dict) -> list:
    """Calls that assemble torchcodec's libraries (CUDA NPP / libpython) outside common.add_torchcodec_to_env. A hook
    adding only the FFmpeg binaries (ingest workers that shell out to ffmpeg) is not a torchcodec hook."""
    out = []
    for name, src in sources.items():
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if ("add_cuda_npp_to_env(" in code or "add_venv_python_lib_to_env(" in code) and "def " not in code:
                if name != "common.py":
                    out.append(f"{name}:{i}")
    return out


def timing_copies(sources: dict) -> list:
    """RL rewards keeping their own timing counters or client factory."""
    out = []
    for name, src in sources.items():
        for i, line in enumerate(src.splitlines(), 1):
            if "self._t_asr" in line or "self._t_judge" in line:
                out.append(f"{name}:{i}")
            if line.strip().startswith("def _ensure_client") and name != "judge_call.py":
                out.append(f"{name}:{i}")
    return out


def main():
    i6 = {p.name: p.read_text() for p in sorted(I6.glob("*.py"))}
    rl = {p.name: p.read_text() for p in sorted(RL.glob("*.py"))}
    check(len(i6) > 50 and len(rl) > 10, f"found the sources ({len(i6)} i6, {len(rl)} rl modules)")

    bad = progress_copies(i6)
    check(not bad, f"every completed_fraction goes through job_progress_fraction ({bad})")
    copy = {
        "x.py": 'class J:\n    def completed_fraction(self):\n        return json.load(open("progress.json"))["done"]\n'
    }
    check(progress_copies(copy) == ["x.py:2"], "...and a re-added copy is caught")

    bad = env_copies(i6)
    check(not bad, f"FFmpeg/NPP reach a worker only via add_torchcodec_to_env ({bad})")
    copy = {"x.py": "def env_hook(env):\n    InstallFFmpeg.add_to_env(p, env)\n    add_cuda_npp_to_env(py, env)\n"}
    check(env_copies(copy) == ["x.py:3"], "...and a re-added env hook is caught")

    bad = timing_copies(rl)
    check(not bad, f"RL rewards time through judge_call.Timings and one _ensure_client ({bad})")
    copy = {
        "x.py": "class R:\n    def score(self):\n        self._t_asr += 1.0\n    def _ensure_client(self):\n        pass\n"
    }
    check(timing_copies(copy) == ["x.py:3", "x.py:4"], "...and re-added counters / client factories are caught")

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED")
        sys.exit(1)
    print("\nshared helpers stay single")


if __name__ == "__main__":
    main()
