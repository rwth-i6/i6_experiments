"""InteractionBench (arXiv 2611.02430) for moshi_family models, graded by a local judge.

The benchmark's own graders and aggregation run unmodified from a pinned checkout; the model is driven offline with
the user tracks upstream fed Moshi and PersonaPlex; Gemini 2.5 Flash is replaced by a local vLLM judge and MLX
Parakeet by NeMo Parakeet (the same checkpoint). Code: ``speech_llm/full_duplex/interaction_bench`` (its package
docstring has the design, ``protocol`` the deviations). Numbers are internal ("MIB"): not comparable with the paper's.

Jobs:
  * ``InteractionBenchCode``      -- clone the harness at ``IB_CODE_COMMIT``.
  * ``InteractionBenchData``      -- download released files from the Drive folder: ``release_stimuli.json`` (user
    audio + benchmark.json, 534 files) and, on top, ``release_runs.json`` (the released Moshi / PersonaPlex runs for
    the judge-swap check, 2352 files). Both lists are a selection from the folder's listing of 2026-11-16.
  * ``InteractionBenchInference`` -- ``interaction_bench.driver``: one run dir per item.
  * ``InteractionBenchGrade``     -- ``interaction_bench.grade`` under a co-launched vLLM judge: ``summary.json``.
  * ``interaction_bench_py`` / ``interaction_bench_release_check_py`` -- wiring.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

from sisyphus import Job, Task, tk

IB_CODE_REPO = "https://github.com/avera-labs/InteractionBench"
#: the harness revision our graders import (2026-09-24, the paper's release); hash-excluded at this value
IB_CODE_COMMIT = "414e52abb18e8ce3e11e41e859eb382501da19b8"
IB_JUDGE_MODEL = "Qwen/Qwen3.8-27B"
#: the grader venv's pins: upstream's uv.lock at IB_CODE_COMMIT (google-genai left out: the judge is local) + gdown
IB_GRADER_PACKAGES = [
    "python-dotenv==1.2.2",
    "pydub==0.25.1",
    "numpy==2.4.6",
    "scipy==1.17.1",
    "soundfile==0.14.0",
    "webrtcvad==2.0.10",
    "setuptools==81.0.0",
    "praat-parselmouth==0.4.7",
    "librosa==0.11.0",
    "numba==0.66.0",
    "llvmlite==0.48.0",
    "openai==2.53.0",
    "gdown==5.2.0",
]


def ib_grader_venv() -> tk.Path:
    from i6_experiments.users.dorian_koch.jobs.venv import CreateVenv

    venv = CreateVenv(packages=[IB_GRADER_PACKAGES], hash_overwrite="interaction_bench_grader_v1")
    tk.register_output("venv/interaction_bench_grader", venv.out_env_path)
    return venv.out_python_path


def _full_duplex_dir() -> str:
    from .inference_harness import _moshi_family_pythonpath

    return _moshi_family_pythonpath()


class InteractionBenchCode(Job):
    """The harness repo at a pinned commit (login node, needs internet). Output dir = the repo root."""

    __sis_hash_exclude__ = {"commit": IB_CODE_COMMIT}

    def __init__(self, *, repo_url: str = IB_CODE_REPO, commit: str = IB_CODE_COMMIT):
        self.repo_url = repo_url
        self.commit = commit
        self.out_dir = self.output_path("code", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        out = self.out_dir.get_path()
        tmp = out + ".gitclone"
        subprocess.run(["git", "clone", "-q", self.repo_url, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "-q", self.commit], check=True)
        head = subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        assert head.stdout.strip() == self.commit, (head.stdout, self.commit)
        for name in os.listdir(tmp):
            os.rename(os.path.join(tmp, name), os.path.join(out, name))
        os.rmdir(tmp)


class InteractionBenchData(Job):
    """The released files named in ``file_list`` (``[{"id", "path"}]``, a listing of the Drive folder), downloaded
    one by one with gdown on the login node; ``base_dir`` (another data job's ``out_dir``) is copied in first.

    Each file comes straight from ``drive.usercontent.google.com`` (gdown only as the fallback): gdown's ``uc`` page
    got throttled after 86 files on a laptop and after 250 on the login node (2026-11-16) while the direct endpoint
    still served. Paced, resumable (a present file is kept), retried with backoff; a reply that is an HTML page is a
    failure, not a file (a stale ``.part`` is overwritten by the next try). Files still missing at the end fail the
    task with their count, and rerunning it continues where it stopped. ``out_dir`` mirrors the folder's paths.
    """

    DIRECT_URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

    def _direct(self, file_id: str, part: str) -> str:
        """Fetch one file to ``part``; '' on success, else the reason."""
        import urllib.request

        try:
            req = urllib.request.Request(self.DIRECT_URL.format(file_id), headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if "text/html" in r.headers.get("Content-Type", ""):
                    return f"html reply ({r.status})"
                data = r.read()
            with open(part, "wb") as fh:
                fh.write(data)
            return ""
        except Exception as e:  # noqa: BLE001 -- any network error is a retry
            return repr(e)

    __sis_hash_exclude__ = {"base_dir": None}

    def __init__(
        self, *, file_list: tk.Path, grader_python: tk.Path, pace_s: float = 1.0, base_dir: tk.Path | None = None
    ):
        self.file_list = file_list
        self.grader_python = grader_python
        self.pace_s = pace_s
        self.base_dir = base_dir
        self.out_dir = self.output_path("data", directory=True)
        self.out_manifest = self.output_path("manifest.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import shutil

        files = json.load(open(self.file_list.get_path()))
        out = self.out_dir.get_path()
        if self.base_dir is not None:  # another data job's files under the same root (the stimuli under the runs)
            shutil.copytree(self.base_dir.get_path(), out, dirs_exist_ok=True)
        todo = [f for f in files if not os.path.exists(os.path.join(out, f["path"]))]
        print(f"[ib-data] {len(files)} files listed, {len(todo)} to fetch", flush=True)
        failed = []
        for i, f in enumerate(todo):
            dst = os.path.join(out, f["path"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ok = False
            for attempt in range(5):
                why = self._direct(f["id"], dst + ".part")
                if why and attempt == 4:  # last try: gdown
                    r = subprocess.run(
                        [self.grader_python.get(), "-m", "gdown", "-q", f["id"], "-O", dst + ".part"],
                        capture_output=True, text=True,
                    )  # fmt: skip
                    why = "" if r.returncode == 0 else r.stderr.strip()[-300:]
                if not why and os.path.exists(dst + ".part"):
                    os.rename(dst + ".part", dst)
                    ok = True
                    break
                time.sleep(30 * (attempt + 1))
            if not ok:
                failed.append(f["path"])
                print(f"[ib-data] FAILED {f['path']}: {why}", flush=True)
                if len(failed) >= 20:  # throttled: stop, a rerun resumes
                    break
            if (i + 1) % 50 == 0:
                print(f"[ib-data] {i + 1}/{len(todo)}", flush=True)
            time.sleep(self.pace_s)
        missing = [f["path"] for f in files if not os.path.exists(os.path.join(out, f["path"]))]
        if missing:
            raise RuntimeError(f"{len(missing)} of {len(files)} files missing (first: {missing[:3]}); rerun to resume")
        with open(self.out_manifest.get_path(), "w") as fh:
            json.dump({"files": len(files), "list": self.file_list.get_path()}, fh)


class InteractionBenchInference(Job):
    """``interaction_bench.driver``: every item's run dir (``out_dir/<root>/<task>/<item>/live_<id>_sut``).

    Backend identity as hashable primitives (as ``MTRDuplexInference``): the inference venv's python, the driver's
    backend id, the HF repo, the finetune overlay (+ LoRA rank). ``task_names`` / ``limit`` restrict the items (smoke);
    ``()`` / 0 = all, hash-excluded at those values."""

    __sis_hash_exclude__ = {"task_names": (), "limit": 0, "overlay": None, "lora_rank": None, "lora_scaling": 2.0}

    def __init__(
        self,
        *,
        data_dir: tk.Path,
        inference_venv,
        backend: str,
        hf_repo: str,
        seed: int = 0,
        overlay: tk.Path | None = None,
        lora_rank: int | None = None,
        lora_scaling: float = 2.0,
        task_names: tuple = (),
        limit: int = 0,
    ):
        self.data_dir = data_dir
        self.inference_venv = inference_venv
        self.backend = backend
        self.hf_repo = hf_repo
        self.seed = seed
        self.overlay = overlay
        self.lora_rank = lora_rank
        self.lora_scaling = lora_scaling
        self.task_names = tuple(task_names)
        self.limit = limit
        self.out_dir = self.output_path("runs", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 48, "time": 2 if limit else 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .mtr_duplexbench import _pythonpath_env

        python = self.inference_venv.get()  # a Variable (venv python path), as MTRDuplexInference
        cmd = [python, "-m", "interaction_bench.driver", "--data_root", self.data_dir.get_path(), "--out_dir",
               self.out_dir.get_path(), "--backend", self.backend, "--hf_repo", self.hf_repo, "--seed", str(self.seed)]  # fmt: skip
        if self.overlay is not None:
            cmd += ["--overlay", self.overlay.get_path()]
        if self.lora_rank is not None:
            cmd += ["--lora_rank", str(self.lora_rank), "--lora_scaling", str(self.lora_scaling)]
        if self.task_names:
            cmd += ["--tasks", ",".join(self.task_names)]
        if self.limit:
            cmd += ["--limit", str(self.limit)]
        print("[ib-inference]", " ".join(cmd), flush=True)
        subprocess.run(cmd, env=_pythonpath_env(), check=True)


class InteractionBenchGrade(Job):
    """``interaction_bench.grade`` with a co-launched vLLM judge and the NeMo Parakeet worker on the same GPU.

    ``runs`` = an inference job's ``out_dir``; or ``release_system`` (``moshi``, ``personaplex``) re-grades that
    system's released recordings from ``data_dir`` and compares every verdict with upstream's (the judge-swap check,
    ``summary["released"]``). ``out_tree`` keeps every transcript, grade file and judge reason."""

    __sis_hash_exclude__ = {"code_version": 1}

    def __init__(
        self,
        *,
        code_dir: tk.Path,
        grader_python: tk.Path,
        nemo_python: tk.Path,
        runs: tk.Path | None = None,
        release_system: str | None = None,
        data_dir: tk.Path | None = None,
        judge_model: str = IB_JUDGE_MODEL,
        code_version: int = 1,
    ):
        assert (runs is None) != (release_system is None), "runs or release_system"
        assert release_system is None or data_dir is not None
        self.code_dir = code_dir
        self.grader_python = grader_python
        self.nemo_python = nemo_python
        self.runs = runs
        self.release_system = release_system
        self.data_dir = data_dir
        self.judge_model = judge_model
        self.code_version = code_version
        self.out_tree = self.output_path("tree", directory=True)
        self.out_summary = self.output_path("summary.json")
        from .common import vllm_gpu_mem_gb

        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 48, "time": 6, "gpu_mem_gb": vllm_gpu_mem_gb(judge_model)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .common import vllm_server
        from .mtr_duplexbench import _pythonpath_env

        fd = _full_duplex_dir()
        cmd = [self.grader_python.get(), "-m", "interaction_bench.grade", "--upstream", self.code_dir.get_path(),
               "--tree", self.out_tree.get_path(), "--out_summary", self.out_summary.get_path(),
               "--nemo_python", self.nemo_python.get(),
               "--parakeet_worker", os.path.join(fd, "moshi_family", "rl", "parakeet_worker.py")]  # fmt: skip
        if self.runs is not None:
            cmd += ["--runs", self.runs.get_path()]
        else:
            cmd += ["--release_system", self.release_system, "--data_root", self.data_dir.get_path()]
        # 0.85: the Parakeet worker (~3 GiB) shares the card, as beside the RL judge (common._VLLM_MODEL_ARGS)
        with vllm_server(self.judge_model, gpu_memory_utilization=0.85) as url:
            env = _pythonpath_env()
            env.update(IB_JUDGE_BASE_URL=url, IB_JUDGE_MODEL=self.judge_model)
            print("[ib-grade]", " ".join(cmd), flush=True)
            subprocess.run(cmd, env=env, check=True)


def _shared(with_runs: bool = False):
    """(code job, grader python, data job: stimuli, or stimuli + released runs, NeMo python)."""
    from .common import moshi_family_lib_parent
    from .fdb import fdb_asr_venv

    code = InteractionBenchCode()
    grader = ib_grader_venv()
    lists = os.path.join(moshi_family_lib_parent(), "interaction_bench")
    data = InteractionBenchData(file_list=tk.Path(os.path.join(lists, "release_stimuli.json")), grader_python=grader)
    tk.register_output("benchmark/interaction_bench/code", code.out_dir)
    tk.register_output("benchmark/interaction_bench/data", data.out_dir)
    if with_runs:
        data = InteractionBenchData(file_list=tk.Path(os.path.join(lists, "release_runs.json")), grader_python=grader,
                                    base_dir=data.out_dir)  # fmt: skip
        tk.register_output("benchmark/interaction_bench/data_with_released_runs", data.out_dir)
    return code, grader, data, fdb_asr_venv()


def interaction_bench_py(
    *,
    tag: str,
    inference_venv,
    backend: str,
    hf_repo: str,
    overlay: tk.Path | None = None,
    lora_rank: int | None = None,
    seed: int = 0,
    task_names: tuple = (),
    limit: int = 0,
    judge_model: str = IB_JUDGE_MODEL,
):
    """Inference + grading of one model; registers ``benchmark/interaction_bench/<tag>/{summary.json,tree,runs}``."""
    code, grader, data, nemo = _shared()
    inf = InteractionBenchInference(
        data_dir=data.out_dir, inference_venv=inference_venv, backend=backend, hf_repo=hf_repo, seed=seed,
        overlay=overlay, lora_rank=lora_rank, task_names=task_names, limit=limit,
    )  # fmt: skip
    gr = InteractionBenchGrade(code_dir=code.out_dir, grader_python=grader, nemo_python=nemo, runs=inf.out_dir,
                               judge_model=judge_model)  # fmt: skip
    tk.register_output(f"benchmark/interaction_bench/{tag}/runs", inf.out_dir)
    tk.register_output(f"benchmark/interaction_bench/{tag}/tree", gr.out_tree)
    tk.register_output(f"benchmark/interaction_bench/{tag}/summary.json", gr.out_summary)
    return gr.out_summary


def interaction_bench_release_check_py(system: str, *, judge_model: str = IB_JUDGE_MODEL):
    """Re-grade upstream's released ``system`` recordings with our ASR + judge; per-task agreement in the summary."""
    code, grader, data, nemo = _shared(with_runs=True)
    gr = InteractionBenchGrade(code_dir=code.out_dir, grader_python=grader, nemo_python=nemo, release_system=system,
                               data_dir=data.out_dir, judge_model=judge_model)  # fmt: skip
    tk.register_output(f"benchmark/interaction_bench/release_check/{system}/summary.json", gr.out_summary)
    tk.register_output(f"benchmark/interaction_bench/release_check/{system}/tree", gr.out_tree)
    return gr.out_summary
