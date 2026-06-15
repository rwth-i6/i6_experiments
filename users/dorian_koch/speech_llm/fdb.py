from glob import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
)
import os
import signal
import sys
import subprocess
from .common import HF_CACHE_DIR, vllm_server
from i6_experiments.users.dorian_koch.jobs.venv import CreateVenv
from functools import lru_cache
from contextlib import contextmanager
import shutil
import time
import random
import socket

from .moshi_client import MoshiFileClient, _ws_url, moshi_server

# `moshified_fdb_v1_v15` lives at recipe/moshified_fdb_v1_v15 (a sibling of i6_experiments).
# Sisyphus's RecipeFinder resolves recipe imports relative to the cwd, but a worker's cwd can
# change during run() (e.g. the moshi server block below), which breaks the lazy imports of
# moshified_fdb_v1_v15. Pin the recipe dir on sys.path so those imports work regardless of cwd.
import i6_experiments as _i6_experiments

_recipe_dir = os.path.dirname(os.path.dirname(_i6_experiments.__file__))
if _recipe_dir not in sys.path:
    sys.path.insert(0, _recipe_dir)


@lru_cache
def fdb_asr_venv():
    """Dedicated venv for the Full-Duplex-Bench NeMo ASR scoring (asr.py).

    The manager .venv pins NeMo 2.0.0 (forced by transformers>=5.5.0), which lacks
    transcribe(timestamps=True). This isolated venv installs NeMo >=2.2 so asr.py
    runs unmodified (upstream-style), invoked as a subprocess.
    """
    venv = CreateVenv(
        packages=[
            [
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            ["nemo_toolkit[asr]>=2.2", "soundfile"],
        ],
        hash_overwrite="fdb_asr_venv_v1",
    )
    tk.register_output("venv/fdb_asr", venv.out_env_path)
    return venv.out_python_path


@lru_cache
def fdb_eval_venv():
    """Dedicated venv for the Full-Duplex-Bench non-LLM scoring (the fork's evaluate.py).

    The fork eval scripts (backchannel / pause / turn-taking) need silero-vad + scipy +
    tqdm on top of torch/torchaudio/soundfile, and load wavs via soundfile (no torchcodec).
    Kept separate from the manager .venv so its torch/torchaudio churn (which made
    torchaudio.load hard-require torchcodec) can't break scoring. Versions are pinned to
    the manager .venv at port time so metrics stay identical; CPU wheels suffice since the
    eval runs as a login-node mini_task. Invoked as a subprocess via out_python_path.
    """
    venv = CreateVenv(
        packages=[
            [
                "torch==2.11.0",
                "torchaudio==2.11.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            ["silero-vad==6.2.1", "scipy==1.17.1", "numpy==2.2.6", "tqdm==4.67.3", "soundfile"],
        ],
        hash_overwrite="fdb_eval_venv_v1",
    )
    tk.register_output("venv/fdb_eval", venv.out_env_path)
    return venv.out_python_path


def fdb_files_for_tasks(ds_path: Path, tasks: Sequence[str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    for t in tasks:
        pattern = ds_path / f"{t}/*/input.wav"
        files += [(t, Path(p)) for p in sorted(glob(str(pattern)))]
    return files


def get_fdb_asr_download():
    repo = DownloadHuggingFaceRepoJob(model_id="kyutai/moshiko-pytorch-bf16")
    repo.out_hub_cache_dir = HF_CACHE_DIR
    return repo.out_hub_cache_dir


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


FDB_TASK_MAP = {
    "candor_pause_handling": "pause_handling",
    "candor_turn_taking": "smooth_turn_taking",
    "icc_backchannel": "backchannel",
    "synthetic_pause_handling": "pause_handling",
    "synthetic_user_interruption": "user_interruption",
}


class FullDuplexBenchEval_Inference(Job):
    # Optional LoRA adapter; excluded from the hash when absent so existing
    # base-model FDB runs keep their hash and are not re-run.
    __sis_hash_exclude__ = {"lora_weights": None, "lora_config": None}

    def __init__(
        self,
        *,
        fdb_task: str,
        model,
        lora_weights: tk.Path | None = None,
        lora_config: tk.Path | None = None,
        asr_venv_python: tk.AbstractPath | None = None,
        server_venv_python: tk.AbstractPath | None = None,
    ):
        self.fdb_data = tk.Path(
            "/home/tt201262/setups/2026-01-speech-llm/projects/Full-Duplex-Bench/v1_v1.5/dataset/v1.0",
            hash_overwrite="FullDuplexBench-datasets",
        )  # TODO... the dataset is available as a google drive link, so no good way to write a download job?

        self.fdb_task = fdb_task
        self.model = model
        # None -> base moshiko; set -> serve a LoRA finetune (passed to moshi_server).
        self.lora_weights = lora_weights
        self.lora_config = lora_config
        # Interpreter from a dedicated CreateVenv with NeMo >=2.2 for ASR scoring.
        self.asr_venv_python = asr_venv_python
        # Interpreter for serving Moshi; should match the finetune's training env
        # (moshi_venv). None -> the worker's own setup .venv (sys.executable).
        self.server_venv_python = server_venv_python

        self.out_audios = self.output_path("audios", directory=True)

        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 4,
        }

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        # check if dataset is valid
        assert os.path.exists(os.path.join(self.fdb_data, "candor_pause_handling/1/pause.json")), (
            f"Dataset not found at {self.fdb_data}"
        )

        files = fdb_files_for_tasks(Path(self.fdb_data.get_path()), [self.fdb_task])
        assert len(files) > 0, f"No files found for task {self.fdb_task} in dataset {self.fdb_data.get_path()}"

        with self.model(
            lora_weight=self.lora_weights.get_path() if self.lora_weights is not None else None,
            config_path=self.lora_config.get_path() if self.lora_config is not None else None,
            python_exe=self.server_venv_python.get() if self.server_venv_python is not None else None,
        ) as url:
            url = _ws_url(url)
            print(f"Running inference with Moshi server at {url}...")

            # Run inference on all files
            for task, inp in files:
                ind = inp.parent.name  # e.g. "1" in "v1.0/candor_pause_handling/1/input.wav"

                out = Path(self.out_audios.get_path()) / str(ind) / "output.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                print("[RUN]", task, inp)
                for _ in range(3):  # retry a few times if it fails
                    try:
                        MoshiFileClient(url, inp, out).run()
                        break
                    except Exception as e:
                        print(f"Error processing {inp}: {e}")
                        time.sleep(1)

                # Find all other .json files in that folder, and copy them
                for json_file in inp.parent.glob("*.json"):
                    out_json = out.parent / json_file.name
                    shutil.copy(json_file, out_json)

        # Score the generated audio with the benchmark's NeMo ASR (asr.py writes an
        # output.json next to each output.wav). This runs in a dedicated CreateVenv
        # (NeMo >=2.2 for transcribe(timestamps=True)), since the manager .venv pins
        # NeMo 2.0.0. We invoke asr.py's CLI as a subprocess with the venv interpreter.
        assert self.asr_venv_python is not None, "asr_venv_python is required for scoring"

        import moshified_fdb_v1_v15

        asr_script = os.path.join(os.path.dirname(moshified_fdb_v1_v15.__file__), "get_transcript", "asr.py")
        # asr.py only special-cases "user_interruption" (crops the lead-in); every
        # other task is scored as "default".
        asr_task = (
            "user_interruption" if FDB_TASK_MAP.get(self.fdb_task, self.fdb_task) == "user_interruption" else "default"
        )
        cmd = [
            self.asr_venv_python.get(),
            asr_script,
            "--root_dir",
            self.out_audios.get_path(),
            "--task",
            asr_task,
        ]
        print("[asr]", " ".join(cmd), flush=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(cmd, env=env, check=True)


class FullDuplexBenchEval_Evaluation(Job):
    def __init__(
        self,
        *,
        fdb_task: str,
        in_audios: tk.Path,
        hf_model: str = "openai/gpt-oss-120b",
        eval_venv_python: tk.AbstractPath | None = None,
    ):
        self.fdb_task = fdb_task

        self.in_audios = in_audios
        # Venv python used to run the fork's evaluate.py as a subprocess for the non-LLM
        # tasks (hash-excluded: only affects how scoring runs, not the produced result).
        self.eval_venv_python = eval_venv_python
        self.out_log = self.output_path("evaluation_output.txt")
        self.out_eval = self.output_path("eval.json")
        self.needs_llm_inference = FDB_TASK_MAP.get(self.fdb_task, self.fdb_task) in [
            "user_interruption",
            "behavior",
            "general_before_after",
        ]
        if self.needs_llm_inference:
            self.rqmt = {
                "gpu": 1,
                "cpu": 2,
                "mem": 16,
                "time": 2,
            }
            self.hf_model = hf_model
        else:
            self.rqmt = None

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        # Hash the eval venv only when it is actually used (non-LLM tasks); excluding it
        # when None keeps the expensive LLM (user_interruption) eval job hashes stable.
        if d.get("eval_venv_python") is None:
            d.pop("eval_venv_python", None)
        d["__version"] = 5
        return super().hash(d)

    def tasks(self):
        if self.rqmt is None:
            yield Task("run", mini_task=True)
        else:
            yield Task("run", rqmt=self.rqmt)

    def run(self):
        import moshified_fdb_v1_v15

        mapped_task = FDB_TASK_MAP.get(self.fdb_task, self.fdb_task)

        if not self.needs_llm_inference:
            # Non-LLM scoring: run the fork's evaluate.py as a subprocess in a venv with a
            # working audio stack (fdb_asr_venv). Avoids importing the fork into the manager
            # .venv, whose torchaudio now hard-requires torchcodec (absent here and on the
            # login node). The fork loads wavs via soundfile, so no torchcodec is needed.
            assert self.eval_venv_python is not None, "eval_venv_python required for non-LLM scoring"
            evaluate_py = os.path.join(os.path.dirname(moshified_fdb_v1_v15.__file__), "evaluation", "evaluate.py")
            cmd = [
                self.eval_venv_python.get(),
                evaluate_py,
                "--task",
                mapped_task,
                "--root_dir",
                self.in_audios.get_path(),
                "--out-json",
                self.out_eval.get_path(),
            ]
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            with open(self.out_log, "w", encoding="utf-8") as f:
                subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
            return

        # LLM-judged task (user_interruption): needs the vLLM server, so run in-process in
        # the manager .venv. eval_user_interruption does not load audio via torchaudio, so
        # torchcodec is not involved on this path.
        import moshified_fdb_v1_v15.evaluation.evaluate as moshified_fdb_v1_v15_evaluate
        from moshified_fdb_v1_v15.evaluation.evaluate import main as evaluate_main

        sys.argv = [
            "evaluate.py",
            "--task",
            mapped_task,
            "--root_dir",
            self.in_audios.get_path(),
        ]
        with open(self.out_log, "w", encoding="utf-8") as f:
            tee = Tee(sys.stdout, f)
            sys.stdout = tee
            sys.path.append(str(Path(moshified_fdb_v1_v15_evaluate.__file__).parent))
            with vllm_server(self.hf_model) as llm_url:
                sys.argv += ["--llm-api-url", llm_url, "--llm-model", self.hf_model]
                os.environ["OPENAI_API_KEY"] = "fake_key_for_vllm"
                res = evaluate_main()
            with open(self.out_eval, "w", encoding="utf-8") as f_eval:
                json.dump(res, f_eval, indent=4)
            sys.stdout = sys.__stdout__


def moshified_fdb_eval(
    fdb_task: str, model, lora_weights=None, lora_config=None, asr_venv_python=None, server_venv_python=None
):
    infer = FullDuplexBenchEval_Inference(
        fdb_task=fdb_task,
        model=model,
        lora_weights=lora_weights,
        lora_config=lora_config,
        asr_venv_python=asr_venv_python,
        server_venv_python=server_venv_python,
    )
    needs_llm = FDB_TASK_MAP.get(fdb_task, fdb_task) in [
        "user_interruption",
        "behavior",
        "general_before_after",
    ]
    _eval = FullDuplexBenchEval_Evaluation(
        fdb_task=fdb_task,
        in_audios=infer.out_audios,
        eval_venv_python=None if needs_llm else fdb_eval_venv(),
    )
    return _eval.out_eval


FDB_TASKS = [
    "icc_backchannel",
    "candor_turn_taking",
    "candor_pause_handling",
    "synthetic_pause_handling",
    "synthetic_user_interruption",
]


def fdb_benchmark_py(
    tag: str = "moshi_base",
    moshi_checkpoint: tk.Path | None = None,
    checkpoint_step: int | None = None,
    server_venv_python: tk.AbstractPath | None = None,
):
    """Full Duplex Bench eval over all tasks for one model, namespaced by ``tag``.

    ``moshi_checkpoint`` is a ``MoshiFinetune.out_rundir`` to eval a fine-tuned (LoRA)
    model via the moshi server; ``None`` evals the base ``kyutai/moshiko``. Registered
    under ``fdb/<tag>/<task>/...`` so base and finetuned runs coexist.
    """
    lora_weights = lora_config = None
    if moshi_checkpoint is not None:
        # Local import avoids a module-load cycle (knowledge_benchmark imports fdb-side helpers).
        from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import (
            ResolveLoraCheckpoint,
        )

        ckpt = ResolveLoraCheckpoint(run_dir=moshi_checkpoint, step=checkpoint_step)
        lora_weights, lora_config = ckpt.out_lora, ckpt.out_config

    asr_venv_python = fdb_asr_venv()
    for t in FDB_TASKS:
        bench = moshified_fdb_eval(
            fdb_task=t,
            model=moshi_server,
            lora_weights=lora_weights,
            lora_config=lora_config,
            asr_venv_python=asr_venv_python,
            server_venv_python=server_venv_python,
        )
        tk.register_output(f"fdb/{tag}/{t}/eval", bench)
        tk.register_output(f"fdb/{tag}/{t}/audio", bench.creator.in_audios)
        bench.creator.add_alias(f"fdb/{tag}/{t}")
