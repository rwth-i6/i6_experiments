from glob import glob
import json
from pathlib import Path
from typing import List, Sequence, Tuple
from sisyphus import Job, Task, tk
import os
import sys
import subprocess
from .common import vllm_server
from .speech_inference import SpeechInference
from .inference_harness import FDB_TASK_MAP
from i6_experiments.users.dorian_koch.jobs.venv import CreateVenv
from functools import lru_cache

from .moshi_client import MoshiFileClient, _ws_url, moshi_server
from .speech_backends import MOSHI_BACKEND

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
    fdb_task: str,
    backend=MOSHI_BACKEND,
    lora_weights=None,
    lora_config=None,
    asr_venv_python=None,
    server_venv_python=None,
    unmute_llm=None,
    code_version=1,
):
    # Cloud realtime backends (Gemini/OpenAI) run as a login-node mini_task.
    infer = SpeechInference(
        mode="fdb",
        fdb_task=fdb_task,
        code_version=code_version,
        server=backend.server,
        file_client=backend.file_client,
        ws_url=backend.ws_url,
        # FDB uses the offline driver only for server-less (end-to-end) backends; Moshi &
        # Unmute (both have a server) stay on the realtime streaming path. SpeechInference
        # infers offline-vs-server from offline_script/module presence, so leave them None
        # for served backends.
        offline_script=backend.offline_script if backend.server is None else None,
        offline_module=backend.offline_module if backend.server is None else None,
        offline_extra_args=backend.offline_extra_args if backend.server is None else (),
        # RAG retrieval (MoshiRAG) only applies to the offline/server-less path.
        retrieval_llm=backend.retrieval_llm if backend.server is None else None,
        lora_weights=lora_weights,
        lora_config=lora_config,
        asr_venv_python=asr_venv_python,
        venv_python_path=server_venv_python,
        unmute_llm=unmute_llm,
        cloud_api=backend.cloud_api,
    )
    # rqmt is not hashed: apply the backend resource override by mutating the built job
    # (server-less/offline backends only; served backends keep the default rqmt).
    if backend.server is None and backend.rqmt_override is not None:
        infer.rqmt = {**infer.rqmt, **backend.rqmt_override}
    needs_llm = FDB_TASK_MAP.get(fdb_task, fdb_task) in [
        "user_interruption",
        "behavior",
        "general_before_after",
    ]
    _eval = FullDuplexBenchEval_Evaluation(
        fdb_task=fdb_task,
        in_audios=infer.out_dir,
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
    pplex_checkpoint: tk.Path | None = None,
    pplex_step: int | None = None,
    server_venv_python: tk.AbstractPath | None = None,
    backend=MOSHI_BACKEND,
    unmute_llm: str | None = None,
    code_version: int = 1,
):
    """Full Duplex Bench eval over all tasks for one model, namespaced by ``tag``.

    ``moshi_checkpoint`` is a ``MoshiFinetune.out_rundir`` to eval a fine-tuned (LoRA)
    model via the moshi server; ``None`` evals the base ``kyutai/moshiko``. Registered
    under ``fdb/<tag>/<task>/...`` so base and finetuned runs coexist.
    """
    # Local import avoids a module-load cycle (knowledge_benchmark imports fdb-side helpers).
    from i6_experiments.users.dorian_koch.speech_llm.knowledge_benchmark import (
        resolve_lora,
        resolve_personaplex_weights,
    )

    # PersonaPlex finetune = a partial state_dict overlay (trained_heads.safetensors, no config);
    # Moshi finetune = a LoRA adapter. Both ride the same per-run lora_weights/lora_config seam.
    if pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    else:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step)

    asr_venv_python = fdb_asr_venv()
    for t in FDB_TASKS:
        bench = moshified_fdb_eval(
            fdb_task=t,
            code_version=code_version,
            backend=backend,
            lora_weights=lora_weights,
            lora_config=lora_config,
            asr_venv_python=asr_venv_python,
            server_venv_python=server_venv_python,
            unmute_llm=unmute_llm,
        )
        tk.register_output(f"fdb/{tag}/{t}/eval", bench)
        tk.register_output(f"fdb/{tag}/{t}/audio", bench.creator.in_audios)
        bench.creator.add_alias(f"fdb/{tag}/{t}")
