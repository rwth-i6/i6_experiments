"""VoiceBench integration — faithful spoken-QA / knowledge eval for all our speech LLMs.

VoiceBench (https://github.com/MatthewCYM/VoiceBench) is a standard, leaderboard-comparable spoken
instruction/QA benchmark. We use its **own repo + datasets + scorers VERBATIM** — the authoritative,
trust-critical parts — and add only the inference glue that is inherently ours (our speech-out models):

  theirs (unchanged):  hlt-lab/voicebench datasets + src/evaluator/* + api_judge.py  (pinned commit)
  ours (thin):         materialize prompt audio -> our SpeechInference reply -> ASR-of-reply -> their JSONL

Nothing from their repo is reimplemented or copied into our tree; `VoiceBenchScore` imports/executes their
code. The rule-based subsets (mcq -> mmsu/openbookqa, ifeval, harm -> advbench, bbh) are torch-free and
deterministic -> 100% faithful. Only the open-ended subsets (open/qa) use the LLM judge, which we point at
our local `gpt-oss-120b` served under the alias `gpt-4o-mini` (so their `api_judge.py` stays byte-verbatim);
documented deviation, switchable to real OpenAI gpt-4o-mini via env.

Response convention: ASR-of-reply (reuse our SpeechInference + faster-whisper), uniform across ALL our
backends (moshiko / our FT / RL / PersonaPlex / MoshiRAG / AudexDuplex / HF baselines). VoiceBench's own
src/models/moshi.py instead reads Moshi's inner-monologue text; ASR-of-reply matches our existing
knowledge-benchmark methodology and works identically for every backend (minor, documented choice).

Usage (from synthetic_train_data.py):
    from .voicebench import voicebench_all_py
    voicebench_all_py(models={"moshi_base": MOSHI_BACKEND, "moshi_rl_ckpt1400": ...}, subsets=("openbookqa", "mmsu"))
"""

import subprocess
from pathlib import Path

from sisyphus import Job, Task, tk

from .common import run_worker_script
from .knowledge_benchmark import (
    resolve_lora,
    resolve_personaplex_weights,
    sharded_knowledge_inference,
)
from .speech_backends import MOSHI_BACKEND

# Pinned upstream commit — reproducibility contract. Bumping this re-hashes the whole chain (intended).
VOICEBENCH_REPO = "https://github.com/MatthewCYM/VoiceBench.git"
VOICEBENCH_COMMIT = "d9c570e3f5c637dbc2001716f1543395330b26a1"

# subset -> (their evaluator name, needs the LLM judge, hf split). Mirrors their README's evaluator table.
# Rule-based (needs_judge=False) subsets are fully faithful; only open/qa call the judge.
SUBSET_SPEC = {
    "openbookqa": ("mcq", False, "test"),
    "mmsu": ("mcq", False, "test"),
    "ifeval": ("ifeval", False, "test"),
    "bbh": ("bbh", False, "test"),
    "advbench": ("harm", False, "test"),
    "commoneval": ("open", True, "test"),
    "wildvoice": ("open", True, "test"),
    "alpacaeval": ("open", True, "test"),
    "alpacaeval_full": ("open", True, "test"),
    "sd-qa": ("qa", True, "usa"),  # sd-qa uses region-code splits; 'usa' is the English default
}

# Knowledge-focused default subset set (all rule-based -> zero judge dependency, fully reproducible).
KNOWLEDGE_SUBSETS = ("openbookqa", "mmsu", "bbh", "ifeval")


class VoiceBenchClone(Job):
    """Clone the upstream VoiceBench repo pinned at a fixed commit (durable, reproducible)."""

    def __init__(self, *, repo_url: str = VOICEBENCH_REPO, commit: str = VOICEBENCH_COMMIT):
        self.repo_url = repo_url
        self.commit = commit
        self.out_repo = self.output_path("VoiceBench", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dst = self.out_repo.get()
        subprocess.run(["git", "clone", self.repo_url, dst], check=True)
        subprocess.run(["git", "-C", dst, "checkout", self.commit], check=True)


class VoiceBenchData(Job):
    """Materialize one VoiceBench subset: `{i}.wav` prompt audio (16 kHz, dataset order) + the non-audio
    columns as an HF dataset (`out_ref`, index-aligned) carrying every field the scorers need
    (`prompt`, `reference`, ifeval kwargs, ...). Loaded exactly as their main.py (`hlt-lab/voicebench`)."""

    def __init__(self, *, venv_python_path: tk.AbstractPath, subset: str, split: str = "test"):
        self.venv_python_path = venv_python_path
        self.subset = subset
        self.split = split
        self.out_dir = self.output_path("prompts", directory=True)
        self.out_ref = self.output_path("ref_ds", directory=True)
        self.rqmt = {"cpu": 2, "mem": 16, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        script_path = Path(__file__).resolve().parent / "voicebench_data.py"
        run_worker_script(
            self.venv_python_path.get(),
            script_path,
            [
                "--subset",
                self.subset,
                "--split",
                self.split,
                "--out_dir",
                self.out_dir.get(),
                "--out_ref",
                self.out_ref.get(),
            ],
            log_label=f"VoiceBench data [{self.subset}]",
            with_hf_home=True,
        )


class VoiceBenchResponses(Job):
    """ASR-transcribe our model's reply wavs and assemble VoiceBench's response JSONL schema
    (`{**non_audio_fields, 'response': transcript}`), index-aligned to `out_ref`. Reuses the same
    faster-whisper + Silero-VAD approach as our WhisperTranscription."""

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_dir: tk.Path,
        ref_ds: tk.Path,
        model_tag: str,
        subset: str,
        whisper_model: str = "large-v3-turbo",
    ):
        self.venv_python_path = venv_python_path
        self.in_dir = in_dir
        self.ref_ds = ref_ds
        self.model_tag = model_tag
        self.subset = subset
        self.whisper_model = whisper_model
        # Their filename convention: {model}-{data}-{split}-{modality}.jsonl
        self.out_jsonl = self.output_path(f"{model_tag}-{subset}-test-audio.jsonl")
        self.rqmt = {"gpu": 1, "cpu": 2, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        script_path = Path(__file__).resolve().parent / "voicebench_responses.py"
        run_worker_script(
            self.venv_python_path.get(),
            script_path,
            [
                "--in_dir",
                self.in_dir.get(),
                "--ref_ds",
                self.ref_ds.get(),
                "--out_jsonl",
                self.out_jsonl.get(),
                "--model_size",
                self.whisper_model,
            ],
            log_label=f"VoiceBench responses [{self.model_tag}/{self.subset}]",
            with_hf_home=False,
        )


class VoiceBenchScore(Job):
    """Run VoiceBench's OWN scorers (verbatim) on our response JSONL -> per-subset score + summary.json.

    open/qa subsets: first run their `api_judge.py` (judge endpoint via env -> our served gpt-4o-mini
    alias), then their evaluator. Rule-based subsets: their evaluator directly. In all cases the scoring
    logic is theirs, imported/executed unchanged (`voicebench_score_runner.py` imports
    `src.evaluator.evaluator_mapping`)."""

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        repo: tk.Path,
        responses: tk.Path,
        subset: str,
        judge_base_url: str | None = None,
        judge_model: str = "gpt-4o-mini",
    ):
        self.venv_python_path = venv_python_path
        self.repo = repo
        self.responses = responses
        self.subset = subset
        self.evaluator, self.needs_judge, _ = SUBSET_SPEC[subset]
        self.judge_base_url = judge_base_url
        self.judge_model = judge_model
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"cpu": 4, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        script_path = Path(__file__).resolve().parent / "voicebench_score_runner.py"
        args = [
            "--repo",
            self.repo.get(),
            "--responses",
            self.responses.get(),
            "--evaluator",
            self.evaluator,
            "--out_summary",
            self.out_summary.get(),
        ]
        if self.needs_judge:
            args += ["--run_judge"]
            if self.judge_base_url:
                args += ["--judge_base_url", self.judge_base_url, "--judge_model", self.judge_model]
        run_worker_script(
            self.venv_python_path.get(),
            script_path,
            args,
            log_label=f"VoiceBench score [{self.subset}/{self.evaluator}]",
            with_hf_home=False,
        )


def _inference_out_dir(
    *, speech_backend, prompts_dir, tag, lora_weights, lora_config, ref_ds, code_version, batch_size
):
    """Backend -> reply-wav dir, mirroring knowledge_benchmark_py's SpeechInference plumbing."""
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import moshi_venv

    inference_venv = speech_backend.inference_venv() if speech_backend.inference_venv else moshi_venv()
    single = speech_backend.cloud_api or speech_backend.retrieval_llm is not None or speech_backend.needs_oracle_dataset
    if single:
        batch_size = 1
    _bs_kw = {} if batch_size is None else {"batch_size": batch_size}
    _oracle_kw = {"oracle_dataset": ref_ds} if speech_backend.needs_oracle_dataset else {}
    return sharded_knowledge_inference(
        num_shards=1 if single else 2,
        venv_python_path=inference_venv,
        **_bs_kw,
        **_oracle_kw,
        in_dir=prompts_dir,
        lora_weights=lora_weights,
        lora_config=lora_config,
        offline_script=(speech_backend.offline_script or "moshi_offline_inference.py")
        if (speech_backend.offline_script or speech_backend.offline_module)
        else None,
        offline_module=speech_backend.offline_module,
        offline_extra_args=speech_backend.offline_extra_args,
        server=speech_backend.server,
        file_client=speech_backend.file_client,
        ws_url=speech_backend.ws_url,
        cloud_api=speech_backend.cloud_api,
        retrieval_llm=speech_backend.retrieval_llm,
        rqmt_override=speech_backend.rqmt_override,
        code_version=code_version,
    )


class VoiceBenchMonologueResponses(Job):
    """Assemble VoiceBench's response JSONL from the model's INNER-MONOLOGUE TEXT (`<i>.txt` dumped by
    moshi_engine.run_pairs) as the `response`, instead of ASR-of-reply -- matches the VoiceBench PAPER
    protocol ("assess text responses, not speech transcription") + official src/models/moshi.py. Same
    schema as VoiceBenchResponses ({**non_audio_fields, 'response': text}), index-aligned to out_ref.
    Light CPU mini_task (no GPU / no ASR). moshi-family only (needs the run_pairs text dump)."""

    def __init__(self, *, in_dir: tk.Path, ref_ds: tk.Path, model_tag: str, subset: str):
        self.in_dir = in_dir
        self.ref_ds = ref_ds
        self.model_tag = model_tag
        self.subset = subset
        self.out_jsonl = self.output_path("responses.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import os
        from datasets import load_from_disk

        ref = load_from_disk(self.ref_ds.get())
        in_dir = self.in_dir.get()
        n = empty = 0
        with open(self.out_jsonl.get(), "w") as f:
            for i, ex in enumerate(ref):
                p = os.path.join(in_dir, f"{i}.txt")
                mono = open(p, encoding="utf-8").read().strip() if os.path.exists(p) else ""
                if not mono:
                    empty += 1
                row = {k: ex[k] for k in ex}
                row["response"] = mono
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"[vb-monologue {self.model_tag}/{self.subset}] {n} rows ({empty} empty)", flush=True)


def voicebench_benchmark_py(
    *,
    tag: str = "moshi_base",
    speech_backend=MOSHI_BACKEND,
    subsets=KNOWLEDGE_SUBSETS,
    moshi_checkpoint: tk.Path | None = None,
    checkpoint_step: int | None = None,
    pplex_checkpoint: tk.Path | None = None,
    pplex_step: int | None = None,
    whisper_model: str = "large-v3-turbo",
    judge_base_url: str | None = None,
    code_version: int = 1,
    batch_size: int | None = None,
    monologue: bool = False,
) -> dict:
    """Full VoiceBench pipeline for ONE model over `subsets`. Data stages are model-independent
    (shared under `voicebench/data/<subset>`); model stages are namespaced by `tag`. Returns
    `{subset: summary_path}`."""
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
        whisper_venv,
        voicebench_venv,
    )

    repo = VoiceBenchClone().out_repo
    tk.register_output("voicebench/repo", repo)

    if pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    else:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step)

    summaries = {}
    for subset in subsets:
        _, _, split = SUBSET_SPEC[subset]
        # 1. Data (shared across models)
        data = VoiceBenchData(venv_python_path=whisper_venv(), subset=subset, split=split)
        tk.register_output(f"voicebench/data/{subset}/prompts", data.out_dir)

        # 2. Inference (our model over the prompt wavs)
        reply_dir = _inference_out_dir(
            speech_backend=speech_backend,
            prompts_dir=data.out_dir,
            tag=tag,
            lora_weights=lora_weights,
            lora_config=lora_config,
            ref_ds=data.out_ref,
            code_version=code_version,
            batch_size=batch_size,
        )
        tk.register_output(f"voicebench/{tag}/{subset}/reply", reply_dir)

        # 3. ASR-of-reply -> their response JSONL schema
        responses = VoiceBenchResponses(
            venv_python_path=whisper_venv(),
            in_dir=reply_dir,
            ref_ds=data.out_ref,
            model_tag=tag,
            subset=subset,
            whisper_model=whisper_model,
        ).out_jsonl
        tk.register_output(f"voicebench/{tag}/{subset}/responses", responses)

        # 4. Their scorers (verbatim) -> score
        score = VoiceBenchScore(
            venv_python_path=voicebench_venv(),
            repo=repo,
            responses=responses,
            subset=subset,
            judge_base_url=judge_base_url,
        )
        tk.register_output(f"voicebench/{tag}/{subset}/summary", score.out_summary)
        summaries[subset] = score.out_summary

        # Optional: ALSO score the INNER-MONOLOGUE TEXT (VoiceBench-paper protocol) under {tag}_monologue,
        # keeping the ASR path above. Reads the <i>.txt dumps run_pairs writes next to the reply wavs.
        if monologue:
            mono_responses = VoiceBenchMonologueResponses(
                in_dir=reply_dir, ref_ds=data.out_ref, model_tag=tag, subset=subset
            ).out_jsonl
            tk.register_output(f"voicebench/{tag}_monologue/{subset}/responses", mono_responses)
            mono_score = VoiceBenchScore(
                venv_python_path=voicebench_venv(),
                repo=repo,
                responses=mono_responses,
                subset=subset,
                judge_base_url=judge_base_url,
            )
            tk.register_output(f"voicebench/{tag}_monologue/{subset}/summary", mono_score.out_summary)
    return summaries


def voicebench_all_py(*, models: dict, subsets=KNOWLEDGE_SUBSETS, **kwargs) -> dict:
    """Run VoiceBench for many models. `models` maps `tag -> speech_backend` (or a dict of
    per-model kwargs for voicebench_benchmark_py). Returns `{tag: {subset: summary_path}}`."""
    out = {}
    for tag, spec in models.items():
        if isinstance(spec, dict):
            out[tag] = voicebench_benchmark_py(tag=tag, subsets=subsets, **{**kwargs, **spec})
        else:
            out[tag] = voicebench_benchmark_py(tag=tag, speech_backend=spec, subsets=subsets, **kwargs)
    return out
