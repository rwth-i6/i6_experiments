"""
Knowledge benchmark for speech LLMs.

Evaluates a speech LLM's ability to answer factual questions.
Pipeline: HF dataset -> LLM preprocessing -> TTS -> speech LLM -> ASR -> LLM grading.

Extensible via DATASET_REGISTRY -- add new datasets by registering a loader.

Usage (from synthetic_train_data.py):

    from .knowledge_benchmark import knowledge_benchmark_py
    knowledge_benchmark_py()                                   # base moshiko
    knowledge_benchmark_py(moshi_checkpoint=ft.out_rundir, tag="moshi_ft")  # fine-tuned
"""

from sisyphus import Job, Task, tk
from datasets import load_from_disk
import json
import os
from pathlib import Path

from .common import run_worker_script
from .inference_harness import BackendInferenceMixin
from .moshi_client import moshi_server, _ws_url, MoshiFileClient
from .speech_backends import MOSHI_BACKEND
from .speech_inference import SpeechInference, ResolveOverlayCheckpoint


# ---------------------------------------------------------------------------
# Dataset loading job
# ---------------------------------------------------------------------------


class LoadRawDataset(Job):
    """Load and normalize a benchmark dataset into a standard format.

    Output HF dataset has keys: question, answer, aliases, category
    """

    def __init__(self, *, dataset_name: str, split: str = "validation", max_examples: int | None = None):
        self.dataset_name = dataset_name
        self.split = split
        self.max_examples = max_examples
        self.out_hf = self.output_path("out_hf", directory=True)
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        loader = DATASET_REGISTRY[self.dataset_name]
        ds = loader(split=self.split)
        if self.max_examples is not None:
            # reproducible random subsample
            ds = ds.shuffle(seed=42).select(range(min(self.max_examples, len(ds))))
        ds.save_to_disk(str(self.out_hf.get()))


class SubsampleDataset(Job):
    """Reproducible random subsample of an HF dataset.

    Kept as a separate downstream step so the expensive LLM preprocessing can run once on the
    full dataset (and stay cached); changing the sample size/seed only re-runs this cheap job.
    """

    def __init__(self, *, in_hf: tk.Path, n: int, seed: int = 42):
        self.in_hf = in_hf
        self.n = n
        self.seed = seed
        self.out_hf = self.output_path("out_hf", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        ds = load_from_disk(str(self.in_hf.get()))
        ds = ds.shuffle(seed=self.seed).select(range(min(self.n, len(ds))))
        ds.save_to_disk(str(self.out_hf.get()))


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, callable] = {}


def register_dataset(name: str):
    """Decorator to register a dataset loader.

    Loader signature: (split: str) -> datasets.Dataset
    Must yield examples with keys: question, answer, aliases, category
    """

    def wrapper(fn):
        DATASET_REGISTRY[name] = fn
        return fn

    return wrapper


@register_dataset("triviaqa")
def load_triviaqa(split: str = "validation"):
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split=split, trust_remote_code=True)

    def normalize(example):
        return {
            "question": example["question"],
            "answer": example["answer"]["value"],
            "aliases": example["answer"]["aliases"],
            "category": example.get("category", "unknown"),
        }

    return ds.map(normalize, remove_columns=ds.column_names)


# ---------------------------------------------------------------------------
# LLM preprocessing -- makes questions speech-digestible
# ---------------------------------------------------------------------------

LLM_PREPROCESS_INSTRUCTIONS = (
    "You are preparing questions for a speech-based AI assistant. "
    "Clean up the following question for spoken output: remove formatting artifacts, "
    "special characters, LaTeX, URLs, or anything not suitable for speech. "
    "Keep the meaning intact. Make it sound like a natural spoken question. "
    "If the question references a table, image, or code that isn't provided, "
    "rewrite it to be self-contained. "
    "Return JSON with keys: question (cleaned), answer (unchanged), aliases (unchanged list). "
    "ONLY output the JSON, nothing else."
)


class LLMPreprocess(Job):
    """Clean raw dataset questions for speech synthesis via LLM."""

    def __init__(self, *, in_hf: tk.Path, llm_name: str = "google/gemma-4-31B-it"):
        self.in_hf = in_hf
        self.llm_name = llm_name
        self.out_hf = self.output_path("out_hf", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 32, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .common import vllm_server
        from openai import OpenAI

        ds = load_from_disk(str(self.in_hf.get()))

        with vllm_server(self.llm_name) as llm_url:
            _client = OpenAI(api_key="EMPTY", base_url=llm_url)
            _client.models.list()

            def preprocess_fn(example):
                client = OpenAI(api_key="EMPTY", base_url=llm_url)
                aliases = example["aliases"] if isinstance(example["aliases"], list) else json.loads(example["aliases"])
                prompt = (
                    f"Question: {example['question']}\n"
                    f"Answer: {example['answer']}\n"
                    f"Aliases: {', '.join(aliases)}\n\n"
                    f"{LLM_PREPROCESS_INSTRUCTIONS}"
                )
                resp = client.chat.completions.create(
                    model=self.llm_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    data = json.loads(resp.choices[0].message.content)
                    return {
                        "question": data.get("question", example["question"]),
                        "answer": data.get("answer", example["answer"]),
                        "aliases": json.dumps(data.get("aliases", aliases)),
                        "category": example.get("category", "unknown"),
                    }
                except (json.JSONDecodeError, KeyError):
                    return {
                        "question": example["question"],
                        "answer": example["answer"],
                        "aliases": json.dumps(aliases),
                        "category": example.get("category", "unknown"),
                    }

            ds = ds.map(preprocess_fn, num_proc=64)
            ds.save_to_disk(str(self.out_hf.get()))


# ---------------------------------------------------------------------------
# TTS -- single-speaker synthesis for benchmark questions
# ---------------------------------------------------------------------------


class ChatterboxSingleSpeakerInference(Job):
    """Synthesize a single speaker voice from question text using Chatterbox TTS."""

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_hf: tk.Path,
        speaker_dir: tk.Path,
        speaker_name: str = "user_voices/rng_a",
    ):
        self.venv_python_path = venv_python_path
        self.in_hf = in_hf
        self.speaker_dir = speaker_dir
        self.speaker_name = speaker_name
        self.out_dir = self.output_path("tts_output", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 16, "time": 24}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        script_path = Path(__file__).resolve().parent / "chatterbox_benchmark_inference.py"
        args = [
            "--in_hf",
            self.in_hf.get(),
            "--speaker_dir",
            self.speaker_dir.get(),
            "--speaker_name",
            self.speaker_name,
            "--out_dir",
            self.out_dir.get(),
        ]
        run_worker_script(
            self.venv_python_path.get(),
            script_path,
            args,
            log_label="Chatterbox benchmark inference",
            with_hf_home=False,
        )


# ---------------------------------------------------------------------------
# Speech LLM inference (Moshi)
# ---------------------------------------------------------------------------


def resolve_lora(moshi_checkpoint, checkpoint_step=None):
    """Map a MoshiFinetune run_dir (+ optional step) to (lora_weights, lora_config) output
    handles, or (None, None) for the base model. Shared by both benchmark builders. Backed by
    the unified ResolveOverlayCheckpoint (overlay_kind="lora")."""
    if moshi_checkpoint is None:
        return None, None
    ckpt = ResolveOverlayCheckpoint(run_dir=moshi_checkpoint, overlay_kind="lora", step=checkpoint_step)
    return ckpt.out_weights, ckpt.out_config


def resolve_personaplex_weights(run_dir, step=None):
    """Map a PersonaPlex SpeechFinetune run_dir (+ optional step) to a single trained-weights
    output handle for the offline driver's --trained_weights overlay (no separate config). Backed
    by the unified ResolveOverlayCheckpoint (overlay_kind="personaplex_heads")."""
    return ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="personaplex_heads", step=step).out_weights


class MergeMoshiOutputsViaSymlinks(Job):
    """Merge sharded SpeechInference (knowledge-mode) outputs into a single dir via symlinks."""

    def __init__(self, *, in_dirs: list[tk.Path]):
        self.in_dirs = in_dirs
        self.out_merged = self.output_path("moshi_output", directory=True)

    def tasks(self):
        yield Task("merge", mini_task=True)

    def merge(self):
        out = self.out_merged.get()
        os.makedirs(out, exist_ok=True)
        for in_dir in self.in_dirs:
            src_dir = in_dir.get()
            for name in os.listdir(src_dir):
                if not name.endswith(".wav"):
                    continue
                link = os.path.join(out, name)
                if not os.path.exists(link):
                    os.symlink(os.path.join(src_dir, name), link)


def sharded_knowledge_inference(*, num_shards: int, **kwargs) -> tk.Path:
    """Build the knowledge-mode SpeechInference, sharded across GPUs for throughput.

    num_shards==1 -> a single job's out_dir; >1 -> one SpeechInference per shard
    (wavs[shard::num_shards]) merged back via MergeMoshiOutputsViaSymlinks."""
    if num_shards == 1:
        return SpeechInference(mode="knowledge", **kwargs).out_dir
    assert num_shards > 1
    shards = [SpeechInference(mode="knowledge", shard=i, num_shards=num_shards, **kwargs) for i in range(num_shards)]
    return MergeMoshiOutputsViaSymlinks(in_dirs=[s.out_dir for s in shards]).out_merged


class WhisperTranscription(Job):
    """Transcribe Moshi response audio using Whisper."""

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_dir: tk.Path,
        reference_data: tk.Path,
        model_size: str = "medium",
    ):
        self.venv_python_path = venv_python_path
        self.in_dir = in_dir
        self.reference_data = reference_data
        self.model_size = model_size
        self.out_json = self.output_path("transcriptions.jsonl")
        self.rqmt = {"gpu": 1, "cpu": 2, "mem": 8, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        script_path = Path(__file__).resolve().parent / "whisper_benchmark_inference.py"
        args = [
            "--in_dir",
            self.in_dir.get(),
            "--reference_data",
            self.reference_data.get(),
            "--out_json",
            self.out_json.get(),
            "--model_size",
            self.model_size,
        ]
        run_worker_script(
            self.venv_python_path.get(),
            script_path,
            args,
            log_label="Whisper benchmark transcription",
            with_hf_home=False,
        )


# ---------------------------------------------------------------------------
# LLM grading
# ---------------------------------------------------------------------------

GRADING_PROMPT_TEMPLATE = """You are evaluating a speech AI assistant's response to a factual question.

Question: {question}
Reference answer: {answer}
Also acceptable: {aliases}

The assistant's response (transcribed via ASR): {transcription}

Grade the response:
1. binary: Is the response correct? (1 = correct, 0 = incorrect)
2. quality: Rate overall quality 1-5 (1=wrong/gibberish, 2=partially relevant, 3=approximately correct, 4=correct but verbose, 5=concise and correct)

Return ONLY JSON: {{"binary": 0 or 1, "quality": 1-5, "reasoning": "brief explanation"}}"""


class LLMGrading(Job):
    """Grade ASR transcriptions using an LLM judge."""

    def __init__(self, *, in_json: tk.Path, llm_name: str = "google/gemma-4-31B-it"):
        self.in_json = in_json
        self.llm_name = llm_name
        self.out_eval = self.output_path("eval_results.jsonl")
        self.out_summary = self.output_path("summary.json")
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from datasets import Dataset

        from .common import vllm_server
        from openai import OpenAI

        results = []
        with open(str(self.in_json.get())) as f:
            for line in f:
                results.append(json.loads(line))

        with vllm_server(self.llm_name) as llm_url:
            _client = OpenAI(api_key="EMPTY", base_url=llm_url)
            _client.models.list()  # block until the server is ready to serve

            def grade_fn(r):
                client = OpenAI(api_key="EMPTY", base_url=llm_url)
                aliases = r["aliases"] if isinstance(r["aliases"], list) else json.loads(r["aliases"])
                prompt = GRADING_PROMPT_TEMPLATE.format(
                    question=r["question"],
                    answer=r["answer"],
                    aliases=", ".join(aliases),
                    transcription=r["transcription"],
                )
                try:
                    resp = client.chat.completions.create(
                        model=self.llm_name,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                    )
                    grade = json.loads(resp.choices[0].message.content)
                except Exception:
                    grade = {"binary": 0, "quality": 1, "reasoning": "LLM request/parse error"}
                return {
                    "binary_correct": grade.get("binary", 0),
                    "quality_score": grade.get("quality", 1),
                    "reasoning": grade.get("reasoning", ""),
                }

            # Concurrency via datasets.map(num_proc=...), matching the other vLLM jobs
            # (e.g. LLMPreprocess): each worker opens its own client and the vLLM server
            # multiplexes the requests, instead of the old one-request-at-a-time loop.
            graded = Dataset.from_list(results).map(grade_fn, num_proc=64)

        eval_results = [
            {
                "question": r["question"],
                "reference": r["answer"],
                "transcription": r["transcription"],
                "category": r.get("category", "unknown"),
                "binary_correct": r["binary_correct"],
                "quality_score": r["quality_score"],
                "reasoning": r["reasoning"],
            }
            for r in graded
        ]

        with open(str(self.out_eval.get()), "w") as f:
            for er in eval_results:
                f.write(json.dumps(er) + "\n")

        # Aggregate by category
        cats: dict[str, list] = {}
        for er in eval_results:
            cats.setdefault(er["category"], []).append(er)
        summary = {}
        for cat, items in cats.items():
            summary[cat] = {
                "n": len(items),
                "accuracy": sum(i["binary_correct"] for i in items) / len(items),
                "avg_quality": sum(i["quality_score"] for i in items) / len(items),
            }
        n = len(eval_results)
        summary["overall"] = {
            "n": n,
            "accuracy": sum(i["binary_correct"] for i in eval_results) / n if n else 0.0,
            "avg_quality": sum(i["quality_score"] for i in eval_results) / n if n else 0.0,
        }
        with open(str(self.out_summary.get()), "w") as f:
            json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def knowledge_benchmark_py(
    dataset_name: str = "triviaqa",
    split: str = "validation",
    llm_name: str = "google/gemma-4-31B-it",
    whisper_model: str = "large-v3-turbo",
    max_examples: int | None = None,
    moshi_checkpoint: tk.Path | None = None,
    checkpoint_step: int | None = None,
    pplex_checkpoint: tk.Path | None = None,
    pplex_step: int | None = None,
    tag: str = "moshi_base",
    speech_backend=MOSHI_BACKEND,
    unmute_llm: str | None = None,
    code_version: int = 1,
):
    """Build the knowledge benchmark pipeline.

    Imports venvs and speakers from synthetic_train_data to reuse existing jobs.

    The data stages (load -> preprocess -> subsample -> TTS) are model-independent and shared
    across runs (registered under ``benchmark/...``); the model-dependent stages
    (Moshi -> Whisper -> grading) are namespaced under ``benchmark/<tag>/...`` so a base and a
    fine-tuned run coexist without clobbering each other.

    Args:
        moshi_checkpoint: a ``MoshiFinetune.out_rundir`` to benchmark a fine-tuned (LoRA) model.
            ``None`` benchmarks the base ``kyutai/moshiko``.
        checkpoint_step: which fine-tune checkpoint step to use (``None`` = latest).
        tag: output namespace for the model-dependent stages (e.g. ``moshi_base`` / ``moshi_ft``).
    """
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
        chatterbox_venv,
        moshi_venv,
        whisper_venv,
        make_speakers,
    )

    # 1. Load the full dataset
    raw = LoadRawDataset(dataset_name=dataset_name, split=split)
    tk.register_output("benchmark/raw_dataset", raw.out_hf)

    # 2. LLM preprocessing on the FULL dataset (cached; reused across subsamples)
    preprocess = LLMPreprocess(in_hf=raw.out_hf, llm_name=llm_name)
    tk.register_output("benchmark/preprocessed", preprocess.out_hf)

    # 2b. Reproducible random subsample AFTER preprocessing, so changing the sample size/seed
    # only re-runs this cheap step rather than the whole preprocess.
    data = preprocess.out_hf
    if max_examples is not None:
        data = SubsampleDataset(in_hf=preprocess.out_hf, n=max_examples).out_hf
        tk.register_output("benchmark/sampled", data)

    # 3. Speaker voice (reuse existing speaker pool)
    speakers = make_speakers()

    # 4. TTS synthesis
    tts = ChatterboxSingleSpeakerInference(
        venv_python_path=chatterbox_venv(),
        in_hf=data,
        speaker_dir=speakers.out_dir,
    )
    tk.register_output("benchmark/tts_output", tts.out_dir)

    # --- model-dependent stages (namespaced by tag) -------------------------

    # Optional trained-weights overlay for a fine-tuned model (shared resolvers). PersonaPlex
    # uses a partial state_dict (trained_heads.safetensors, no config); Moshi uses a LoRA adapter.
    if pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    else:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step)

    # 5. Moshi inference (sharded across GPUs for throughput; a cloud realtime backend
    # runs as a single login-node mini_task, so it is not sharded).
    inference_venv = speech_backend.inference_venv() if speech_backend.inference_venv else moshi_venv()
    # Single shard for a cloud API (one login-node mini_task) or a RAG backend (one shared
    # vLLM retrieval server); otherwise shard across GPUs for throughput.
    single = speech_backend.cloud_api or speech_backend.retrieval_llm is not None
    moshi_out = sharded_knowledge_inference(
        num_shards=1 if single else 2,
        venv_python_path=inference_venv,
        in_dir=tts.out_dir,
        lora_weights=lora_weights,
        lora_config=lora_config,
        # SpeechInference infers offline-vs-server from offline_script/module presence (no
        # explicit `backend`): an offline backend keeps Moshi's default script (or rides its
        # offline_module); a pure server backend passes neither so it streams.
        offline_script=(speech_backend.offline_script or "moshi_offline_inference.py")
        if (speech_backend.offline_script or speech_backend.offline_module)
        else None,
        offline_module=speech_backend.offline_module,
        offline_extra_args=speech_backend.offline_extra_args,
        server=speech_backend.server,
        file_client=speech_backend.file_client,
        ws_url=speech_backend.ws_url,
        unmute_llm=unmute_llm,
        cloud_api=speech_backend.cloud_api,
        retrieval_llm=speech_backend.retrieval_llm,
        rqmt_override=speech_backend.rqmt_override,
        code_version=code_version,
    )
    tk.register_output(f"benchmark/{tag}/moshi_output", moshi_out)

    # 6. Whisper transcription
    transcription = WhisperTranscription(
        venv_python_path=whisper_venv(),
        in_dir=moshi_out,
        reference_data=data,
        model_size=whisper_model,
    )
    tk.register_output(f"benchmark/{tag}/transcription", transcription.out_json)

    # 7. LLM grading
    grading = LLMGrading(in_json=transcription.out_json, llm_name=llm_name)
    tk.register_output(f"benchmark/{tag}/eval_results", grading.out_eval)
    tk.register_output(f"benchmark/{tag}/summary", grading.out_summary)
