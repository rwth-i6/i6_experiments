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
import random
import subprocess
import sys
from pathlib import Path

from .common import HF_CACHE_DIR
from .moshi_client import moshi_server, _ws_url, MoshiFileClient
from .speech_backends import MOSHI_BACKEND


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

        command = [
            self.venv_python_path.get(),
            str(script_path),
            "--in_hf",
            str(self.in_hf.get()),
            "--speaker_dir",
            str(self.speaker_dir.get()),
            "--speaker_name",
            self.speaker_name,
            "--out_dir",
            str(self.out_dir.get()),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        print(f"Running Chatterbox benchmark inference: {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, check=True)


# ---------------------------------------------------------------------------
# Speech LLM inference (Moshi)
# ---------------------------------------------------------------------------


class ResolveLoraCheckpoint(Job):
    """Resolve a moshi-finetune ``run_dir`` to a single checkpoint's config + LoRA weights.

    ``MoshiFinetune`` writes adapters to ``run_dir/checkpoints/checkpoint_<step>/consolidated/``
    (``config.json`` + ``lora.safetensors``). This job picks one checkpoint (latest by default,
    or a specific ``step``) and exposes its two files as stable outputs so the inference job can
    depend on them directly.
    """

    def __init__(self, *, run_dir: tk.Path, step: int | None = None):
        self.run_dir = run_dir
        self.step = step  # None -> latest checkpoint
        self.out_config = self.output_path("config.json")
        self.out_lora = self.output_path("lora.safetensors")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        ckpt_root = Path(self.run_dir.get()) / "checkpoints"
        ckpts = sorted(ckpt_root.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[-1]))
        assert ckpts, f"No checkpoints found in {ckpt_root}"
        if self.step is not None:
            chosen = ckpt_root / f"checkpoint_{self.step:06d}"
            assert chosen.exists(), f"{chosen} not found; have {[c.name for c in ckpts]}"
        else:
            chosen = ckpts[-1]
        consolidated = chosen / "consolidated"
        print(f"Resolved LoRA checkpoint: {chosen.name}", flush=True)
        os.symlink(consolidated / "config.json", self.out_config.get())
        os.symlink(consolidated / "lora.safetensors", self.out_lora.get())


class MergeMoshiOutputsViaSymlinks(Job):
    """Merge sharded MoshiInference outputs into a single dir via symlinks."""

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


class MoshiInference(Job):
    """Run Moshi speech LLM inference on audio questions.

    Default (fast) backend runs offline batched inference; an optional LoRA adapter
    (``lora_weights`` + ``lora_config``) is applied on top of the base model so the same
    job can benchmark either the base or a fine-tuned Moshi.
    """

    __sis_hash_exclude__ = {
        # Pluggable speech-LLM backend (see speech_backends.py): server ctx-mgr,
        # streaming file client, handle->url adapter, and the cascaded-backend LLM.
        # Defaults are Moshi and excluded at those defaults, so existing Moshi inference
        # jobs keep their exact hash; a non-Moshi backend's callables differ and so
        # already yield a distinct hash.
        "server": moshi_server,
        "file_client": MoshiFileClient,
        "ws_url": _ws_url,
        "unmute_llm": None,
        # Offline driver defaults (Moshi); excluded so Moshi jobs keep their hash. A
        # non-Moshi offline backend sets a different script -> distinct hash.
        "offline_script": "moshi_offline_inference.py",
        "offline_extra_args": (),
    }

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_dir: tk.Path,
        shard: int | None = None,
        num_shards: int | None = None,
        lead_in_s: float = 2.0,
        capture_s: float = 24.0,
        backend: str = "offline",
        batch_size: int = 32,
        lora_weights: tk.Path | None = None,
        lora_config: tk.Path | None = None,
        server=moshi_server,
        file_client=MoshiFileClient,
        ws_url=_ws_url,
        unmute_llm: str | None = None,
        offline_script: str = "moshi_offline_inference.py",
        offline_extra_args: tuple = (),
    ):
        self.venv_python_path = venv_python_path
        self.in_dir = in_dir
        self.shard = shard
        self.num_shards = num_shards
        # Feed `lead_in_s` of silence (Moshi greets), the question, then `capture_s` of trailing
        # silence, and capture Moshi's ENTIRE reply -- greeting included, nothing skipped/trimmed.
        self.lead_in_s = lead_in_s
        self.capture_s = capture_s
        # backend "offline" = fast batched, as-fast-as-GPU via moshi_offline_inference.py;
        # "server" = legacy realtime websocket (moshi.server + MoshiFileClient, batch_size 1, ~1x realtime).
        self.backend = backend
        self.batch_size = batch_size
        # Pluggable speech-LLM backend (Moshi by default). ``backend`` above selects the
        # inference *mode* ("offline" = fast batched, Moshi-only; "server" = streaming,
        # used by any cascaded/server backend incl. Unmute); these select *which* model.
        self.server = server
        self.file_client = file_client
        self.ws_url = ws_url
        self.unmute_llm = unmute_llm
        # Offline driver script (under dorian_koch/) + extra CLI args; selects WHICH
        # offline model runs (Moshi by default; PersonaPlex sets its own).
        self.offline_script = offline_script
        self.offline_extra_args = tuple(offline_extra_args)
        # Optional fine-tuned LoRA adapter. None -> base moshiko.
        self.lora_weights = lora_weights
        self.lora_config = lora_config
        self.out_dir = self.output_path("moshi_output", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 16, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def sharded(*, num_shards: int, **kwargs) -> tk.Path:
        if num_shards == 1:
            return MoshiInference(**kwargs).out_dir
        assert num_shards > 1
        shards = [MoshiInference(shard=i, num_shards=num_shards, **kwargs) for i in range(num_shards)]
        return MergeMoshiOutputsViaSymlinks(in_dirs=[s.out_dir for s in shards]).out_merged

    def run(self):
        if self.backend == "offline":
            return self._run_offline()
        return self._run_server()

    def _run_offline(self):
        script = Path(__file__).resolve().parent.parent / self.offline_script
        command = [
            self.venv_python_path.get(),
            str(script),
            "--in_dir",
            str(self.in_dir.get()),
            "--out_dir",
            str(self.out_dir.get()),
            "--lead_in_s",
            str(self.lead_in_s),
            "--capture_s",
            str(self.capture_s),
            "--batch_size",
            str(self.batch_size),
        ]
        if self.lora_weights is not None:
            command += ["--lora_weights", str(self.lora_weights.get())]
            if self.lora_config is not None:
                command += ["--lora_config", str(self.lora_config.get())]
        if self.shard is not None and self.num_shards is not None:
            command += ["--shard", str(self.shard), "--num_shards", str(self.num_shards)]
        command += list(self.offline_extra_args)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        print(f"Running offline inference: {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, check=True)

    def _run_server(self):
        # Streaming path: bring up the selected backend's server and stream each question
        # wav through its file client. Backend-agnostic (Moshi server, Unmute stack, ...);
        # the server ctx-mgr takes a uniform kwargs signature and ignores what it doesn't use.
        in_dir = str(self.in_dir.get())
        out_dir = str(self.out_dir.get())
        os.makedirs(out_dir, exist_ok=True)

        wav_files = sorted(Path(in_dir).glob("*.wav"))
        if self.shard is not None and self.num_shards is not None:
            wav_files = wav_files[self.shard :: self.num_shards]
        with self.server(
            lora_weights=self.lora_weights.get_path() if self.lora_weights is not None else None,
            lora_config=self.lora_config.get_path() if self.lora_config is not None else None,
            python_exe=self.venv_python_path.get(),
            unmute_llm=self.unmute_llm,
        ) as handle:
            url = self.ws_url(handle)
            print(f"[{getattr(self.server, '__name__', 'server')}] processing {len(wav_files)} clips", flush=True)
            for i, wav in enumerate(wav_files):
                out = Path(out_dir) / wav.name
                for attempt in range(3):
                    try:
                        self.file_client(
                            url,
                            wav,
                            out,
                            lead_in_s=self.lead_in_s,
                            capture_s=self.capture_s,
                        ).run()
                        break
                    except Exception as e:
                        print(f"[RETRY {attempt + 1}/3] {wav.name}: {e}")
                if (i + 1) % 50 == 0:
                    print(f"[inference] {i + 1}/{len(wav_files)} clips done", flush=True)


# ---------------------------------------------------------------------------
# ASR -- Whisper transcription
# ---------------------------------------------------------------------------


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

        command = [
            self.venv_python_path.get(),
            str(script_path),
            "--in_dir",
            str(self.in_dir.get()),
            "--reference_data",
            str(self.reference_data.get()),
            "--out_json",
            str(self.out_json.get()),
            "--model_size",
            self.model_size,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        print(f"Running Whisper benchmark transcription: {' '.join(command)}", flush=True)
        subprocess.run(command, env=env, check=True)


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
    tag: str = "moshi_base",
    speech_backend=MOSHI_BACKEND,
    unmute_llm: str | None = None,
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

    # Optional LoRA adapter for a fine-tuned Moshi.
    lora_weights = lora_config = None
    if moshi_checkpoint is not None:
        ckpt = ResolveLoraCheckpoint(run_dir=moshi_checkpoint, step=checkpoint_step)
        lora_weights, lora_config = ckpt.out_lora, ckpt.out_config

    # 5. Moshi inference (sharded across GPUs for throughput)
    inference_venv = speech_backend.inference_venv() if speech_backend.inference_venv else moshi_venv()
    moshi_out = MoshiInference.sharded(
        num_shards=2,
        venv_python_path=inference_venv,
        in_dir=tts.out_dir,
        lora_weights=lora_weights,
        lora_config=lora_config,
        backend="offline" if speech_backend.offline_script else "server",
        offline_script=speech_backend.offline_script or "moshi_offline_inference.py",
        offline_extra_args=speech_backend.offline_extra_args,
        server=speech_backend.server,
        file_client=speech_backend.file_client,
        ws_url=speech_backend.ws_url,
        unmute_llm=unmute_llm,
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
