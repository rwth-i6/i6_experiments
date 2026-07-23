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


def resolve_audex_weights(run_dir, step=None):
    """Map an AudexDuplexFinetune run_dir (+ optional step) to the trained Stage-0 overlay handle
    (a single partial state_dict ``stage0.safetensors``, no config -- the trained Mimi audio emb +
    depformer over the frozen Audex trunk). Backed by ResolveOverlayCheckpoint (overlay_kind=
    "audex_stage0")."""
    return ResolveOverlayCheckpoint(run_dir=run_dir, overlay_kind="audex_stage0", step=step).out_weights


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
                # .wav = reply audio; .txt = inner-monologue dump (run_pairs); .json = RAG ref trace.
                # Merge ALL of them so the monologue/reference paths see them (was .wav-only -> the
                # monologue <i>.txt got dropped -> empty responses -> MCQ parse-fail 100%).
                if not name.endswith((".wav", ".txt", ".json")):
                    continue
                link = os.path.join(out, name)
                if not os.path.exists(link):
                    os.symlink(os.path.join(src_dir, name), link)


def sharded_knowledge_inference(*, num_shards: int, rqmt_override: dict | None = None, **kwargs) -> tk.Path:
    """Build the knowledge-mode SpeechInference, sharded across GPUs for throughput.

    num_shards==1 -> a single job's out_dir; >1 -> one SpeechInference per shard
    (wavs[shard::num_shards]) merged back via MergeMoshiOutputsViaSymlinks.

    rqmt_override is applied by mutating each job's .rqmt AFTER construction (rqmt is not
    part of the Sisyphus hash), so tuning a backend's walltime/GPUs never re-hashes it."""

    def _make(**kw):
        job = SpeechInference(mode="knowledge", **kw)
        if rqmt_override is not None:
            job.rqmt = {**job.rqmt, **rqmt_override}
        return job

    if num_shards == 1:
        return _make(**kwargs).out_dir
    assert num_shards > 1
    shards = [_make(shard=i, num_shards=num_shards, **kwargs) for i in range(num_shards)]
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


class ReferenceStringTranscription(Job):
    """Emit a transcriptions.jsonl whose ``transcription`` is the MoshiRAG RETRIEVED reference string
    (read from each clip's ``<i>.json`` trace written by the rag engine), NOT the ASR of the spoken
    reply. Feeds the existing ``LLMGrading`` so we can score the *reference* directly -- the oracle
    ceiling of retrieval vs. how well the model verbalizes it. Mirrors ``WhisperTranscription``'s I/O
    contract + clip indexing exactly (clip i = ``reference_data`` row i = ``<i>.json``), so it drops
    into the same grading path. Light CPU mini_task (no GPU / no ASR)."""

    def __init__(self, *, in_dir: tk.Path, reference_data: tk.Path):
        self.in_dir = in_dir
        self.reference_data = reference_data
        self.out_json = self.output_path("transcriptions.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import os
        from datasets import load_from_disk

        ref_ds = load_from_disk(self.reference_data.get())
        in_dir = self.in_dir.get()
        n = 0
        empty = 0
        with open(self.out_json.get(), "w") as f:
            for i, example in enumerate(ref_ds):
                trace_path = os.path.join(in_dir, f"{i}.json")
                if not os.path.exists(trace_path):
                    continue  # no reply for this clip (same filter semantics as the wav check)
                with open(trace_path, encoding="utf-8") as tf:
                    trace = json.load(tf)
                reference = (trace.get("reference_text") or "").strip()
                if not reference:
                    empty += 1
                f.write(
                    json.dumps(
                        {
                            "question": example["question"],
                            "answer": example["answer"],
                            "aliases": example["aliases"],
                            "category": example.get("category", "unknown"),
                            "transcription": reference,
                        }
                    )
                    + "\n"
                )
                n += 1
        print(f"[reference-strings] wrote {n} rows ({empty} empty references) to {self.out_json.get()}", flush=True)


# ---------------------------------------------------------------------------
# LLM grading
# ---------------------------------------------------------------------------


class MonologueTranscription(Job):
    """Emit transcriptions.jsonl whose `transcription` is the model's INNER-MONOLOGUE TEXT stream -- the
    `<i>.txt` dumped next to each reply wav by moshi_engine.run_pairs -- NOT ASR-of-speech. Matches
    VoiceBench's paper protocol ("assess the quality of text responses instead of ... speech transcription")
    and the official src/models/moshi.py. Feeds LLMGrading unchanged. clip i = reference_data row i =
    `<i>.txt`. Light CPU mini_task (no GPU / no ASR)."""

    def __init__(self, *, in_dir: tk.Path, reference_data: tk.Path):
        self.in_dir = in_dir
        self.reference_data = reference_data
        self.out_json = self.output_path("transcriptions.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import os
        from datasets import load_from_disk

        ref_ds = load_from_disk(self.reference_data.get())
        in_dir = self.in_dir.get()
        n = empty = 0
        with open(self.out_json.get(), "w") as f:
            for i, example in enumerate(ref_ds):
                txt = os.path.join(in_dir, f"{i}.txt")
                if not os.path.exists(txt):
                    continue
                mono = open(txt, encoding="utf-8").read().strip()
                if not mono:
                    empty += 1
                f.write(
                    json.dumps(
                        {
                            "question": example["question"],
                            "answer": example["answer"],
                            "aliases": example["aliases"],
                            "category": example.get("category", "unknown"),
                            "transcription": mono,
                        }
                    )
                    + "\n"
                )
                n += 1
        print(f"[monologue] wrote {n} rows ({empty} empty) to {self.out_json.get()}", flush=True)


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
        # Grading prompts are <1k tokens, so we serve the judge with a small max_model_len (8192, see
        # run()) -> its KV cache fits c25g's 80 GiB H100 at TP=1, so we keep the default (fast) c25g GPU
        # routing instead of pinning the judge to the scarce c23g big-GPU queue.
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

        with vllm_server(self.llm_name, max_model_len=8192) as llm_url:
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
    audex_checkpoint: tk.Path | None = None,
    audex_step: int | None = None,
    tag: str = "moshi_base",
    speech_backend=MOSHI_BACKEND,
    unmute_llm: str | None = None,
    code_version: int = 1,
    grade_references: bool = False,
    batch_size: int | None = None,
    audex_base_speech: bool = False,
    audex_base_s2s: bool = False,
    monologue: bool = False,
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

    # --- base Audex-2B native SPEECH-in reference (speech-in question audio -> Audex audio-QA -> text) ---
    # Answers the SAME tts_output question wavs through Audex's own NV-Whisper->audio-model->text path,
    # then grades the text answer directly (no Whisper -- it already answers in text). The base-model
    # speech-knowledge ceiling to compare the AudexDuplex graft's ASR-of-reply score against.
    if audex_base_speech:
        from .audex_speech_qa import AudexSpeechQA

        qa = AudexSpeechQA(in_dir=tts.out_dir, reference_data=data)
        tk.register_output(f"benchmark/{tag}/transcription", qa.out_json)
        grading = LLMGrading(in_json=qa.out_json, llm_name=llm_name)
        tk.register_output(f"benchmark/{tag}/eval_results", grading.out_eval)
        tk.register_output(f"benchmark/{tag}/summary", grading.out_summary)
        return

    # --- base Audex-2B native SPEECH-to-SPEECH cascade (true apples-to-apples: pays the speech round-trip
    # + ASR loss like the FD models). Question audio -> Audex audioqa->audiogen->decoder -> reply wav ->
    # existing Whisper -> grade.
    if audex_base_s2s:
        from .audex_speech_s2s import AudexSpeechS2S

        s2s = AudexSpeechS2S(in_dir=tts.out_dir)
        tk.register_output(f"benchmark/{tag}/s2s_wavs", s2s.out_dir)
        transcription = WhisperTranscription(
            venv_python_path=whisper_venv(),
            in_dir=s2s.out_dir,
            reference_data=data,
            model_size=whisper_model,
        )
        tk.register_output(f"benchmark/{tag}/transcription", transcription.out_json)
        grading = LLMGrading(in_json=transcription.out_json, llm_name=llm_name)
        tk.register_output(f"benchmark/{tag}/eval_results", grading.out_eval)
        tk.register_output(f"benchmark/{tag}/summary", grading.out_summary)
        return

    # --- model-dependent stages (namespaced by tag) -------------------------

    # Optional trained-weights overlay for a fine-tuned model (shared resolvers). PersonaPlex
    # uses a partial state_dict (trained_heads.safetensors, no config); Moshi uses a LoRA adapter.
    if audex_checkpoint is not None:
        lora_weights, lora_config = resolve_audex_weights(audex_checkpoint, audex_step), None
    elif pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    else:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step)

    # 5. Moshi inference (sharded across GPUs for throughput; a cloud realtime backend
    # runs as a single login-node mini_task, so it is not sharded).
    inference_venv = speech_backend.inference_venv() if speech_backend.inference_venv else moshi_venv()
    # Single shard for a cloud API (one login-node mini_task) or a RAG backend (one shared
    # vLLM retrieval server); otherwise shard across GPUs for throughput.
    single = speech_backend.cloud_api or speech_backend.retrieval_llm is not None or speech_backend.needs_oracle_dataset
    # Single-shard backends (cloud realtime API; MoshiRAG's serial retrieval pump) have no batched
    # path -- force B=1 there regardless of the caller / the SpeechInference default, so a batched
    # default can't leak into the RAG eval (the worker asserts batch_size==1; see
    # moshirag/offline_inference.py, and the graph-build guard in speech_inference.py).
    if single:
        batch_size = 1
    _bs_kw = {} if batch_size is None else {"batch_size": batch_size}
    _oracle_kw = {"oracle_dataset": data} if speech_backend.needs_oracle_dataset else {}
    moshi_out = sharded_knowledge_inference(
        num_shards=1 if single else 2,
        venv_python_path=inference_venv,
        **_bs_kw,
        **_oracle_kw,
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

    # Optional: ALSO score the INNER-MONOLOGUE TEXT (VoiceBench-paper protocol) alongside ASR-of-reply,
    # under a *_monologue tag, so both are reported (moshi-family only -- needs the run_pairs text dump).
    if monologue:
        mono_t = MonologueTranscription(in_dir=moshi_out, reference_data=data)
        tk.register_output(f"benchmark/{tag}_monologue/transcription", mono_t.out_json)
        mono_g = LLMGrading(in_json=mono_t.out_json, llm_name=llm_name)
        tk.register_output(f"benchmark/{tag}_monologue/eval_results", mono_g.out_eval)
        tk.register_output(f"benchmark/{tag}_monologue/summary", mono_g.out_summary)

    # Optional: also score the RETRIEVED REFERENCE strings directly (the oracle ceiling of retrieval
    # vs. how well the model verbalizes them). Only meaningful for a RAG backend -- its engine writes a
    # per-clip `<i>.json` trace carrying `reference_text`; other backends have no such trace.
    if grade_references:
        ref_transcription = ReferenceStringTranscription(in_dir=moshi_out, reference_data=data)
        tk.register_output(f"benchmark/{tag}_references/transcription", ref_transcription.out_json)
        ref_grading = LLMGrading(in_json=ref_transcription.out_json, llm_name=llm_name)
        tk.register_output(f"benchmark/{tag}_references/eval_results", ref_grading.out_eval)
        tk.register_output(f"benchmark/{tag}_references/summary", ref_grading.out_summary)
