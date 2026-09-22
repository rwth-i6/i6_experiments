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
import time
from pathlib import Path

from .clip_store import is_clip_dataset, merge_clip_datasets, open_clips
from .common import (
    add_cuda_npp_to_env,
    add_venv_python_lib_to_env,
    merge_jsonl_parts,
    run_worker_script,
    run_worker_script_per_gpu,
    vllm_gpu_mem_gb,
)
from .inference_harness import BackendInferenceMixin
from .moshi_client import moshi_server, _ws_url, MoshiFileClient
from .speech_backends import MOSHI_BACKEND
from .speech_inference import SpeechInference, ResolveOverlayCheckpoint
from .tts import InstallFFmpeg


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

#: The decoding seed NEW knowledge tags should pass as ``inference_seed`` (backlog E1). A single
#: named constant rather than a literal per call site, so a seeded tag is visibly the same seed as
#: every other seeded tag and a sweep is an explicit departure from it. Deliberately NOT a default
#: on ``knowledge_benchmark_py``: defaulting it would silently re-hash and re-run all 41 existing
#: call sites, which is the one thing the rule forbids. See the ``inference_seed`` docstring.
KNOWLEDGE_INFERENCE_SEED = 1234

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
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 32, "time": 4, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .common import vllm_server
        from openai import OpenAI

        ds = load_from_disk(str(self.in_hf.get()))

        # Shrink the context so this fits c25g's 80 GB card: gemma-4-31B-it at vLLM's 65536 default
        # loads 58.9 GiB of weights and leaves only 8.2 GiB for KV, which is not enough for 65536 --
        # the engine refused to start and this job errored (2026-08-05), stalling the whole graph.
        # 12288 is measured against BOTH constraints, which is the part a first pass got wrong.
        #   Prompt: over all 12,000 rows of the real input the worst is 13,731 chars (~3.9k tokens --
        #     one row carries a very long alias list), and the completion echoes those aliases back
        #     as JSON with no max_tokens cap, so the context must hold ~2x the worst prompt.
        #   KV budget: c25g leaves 8.21 GiB after the 58.9 GiB of weights, and this model needs
        #     ~0.506 MiB/token -- so 16384 wants 8.29 GiB and vLLM refuses to start ("estimated
        #     maximum model length is 15200"). 12288 needs ~6.2 GiB and fits with ~2 GiB spare,
        #     while still leaving ~8.4k tokens for the completion after the worst prompt.
        # Sizing the prompt headroom without checking it against the KV budget is what cost a second
        # failed run here (2026-09-06); do both halves of the arithmetic.
        with vllm_server(self.llm_name, max_model_len=12288) as llm_url:
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

    __sis_hash_exclude__ = {
        # Clip storage layout: "wav" writes one <i>.wav per question (the original), "hf" writes a
        # single arrow dataset. Excluded at the "wav" default so every existing benchmark keeps its
        # hash and does NOT re-run; only a caller that opts in to "hf" gets a fresh hash. Consumers
        # read either layout (clip_store.open_clips), so the two coexist indefinitely.
        "storage": "wav",
        # Our own FFmpeg build, passed so torchcodec does not depend on the node providing one.
        # Excluded at None so adding it re-hashes nothing.
        "ffmpeg_path": None,
    }

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_hf: tk.Path,
        speaker_dir: tk.Path,
        speaker_name: str = "user_voices/rng_a",
        storage: str = "wav",
        ffmpeg_path: tk.Path | None = None,
        env_ffmpeg_path: tk.Path | None = None,
    ):
        self.ffmpeg_path = ffmpeg_path
        # Same FFmpeg build as ``ffmpeg_path``, reaching the job through a channel that ``hash()``
        # drops. Two channels exist for one thing because they are not interchangeable:
        # ``ffmpeg_path`` is already IN the settled hash of the corpus-pipeline user_audio job
        # (pipelines.py), so it cannot be un-hashed without re-running that job and every corpus
        # downstream of it; and it is None on all ~40 benchmark call sites, so it cannot be
        # populated there without re-hashing every benchmark TTS and cascading through
        # transcription -> grading -> the whole judged ledger. The hash-excluded channel is the
        # only way to give the benchmark jobs the library without moving either set of hashes --
        # the same split, and for the same reason, as SpeechFinetune's ``compute``.
        self.env_ffmpeg_path = env_ffmpeg_path
        self.venv_python_path = venv_python_path
        self.in_hf = in_hf
        self.speaker_dir = speaker_dir
        self.speaker_name = speaker_name
        assert storage in ("wav", "hf"), f"storage must be 'wav' or 'hf', got {storage!r}"
        self.storage = storage
        self.out_dir = self.output_path("tts_output", directory=True)
        # Chatterbox pulls in torchcodec, which dlopens FFmpeg (libavutil.so.56). c25g has no system
        # FFmpeg, so a run scheduled there dies at import with "libavutil.so.56: cannot open shared
        # object file" -- 9 minutes in, after the GPU is allocated. Declared as a CAPABILITY, not a
        # partition, so settings.py owns the mapping (same as ChatterboxInference). rqmt is not part
        # of the Sisyphus hash, so adding this re-runs nothing.
        #
        # mem stays 16 -- and must not be raised again. Under storage="hf" this job OOM-died three
        # times (2026-09-07/08/09), each time AFTER generating every clip, and each time the response
        # was to double the request: 16 -> 48 -> 120. That was treating the symptom. Peak scaled with
        # the CORPUS, because the worker accumulated every clip and handed the whole set to
        # Dataset.from_dict, which copies it again into arrow -- so no value of this number was ever
        # going to be enough. The escalation ended by hitting a wall rather than a fix: 128 was
        # REJECTED outright by c23g's submit filter, which caps memory at 122 GB per GPU ("Can only
        # request up to 122GB per GPU (488GB per node max)!"), and a rejected job is not resubmitted,
        # it just silently never runs.
        #
        # The worker now spools each clip to disk as it is produced and streams the spool into arrow
        # one clip at a time (chatterbox_benchmark_inference.ClipSpool), so peak is ONE clip and the
        # corpus is bounded by disk instead of by this line. If this job OOMs again, the memory
        # profile regressed -- fix that, not this number.
        #
        # ⚠ rqmt is not hashed, so changing it re-runs nothing -- but that ALSO means Sisyphus will
        # not resubmit a job already sitting in the SLURM queue to apply it, and SLURM froze the old
        # request at submit time. On 09-07 the bump landed minutes after the job queued; 18 h later it
        # started still holding ReqMem 16G and died exactly the way the bump was meant to prevent.
        # After changing rqmt on a QUEUED job, hpc-rerun.py it (or scancel and let the manager
        # resubmit) -- otherwise the change applies only to the NEXT submission.
        # No `requires: ["system_ffmpeg"]` -- same reasoning as ChatterboxInference (2026-09-16):
        # FFmpeg, CUDA NPP and libpython all travel with the job via env_hook, proven on c25g.
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 16, "time": 24}

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        # ``env_ffmpeg_path`` only puts a shared library on LD_LIBRARY_PATH; it cannot change a
        # single synthesised sample. Popped unconditionally rather than declared in
        # __sis_hash_exclude__, because that form excludes an argument ONLY while it equals the
        # listed default -- so a populated path would be hashed like any other and the leak would
        # move rather than close (exactly the guard bit that caught SpeechFinetune.compute).
        d.pop("env_ffmpeg_path", None)
        return super().hash(d)

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
            "--storage",
            self.storage,
        ]

        # torchcodec needs our FFmpeg libs AND CUDA NPP on LD_LIBRARY_PATH; supplying both is what
        # makes this job node-independent instead of relying on c23g providing them system-wide.
        def env_hook(env):
            ffmpeg_path = self.ffmpeg_path or self.env_ffmpeg_path
            # Not optional in practice. Dropping `requires: ["system_ffmpeg"]` (2026-09-16) let this
            # job route to c25g, which has no system FFmpeg -- and with BOTH channels None the line
            # below is skipped, so it arrived there carrying nothing and died on
            # "libavutil.so.60: cannot open shared object file", our OWN build's libavutil, i.e. it
            # was never on the loader path at all. The c25g proof that justified dropping the tag
            # passed ffmpeg_path explicitly; no benchmark caller did. Assert rather than warn: a
            # missing library here costs a GPU allocation and ~4 min before it fails.
            assert ffmpeg_path is not None, (
                "ChatterboxSingleSpeakerInference needs an FFmpeg build: torchcodec dlopens "
                "libavutil at import and no partition is guaranteed to provide one. Pass "
                "env_ffmpeg_path=<InstallFFmpeg>.out_path (hash-free) from the caller."
            )
            InstallFFmpeg.add_to_env(ffmpeg_path, env)
            add_cuda_npp_to_env(self.venv_python_path.get(), env)
            add_venv_python_lib_to_env(self.venv_python_path.get(), env)

        # Per-GPU fan-out (backlog G1): clips are strided by index across the workers (clip i's
        # seed is SEED + i, so each clip is bit-identical to the single-worker run). Under
        # storage="wav" the workers write disjoint <i>.wav into the same dir; under "hf" each
        # writes its own arrow part and the parts are merged (indices asserted disjoint).
        out_dir = self.out_dir.get()

        def args_for(k, n):
            a = list(args)
            if n > 1:
                a += ["--shard", k, "--num_shards", n]
                if self.storage == "hf":
                    a[a.index("--out_dir") + 1] = f"{out_dir}.gpu{k}"
            return a

        n = run_worker_script_per_gpu(
            self.venv_python_path.get(),
            script_path,
            args_for,
            log_label="Chatterbox benchmark inference",
            with_hf_home=False,
            env_hook=env_hook,
        )
        if n > 1 and self.storage == "hf":
            import shutil

            parts = [f"{out_dir}.gpu{k}" for k in range(n)]
            merge_clip_datasets(out_dir, parts)
            for p in parts:
                shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Speech LLM inference (Moshi)
# ---------------------------------------------------------------------------


def resolve_lora(moshi_checkpoint, checkpoint_step=None, overlay_kind="lora"):
    """Map a moshi-family run_dir (+ optional step) to (weights, config) output handles, or
    (None, None) for the base model. Shared by both benchmark builders. Backed by the unified
    ResolveOverlayCheckpoint.

    ``overlay_kind`` must be the ARM'S kind, not this function's name: a **full** finetune
    resolves ``model.safetensors`` and has no LoRA config, so it returns ``(weights, None)``.
    Hardcoding "lora" here is what silently broke a11_full (2026-08-21) -- it produced symlinks
    to a ``lora.safetensors`` and a ``config.json`` that a full-FT checkpoint never writes, and
    ``os.symlink`` creates a dangling link without complaining, so the resolver reported success
    and the SpeechInference downstream died on "inputs are not ready" with no log. The default
    keeps every existing LoRA arm's hash unchanged.
    """
    if moshi_checkpoint is None:
        return None, None
    assert overlay_kind in ("lora", "full"), (
        f"resolve_lora handles the moshi checkpoint layouts (lora | full); got {overlay_kind!r}. "
        f"PersonaPlex and Audex have their own resolve_* helpers."
    )
    ckpt = ResolveOverlayCheckpoint(run_dir=moshi_checkpoint, overlay_kind=overlay_kind, step=checkpoint_step)
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
        # Arrow shards concatenate into one dataset. The symlink path below spends one inode PER
        # CLIP a second time, so a sharded 1000-clip benchmark cost ~2000 inodes just to be
        # readable as a single dir -- on a shared /hpcwork volume whose binding limit is inodes.
        # Concatenation costs a handful of files no matter how many shards.
        srcs = [d.get() for d in self.in_dirs]
        if srcs and is_clip_dataset(srcs[0]):
            merge_clip_datasets(out, srcs)
            return
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
        # Per-GPU fan-out (backlog G1): clips strided by index across workers, each writing its own
        # jsonl part + .idx sidecar; merged back into clip order below.
        out_json = self.out_json.get()

        def args_for(k, n):
            a = list(args)
            if n > 1:
                a += ["--shard", k, "--num_shards", n]
                a[a.index("--out_json") + 1] = f"{out_json}.part{k}"
            return a

        n = run_worker_script_per_gpu(
            self.venv_python_path.get(),
            script_path,
            args_for,
            log_label="Whisper benchmark transcription",
            with_hf_home=False,
        )
        if n > 1:
            parts = [f"{out_json}.part{k}" for k in range(n)]
            merge_jsonl_parts(out_json, parts)
            for p in parts:
                os.remove(p)
                os.remove(p + ".idx")


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
        # Either layout: <i>.txt beside the reply wav (original) or the `monologue` column of an
        # arrow clip dataset (storage="hf"). open_clips resolves both; sidecar() returns None when
        # this clip has no monologue, which is the same "skip it" case as a missing .txt.
        clips = open_clips(in_dir)
        n = empty = 0
        with open(self.out_json.get(), "w") as f:
            for i, example in enumerate(ref_ds):
                if i not in clips:
                    continue
                mono = clips.sidecar(i, "monologue")
                if mono is None:
                    continue
                mono = mono.strip()
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


#: Bumped when a grader's scoring RULE changes in a way that makes new numbers incomparable with
#: old ones. It is recorded per summary, so an old number keeps the version it was scored under
#: rather than silently inheriting the new meaning.
GRADER_SCHEMA_VERSION = 1

#: A judge failure is scored as a WRONG ANSWER, not as a missing measurement, so it biases accuracy
#: downward -- retry before accepting one. 64 concurrent workers against a single vLLM server make
#: transient busy/timeout responses ordinary.
JUDGE_MAX_ATTEMPTS = 4
JUDGE_RETRY_BACKOFF_S = 2.0


def _grader_block(name: str, *, model: str | None = None) -> dict:
    """Identity of the grader that produced a summary, written INTO the summary.

    ``name`` is "llm_judge" or "alias_match". These are not interchangeable and must never share a
    ranking: the judge reads for meaning, the alias scorer asks only whether a gold string appears
    anywhere in the reply, and the measured gap is ~6 pts on base (0 on terse arms, which is worse --
    the bias is not even a constant offset). Accuracy is the comparable field; ``avg_quality`` is
    NOT, because the judge's 1-5 rubric distinguishes "correct but verbose" (4) from "concise and
    correct" (5), so it moves with reply LENGTH at constant accuracy -- measured both directions on
    2026-09-18 (a41 18 words shorter: q|correct 2.93 -> 3.94; personaplex_ft 30 words longer:
    3.78 -> 3.02). Read accuracy; treat quality as a style statistic.
    """
    assert name in ("llm_judge", "alias_match"), f"unknown grader {name!r}"
    block = {"name": name, "schema_version": GRADER_SCHEMA_VERSION}
    if model is not None:
        block["model"] = model
    return block


def bench_out_prefix(grader: str, n: int | None) -> str:
    """Where a benchmark tag's outputs are registered: ``benchmark/<grader>/n<N>/<tag>/...``.

    The grader and the sample size are the two axes that have actually been confused when reading
    these numbers, so they are in the PATH rather than implied by which directory a tool happened to
    glob. The old layout put the judged results at ``benchmark/<tag>`` and the alias ones at
    ``benchmark/quick/<tag>`` -- a split that carried no meaning at a glance, and that
    ``read_benchmarks.py`` half-globbed for its whole life, so one grader was invisible to the
    ledger tool. With this layout a listing of ``output/benchmark/`` states the partition itself.

    ``n`` is the REQUESTED sample size, which is known at graph build. It is safe to put in the path:
    measured across all 78 tags on 2026-09-18, transcription == eval_results == summary.overall.n
    everywhere, i.e. rows never drop between sampling and scoring. ``check_grader_provenance.py``
    asserts that equality so a future divergence fails a check instead of mislabelling a directory.
    """
    assert grader in ("llm_judge", "alias_match"), f"unknown grader {grader!r}"
    return f"benchmark/{grader}/n{n if n is not None else 'full'}"


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
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 16, "time": 2, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name)}

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

        # Both numbers are MEASURED, and they trade against each other -- pick them together.
        #
        # Prompt: over all 1,000 real graded rows the worst prompt is 8,905 chars (~2,544 tokens;
        # median 249, p95 480). The long tail is the `aliases` list, not the transcription -- the
        # old "<1k tokens" claim in vllm_server's docstring was wrong. The completion is a short
        # JSON verdict (~200 tokens). So 4096 is ~1.6x the worst case end to end.
        #
        # KV budget on c25g, derived from two real vLLM startups on this model (79.18 GiB card):
        #     KV_available(util) = util * 79.18 - 63.05 GiB      [63.05 = weights + overhead]
        #     KV_needed(len)     = 6.89 GiB * len / 8192
        # 4096 needs 3.45 GiB; at util 0.85 there is 4.25 GiB. Fits, with residue tolerance of
        # (1 - 0.85) * 79.18 = 11.9 GiB -- which covers the 10.8 GiB of someone else's memory that
        # killed this job on n25g0004 (2026-09-06).
        #
        # ⚠ The obvious-looking 8192 @ 0.85 does NOT work: 8192 needs 6.89 GiB against 4.25
        # available. That was tried and failed the same day; lowering util without re-checking
        # KV_needed is exactly the half-of-the-arithmetic mistake this comment exists to stop.
        # enforce_eager is deliberately NOT set here -- see vllm_server's docstring for the
        # measurement that withdrew it. On a WARM node compile is only ~11 s and capture ~8 s, so
        # eager would buy ~19 s of boot and pay it back in slower decoding over ~1,000 requests,
        # which is very likely a net loss. The [vllm-timing] line is what will settle it.
        with vllm_server(
            self.llm_name,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
        ) as llm_url:
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
                # RETRY, because a failure here is recorded as a WRONG ANSWER (binary 0), not as a
                # missing measurement -- so every transient blip biases the arm's accuracy downward.
                # There was no retry at all until 2026-09-18 and the exception was discarded, so the
                # cause was unknowable; 18 of 78 tags carried 1-2 such rows per 1000. The load makes
                # transients likely: `datasets.map(num_proc=64)` below points 64 workers at one vLLM
                # server, so a busy/timeout response is ordinary, and a malformed JSON body happens
                # even under response_format.
                #
                # The row is still KEPT (scored 0) rather than dropped if every attempt fails: `n` is
                # part of the registered output path now, and silently shrinking the denominator would
                # make the path lie about the sample size. The count is reported as `judge_errors` in
                # the summary instead, and the exception is recorded so the next failure is
                # diagnosable rather than anonymous.
                grade = None
                last_err = ""
                for attempt in range(JUDGE_MAX_ATTEMPTS):
                    try:
                        resp = client.chat.completions.create(
                            model=self.llm_name,
                            messages=[{"role": "user", "content": prompt}],
                            response_format={"type": "json_object"},
                        )
                        grade = json.loads(resp.choices[0].message.content)
                        break
                    except Exception as e:  # noqa: BLE001 -- any failure is retryable here
                        last_err = f"{type(e).__name__}: {e}"[:200]
                        if attempt + 1 < JUDGE_MAX_ATTEMPTS:
                            time.sleep(JUDGE_RETRY_BACKOFF_S * (2**attempt))
                if grade is None:
                    grade = {
                        "binary": 0,
                        "quality": 1,
                        "reasoning": f"LLM request/parse error after {JUDGE_MAX_ATTEMPTS} attempts -- {last_err}",
                    }
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
        # WHICH GRADER PRODUCED THIS NUMBER, in the data rather than in a naming convention.
        # AliasMatchGrading deliberately emits this identical schema so it can be a drop-in for the
        # tail of the pipeline -- which also made the two indistinguishable once the number left the
        # job, and a judged score and an alias score differ by ~6 pts on base. The grader was
        # recoverable only from the `quick_` tag prefix, and that has now caused the same misreading
        # twice. `_grader_block` is shared with AliasMatchGrading so the two can never drift.
        summary["grader"] = _grader_block("llm_judge", model=self.llm_name)
        # How often the JUDGE failed, kept rather than hidden. A request/parse error is caught above
        # and written as {"binary": 0, ...}, i.e. "the judge broke" is recorded as "the model got it
        # wrong" -- a strictly downward bias that nothing surfaced. Measured 2026-09-18: 18 of 78
        # tags carry 1-2 such rows per 1000 (<=0.2 pt, so no published number moves), but the count
        # belongs in the summary so a future run that errors on 30% of rows is visible instead of
        # looking like a collapse.
        summary["judge_errors"] = sum(
            1 for er in eval_results if "LLM request/parse error" in str(er.get("reasoning", ""))
        )
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
    #: Overlay layout of ``moshi_checkpoint``: "lora" (adapter) or "full" (whole state dict).
    #: Must match the ARM; see resolve_lora. Default keeps existing LoRA hashes.
    moshi_overlay_kind: str = "lora",
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
    storage: str = "hf",
    ffmpeg_path: tk.Path | None = None,
    inference_seed: int | None = None,
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
        storage: clip layout for the question TTS and the model replies. ``"hf"`` (**the default
            since 2026-09-10**) writes one arrow dataset per stage, cutting a 1000-example run from
            ~2000 inodes to ~3 on the shared /hpcwork volume; ``"wav"`` writes one file per example,
            as every benchmark scored before that date did. Both read identically downstream
            (``clip_store.open_clips``), so a tag can switch without changing its numbers -- but it
            DOES change the job hashes of that tag's TTS/inference stages, so switching re-runs them.
            ⚠ **That is why all 40 pre-existing call sites pass ``storage="wav"`` explicitly.** They
            are pinned to the layout their numbers were measured in; the pin is what made flipping
            this default hash-neutral (verified: 1,780 job ids byte-identical before and after).
            Do not "tidy them up" -- deleting a pin re-runs that benchmark. New tags inherit ``"hf"``
            and should simply not mention storage. Note the ``SpeechInference``/TTS **jobs** still
            default to ``"wav"`` for the same reason: constructing one directly without the kwarg
            must keep its hash.
        ffmpeg_path: our own ``InstallFFmpeg`` build, handed to the question-TTS job so torchcodec
            can load without the NODE providing FFmpeg. With this (plus nvidia-npp-cu12 in the
            venv) the job no longer needs ``requires: ["system_ffmpeg"]`` and stops being pinned to
            c23g. Hash-excluded at ``None`` on the job, so passing it re-runs only the tag that
            opts in.
        inference_seed: RNG seed for the model's decoding (backlog E1). ``None`` (default) is the
            historical unseeded behaviour. ⚠ Only the **lib** backends honour it -- the fork drivers
            have their own argparse and are deliberately not sent ``--seed`` (they would exit(2)
            after the GPU is allocated), so setting this on a fork-backend tag silently does
            nothing. Hash-excluded at ``None``, so it re-runs only the tag that opts in.

            **The standing rule, so this is not re-litigated (user, 2026-09-15: "E1 is obvious"):
            pass ``inference_seed=KNOWLEDGE_INFERENCE_SEED`` on NEW tags; never retro-seed an
            existing one.** Retro-seeding re-runs ~30 already-scored numbers and not one of them
            would reproduce, because they were produced unseeded -- so it buys nothing and costs
            every historical comparison. The precedent is E8: apply a determinism fix where it
            changes nothing, and gate the honest version on a new tag.
            **Seeded and unseeded numbers must never appear in the same table.**

            Note the seeding capability is not theoretical -- FDB has used it in production since
            ``seed_variance_py`` (3 models x 3 seeds), one of them through the offline lib driver.
            It is the knowledge line that has 43 unseeded call sites and zero seeded ones.
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

    # Registered under the grader and sample size, so `ls output/benchmark` shows the partition.
    _bench = bench_out_prefix("llm_judge", max_examples)

    # 3. Speaker voice (reuse existing speaker pool)
    speakers = make_speakers()

    # 4. TTS synthesis
    # ``env_ffmpeg_path`` is supplied unconditionally, not left to the ~40 call sites. torchcodec
    # dlopens libavutil at import, so this job cannot run anywhere without an FFmpeg build, and a
    # per-call-site opt-in is precisely what failed on 2026-09-17: every benchmark passed
    # ffmpeg_path=None, the env_hook's `if` skipped, and the job died on c25g after taking a GPU.
    # InstallFFmpeg() is argument-free, so JobSingleton returns the SAME job the corpus pipeline
    # already builds -- no extra work in the graph -- and hash() drops this channel, so supplying
    # it here moves none of the settled benchmark hashes.
    tts = ChatterboxSingleSpeakerInference(
        venv_python_path=chatterbox_venv(),
        in_hf=data,
        speaker_dir=speakers.out_dir,
        storage=storage,
        ffmpeg_path=ffmpeg_path,
        env_ffmpeg_path=InstallFFmpeg().out_path,
    )
    tk.register_output("benchmark/tts_output", tts.out_dir)

    # --- base Audex-2B native SPEECH-in reference (speech-in question audio -> Audex audio-QA -> text) ---
    # Answers the SAME tts_output question wavs through Audex's own NV-Whisper->audio-model->text path,
    # then grades the text answer directly (no Whisper -- it already answers in text). The base-model
    # speech-knowledge ceiling to compare the AudexDuplex graft's ASR-of-reply score against.
    if audex_base_speech:
        from .audex_speech_qa import AudexSpeechQA

        qa = AudexSpeechQA(in_dir=tts.out_dir, reference_data=data)
        tk.register_output(f"{_bench}/{tag}/transcription", qa.out_json)
        grading = LLMGrading(in_json=qa.out_json, llm_name=llm_name)
        tk.register_output(f"{_bench}/{tag}/eval_results", grading.out_eval)
        tk.register_output(f"{_bench}/{tag}/summary", grading.out_summary)
        return

    # --- base Audex-2B native SPEECH-to-SPEECH cascade (true apples-to-apples: pays the speech round-trip
    # + ASR loss like the FD models). Question audio -> Audex audioqa->audiogen->decoder -> reply wav ->
    # existing Whisper -> grade.
    if audex_base_s2s:
        from .audex_speech_s2s import AudexSpeechS2S

        s2s = AudexSpeechS2S(in_dir=tts.out_dir)
        tk.register_output(f"{_bench}/{tag}/s2s_wavs", s2s.out_dir)
        transcription = WhisperTranscription(
            venv_python_path=whisper_venv(),
            in_dir=s2s.out_dir,
            reference_data=data,
            model_size=whisper_model,
        )
        tk.register_output(f"{_bench}/{tag}/transcription", transcription.out_json)
        grading = LLMGrading(in_json=transcription.out_json, llm_name=llm_name)
        tk.register_output(f"{_bench}/{tag}/eval_results", grading.out_eval)
        tk.register_output(f"{_bench}/{tag}/summary", grading.out_summary)
        return

    # --- model-dependent stages (namespaced by tag) -------------------------

    # Optional trained-weights overlay for a fine-tuned model (shared resolvers). PersonaPlex
    # uses a partial state_dict (trained_heads.safetensors, no config); Moshi uses a LoRA adapter.
    if audex_checkpoint is not None:
        lora_weights, lora_config = resolve_audex_weights(audex_checkpoint, audex_step), None
    elif pplex_checkpoint is not None:
        lora_weights, lora_config = resolve_personaplex_weights(pplex_checkpoint, pplex_step), None
    else:
        lora_weights, lora_config = resolve_lora(moshi_checkpoint, checkpoint_step, moshi_overlay_kind)

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
        storage=storage,
        seed=inference_seed,
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
    tk.register_output(f"{_bench}/{tag}/moshi_output", moshi_out)

    # 6. Whisper transcription
    transcription = WhisperTranscription(
        venv_python_path=whisper_venv(),
        in_dir=moshi_out,
        reference_data=data,
        model_size=whisper_model,
    )
    tk.register_output(f"{_bench}/{tag}/transcription", transcription.out_json)

    # 7. LLM grading
    grading = LLMGrading(in_json=transcription.out_json, llm_name=llm_name)
    tk.register_output(f"{_bench}/{tag}/eval_results", grading.out_eval)
    tk.register_output(f"{_bench}/{tag}/summary", grading.out_summary)

    # Optional: ALSO score the INNER-MONOLOGUE TEXT (VoiceBench-paper protocol) alongside ASR-of-reply,
    # under a *_monologue tag, so both are reported (moshi-family only -- needs the run_pairs text dump).
    if monologue:
        mono_t = MonologueTranscription(in_dir=moshi_out, reference_data=data)
        tk.register_output(f"{_bench}/{tag}_monologue/transcription", mono_t.out_json)
        mono_g = LLMGrading(in_json=mono_t.out_json, llm_name=llm_name)
        tk.register_output(f"{_bench}/{tag}_monologue/eval_results", mono_g.out_eval)
        tk.register_output(f"{_bench}/{tag}_monologue/summary", mono_g.out_summary)

    # Optional: also score the RETRIEVED REFERENCE strings directly (the oracle ceiling of retrieval
    # vs. how well the model verbalizes them). Only meaningful for a RAG backend -- its engine writes a
    # per-clip `<i>.json` trace carrying `reference_text`; other backends have no such trace.
    if grade_references:
        ref_transcription = ReferenceStringTranscription(in_dir=moshi_out, reference_data=data)
        tk.register_output(f"{_bench}/{tag}_references/transcription", ref_transcription.out_json)
        ref_grading = LLMGrading(in_json=ref_transcription.out_json, llm_name=llm_name)
        tk.register_output(f"{_bench}/{tag}_references/eval_results", ref_grading.out_eval)
        tk.register_output(f"{_bench}/{tag}_references/summary", ref_grading.out_summary)
