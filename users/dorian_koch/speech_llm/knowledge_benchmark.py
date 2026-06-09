"""
Knowledge benchmark for speech LLMs.

Evaluates a speech LLM's ability to answer factual questions.
Pipeline: HF dataset -> LLM preprocessing -> TTS -> speech LLM -> ASR -> LLM grading.

Extensible via DATASET_REGISTRY -- add new datasets by registering a loader.

Usage (from synthetic_train_data.py):

    from .knowledge_benchmark import knowledge_benchmark_py
    knowledge_benchmark_py()
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


# ---------------------------------------------------------------------------
# Dataset loading job
# ---------------------------------------------------------------------------


class LoadRawDataset(Job):
    """Load and normalize a benchmark dataset into a standard format.

    Output HF dataset has keys: question, answer, aliases, category
    """

    def __init__(self, *, dataset_name: str, split: str = "validation"):
        self.dataset_name = dataset_name
        self.split = split
        self.out_hf = self.output_path("out_hf", directory=True)
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        loader = DATASET_REGISTRY[self.dataset_name]
        ds = loader(split=self.split)
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

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split, trust_remote_code=True)

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

    Starts a Moshi server, sends each question audio, captures the response.
    Uses the same WebSocket protocol as FullDuplexBench.
    """

    def __init__(
        self,
        *,
        venv_python_path: tk.AbstractPath,
        in_dir: tk.Path,
        shard: int | None = None,
        num_shards: int | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.in_dir = in_dir
        self.shard = shard
        self.num_shards = num_shards
        self.out_dir = self.output_path("moshi_output", directory=True)
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 16, "time": 4}

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
        from .moshi_client import moshi_server, _ws_url, MoshiFileClient

        in_dir = str(self.in_dir.get())
        out_dir = str(self.out_dir.get())
        os.makedirs(out_dir, exist_ok=True)

        wav_files = sorted(Path(in_dir).glob("*.wav"))
        if self.shard is not None and self.num_shards is not None:
            wav_files = wav_files[self.shard :: self.num_shards]
        with moshi_server() as server_addr:
            ws_url = _ws_url(server_addr)
            for wav in wav_files:
                out = Path(out_dir) / wav.name
                for attempt in range(3):
                    try:
                        MoshiFileClient(ws_url, wav, out).run()
                        break
                    except Exception as e:
                        print(f"[RETRY {attempt + 1}/3] {wav.name}: {e}")


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
        self.rqmt = {"gpu": 1, "cpu": 2, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .common import vllm_server
        from openai import OpenAI

        results = []
        with open(str(self.in_json.get())) as f:
            for line in f:
                results.append(json.loads(line))

        with vllm_server(self.llm_name) as llm_url:
            client = OpenAI(api_key="EMPTY", base_url=llm_url)
            eval_results = []
            for r in results:
                aliases = r["aliases"] if isinstance(r["aliases"], list) else json.loads(r["aliases"])
                prompt = GRADING_PROMPT_TEMPLATE.format(
                    question=r["question"],
                    answer=r["answer"],
                    aliases=", ".join(aliases),
                    transcription=r["transcription"],
                )
                resp = client.chat.completions.create(
                    model=self.llm_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                try:
                    grade = json.loads(resp.choices[0].message.content)
                except (json.JSONDecodeError, KeyError):
                    grade = {"binary": 0, "quality": 1, "reasoning": "LLM response parse error"}

                eval_results.append(
                    {
                        "question": r["question"],
                        "reference": r["answer"],
                        "transcription": r["transcription"],
                        "category": r.get("category", "unknown"),
                        "binary_correct": grade.get("binary", 0),
                        "quality_score": grade.get("quality", 1),
                        "reasoning": grade.get("reasoning", ""),
                    }
                )

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
        summary["overall"] = {
            "n": len(eval_results),
            "accuracy": sum(i["binary_correct"] for i in eval_results) / len(eval_results),
            "avg_quality": sum(i["quality_score"] for i in eval_results) / len(eval_results),
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
):
    """Build the knowledge benchmark pipeline.

    Imports venvs and speakers from synthetic_train_data to reuse existing jobs.
    """
    from speech_llm.full_duplex.sis_recipe.doriank.synthetic_train_data import (
        chatterbox_venv,
        moshi_venv,
        whisper_venv,
        make_speakers,
    )

    # 1. Load dataset
    raw = LoadRawDataset(dataset_name=dataset_name, split=split)
    tk.register_output("benchmark/raw_dataset", raw.out_hf)

    # 2. LLM preprocessing
    preprocess = LLMPreprocess(in_hf=raw.out_hf, llm_name=llm_name)
    tk.register_output("benchmark/preprocessed", preprocess.out_hf)

    # 3. Speaker voice (reuse existing speaker pool)
    speakers = make_speakers()

    # 4. TTS synthesis
    tts = ChatterboxSingleSpeakerInference(
        venv_python_path=chatterbox_venv(),
        in_hf=preprocess.out_hf,
        speaker_dir=speakers.out_dir,
    )
    tk.register_output("benchmark/tts_output", tts.out_dir)

    # 5. Moshi inference (sharded across GPUs for throughput)
    moshi_out = MoshiInference.sharded(
        num_shards=4,
        venv_python_path=moshi_venv(),
        in_dir=tts.out_dir,
    )
    tk.register_output("benchmark/moshi_output", moshi_out)

    # 6. Whisper transcription
    transcription = WhisperTranscription(
        venv_python_path=whisper_venv(),
        in_dir=moshi_out,
        reference_data=preprocess.out_hf,
        model_size=whisper_model,
    )
    tk.register_output("benchmark/transcription", transcription.out_json)

    # 7. LLM grading
    grading = LLMGrading(in_json=transcription.out_json, llm_name=llm_name)
    tk.register_output("benchmark/eval_results", grading.out_eval)
    tk.register_output("benchmark/summary", grading.out_summary)
