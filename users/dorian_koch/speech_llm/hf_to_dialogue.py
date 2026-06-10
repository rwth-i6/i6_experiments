"""
hf_to_dialogue.py — convert HuggingFace datasets into spoken dialogue training examples.

Supports multiple source datasets (MMLU-Pro multiple-choice, TriviaQA open-ended, …)
via a DATASET_ADAPTER_REGISTRY, multiple LLMs round-robined per shard, and a pool of
dialogue-instruction prompt templates chosen deterministically per example to maximise
style diversity and avoid the "That would be …" monoculture.
"""

from pathlib import Path
from sisyphus import Job, Task, tk
import os
import hashlib
from .common import HF_CACHE_DIR, vllm_server
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    concatenate_datasets,
)
from openai import OpenAI
import json
from i6_experiments.users.dorian_koch.jobs.hf import (
    HfDownloadSplit as implHfDownloadSplit,
    HfMergeShards as implHfMergeShards,
)


class HfDownloadSplit(implHfDownloadSplit):
    pass  # Epic hack to not break hashes


class HfMergeShards(implHfMergeShards):
    pass


# ---------------------------------------------------------------------------
# Dataset adapters: each maps a raw HF row → a normalised fact spec
# {uid, question, answer, aliases: list[str], options?: list[str],
#  background?: str}
# uid is used for deterministic prompt-template selection.
# ---------------------------------------------------------------------------

DATASET_ADAPTER_REGISTRY: dict[str, callable] = {}


def register_adapter(name: str):
    def wrapper(fn):
        DATASET_ADAPTER_REGISTRY[name] = fn
        return fn

    return wrapper


@register_adapter("mmlu_pro")
def adapt_mmlu_pro(example: dict) -> dict:
    options = example.get("options", [])
    answer_index = example.get("answer_index", 0)
    cot = example.get("cot_content") or ""
    return {
        "uid": str(example.get("question_id", 0)),
        "question": example["question"],
        "answer": example.get("answer", options[answer_index] if options else ""),
        "aliases": [],
        "options": options,
        "background": cot if cot else None,
    }


@register_adapter("triviaqa")
def adapt_triviaqa(example: dict) -> dict:
    ans = example.get("answer", {})
    if isinstance(ans, dict):
        value = ans.get("value", "")
        aliases = ans.get("aliases", [])
    else:
        value = str(ans)
        aliases = []
    return {
        "uid": example.get("question_id", example["question"][:64]),
        "question": example["question"],
        "answer": value,
        "aliases": aliases,
        "options": None,
        "background": None,
    }


# ---------------------------------------------------------------------------
# Dialogue instruction templates
# We select one per example by hashing its uid, so the choice is deterministic
# and reproducible but varies across examples.  Each template explicitly forbids
# the most common filler phrases produced by a single-prompt setup.
# ---------------------------------------------------------------------------

_COMMON_SUFFIX = (
    "Important style rules:\n"
    "- Do NOT start any turn with 'That would be', 'I can help with that', "
    "'Great question', 'Certainly', 'Of course', 'Sure!', 'Absolutely', "
    "'Exactly!', or similar filler.\n"
    "- Get to the point quickly. Natural spoken exchanges are concise.\n"
    "- Use varied, idiomatic English appropriate for casual spoken conversation.\n"
    "- Because the dialogue will be spoken by a TTS system, do NOT use special "
    "characters, LaTeX, markdown, or lists.  Translate everything into natural "
    "spoken words.\n"
    "- The TTS system can render the paralinguistic tags [laugh], [chuckle], [cough].\n"
    "- The dialogue must be fully self-contained. A listener with no prior context must understand what is being discussed. Never refer to answer options by letter (A, B, C, F, G, …) — instead speak the actual content. Never say 'this question', 'the problem', or 'option X' without first stating what the question or option is.\n"
    "The FIRST speaker in the dialogue MUST be the user.\n"
    "Output ONLY a JSON array of objects, each with 'speaker' ('user' or 'assistant') "
    "and 'text'. No explanations, no markdown wrapper.\n"
)

DIALOGUE_INSTRUCTION_TEMPLATES = [
    # Template 0 — direct, assistant-led
    (
        "Create a short, natural spoken dialogue where the user asks a question and "
        "the assistant immediately gives a clear, correct answer.  The assistant "
        "speaks first only if the user just asked something.  Keep to 2–4 turns total.\n" + _COMMON_SUFFIX
    ),
    # Template 1 — user guesses, assistant corrects/confirms
    (
        "Create a spoken dialogue where the user has a guess or partial knowledge about "
        "the topic and asks the assistant to confirm or fill in the gap.  The assistant "
        "responds directly without restating the whole question.\n"
        "Keep to 3–5 turns.  The assistant should sound knowledgeable but not pedantic.\n" + _COMMON_SUFFIX
    ),
    # Template 2 — casual, one or two turns
    (
        "Write a very brief (1–2 assistant turns) casual spoken exchange.  The user asks "
        "the question in a relaxed, off-the-cuff way.  The assistant answers concisely "
        "as a knowledgeable friend would — no lecture, no preamble.\n" + _COMMON_SUFFIX
    ),
    # Template 3 — multi-turn explanation for complex topics
    (
        "The user asks a question first, then the assistant breaks the answer into "
        "2–3 short turns, letting the user react or ask a follow-up between them.  "
        "The user MUST speak first.  Total turns: 4–6.\n" + _COMMON_SUFFIX
    ),
    # Template 4 — user is sceptical, assistant convinces
    (
        "Write a dialogue where the user is slightly sceptical or surprised by the "
        "correct answer and the assistant briefly justifies it.  Keep it conversational "
        "and grounded — no over-explaining.  2–4 turns.\n" + _COMMON_SUFFIX
    ),
    # Template 5 — user-initiated Socratic
    (
        "Create a short exchange: the user asks about the topic first, then the "
        "assistant responds with a brief leading question to guide the user toward "
        "the answer, the user tries to answer, and the assistant confirms with the "
        "correct information.  The user MUST speak first.  Keep to 4 turns.  "
        "Tone: friendly and collegiate, not condescending.\n" + _COMMON_SUFFIX
    ),
]


def _pick_template(uid: str) -> str:
    """Deterministically pick a template index from uid."""
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16)
    return DIALOGUE_INSTRUCTION_TEMPLATES[h % len(DIALOGUE_INSTRUCTION_TEMPLATES)]


# ---------------------------------------------------------------------------
# User message builder: adapts the fact spec to the dialogue-gen request
# ---------------------------------------------------------------------------


def _build_user_message(spec: dict, template: str) -> str:
    parts = []
    if spec.get("background"):
        parts.append(f"Background: {spec['background']}")
    parts.append(f"Question: {spec['question']}")
    parts.append(f"Correct answer: {spec['answer']}")
    if spec.get("aliases"):
        parts.append(f"Acceptable answer variants: {', '.join(spec['aliases'][:5])}")
    if spec.get("options"):
        opts = "  ".join(f"{chr(65 + i)}: {o}" for i, o in enumerate(spec["options"]))
        parts.append(f"Multiple-choice options: {opts}")
    parts.append("")
    parts.append(template)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dialogue generator (stateless, one call per example)
# ---------------------------------------------------------------------------


def make_dialogue_gen(llm_url: str, model_name: str, adapter_name: str):
    """Return a datasets.map-compatible function that generates one dialogue."""
    adapter = DATASET_ADAPTER_REGISTRY[adapter_name]

    def make_dialogue(example, seed_offset: int = 0):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "nothing"),
            base_url=llm_url,
        )
        spec = adapter(example)
        template = _pick_template(str(spec["uid"]))
        user_msg = _build_user_message(spec, template)

        # Deterministic but varied seed: combine uid hash with a per-run offset
        # so re-generating with a different offset gives different outputs.
        uid_int = int(hashlib.md5(str(spec["uid"]).encode()).hexdigest(), 16) % (2**31)
        seed = (uid_int + seed_offset) % (2**31)

        # Vary temperature across examples to prevent identical phrasing
        temp = 0.7 + 0.4 * ((uid_int % 5) / 4.0)  # 0.7 … 1.1

        messages = [{"role": "user", "content": user_msg}]
        max_attempts = 5
        for attempt in range(max_attempts):
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                seed=(seed + attempt) % (2**31),
                temperature=temp,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw) if raw else []
            if isinstance(parsed, dict):
                for _v in parsed.values():
                    if isinstance(_v, list):
                        parsed = _v
                        break
            if isinstance(parsed, list) and parsed and parsed[0].get("speaker") == "user":
                return {"dialogue": json.dumps(parsed)}
            print(
                f"Attempt {attempt + 1} failed (uid={spec['uid']!r}): first speaker={parsed[0].get('speaker') if parsed else 'none'!r}"
            )
            print(f"  Generated: {json.dumps(parsed)[:300]}")
        raise ValueError(
            f"Dialogue does not start with user after {max_attempts} attempts (uid={spec['uid']!r}). "
            f"Last output: {json.dumps(parsed)[:200]}"
        )

    return make_dialogue


# ---------------------------------------------------------------------------
# Sisyphus jobs
# ---------------------------------------------------------------------------


class HfToDialogue(Job):
    def __init__(
        self,
        *,
        dataset_split_path: tk.Path,
        llm_name: str,
        dialogue_instructions: str | None = None,  # kept for backward compat; ignored if adapter_name set
        adapter_name: str = "mmlu_pro",
        shard: int | None = None,
        num_shards: int | None = None,
        seed_offset: int = 0,
    ):
        self.dataset_split_path = dataset_split_path
        self.llm_name = llm_name
        # dialogue_instructions kept in hash for backward compat with existing jobs
        self.dialogue_instructions = dialogue_instructions or ""
        self.adapter_name = adapter_name
        self.shard = shard
        self.num_shards = num_shards
        self.seed_offset = seed_offset

        self.out_hf = self.output_path("dialogue_dataset", directory=True)
        self.out_json = self.output_path("json_files")
        self.rqmt = {
            "gpu": 1,
            "cpu": 2,
            "mem": 16,
            "time": 2,
        }

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        # Backward compat: old jobs passed dialogue_instructions and had no
        # adapter_name/seed_offset.  Exclude the new keys and keep __version=2
        # so the existing MMLU-Pro jobs are not re-run unnecessarily.
        if d.get("dialogue_instructions", ""):
            d.pop("adapter_name", None)
            d.pop("seed_offset", None)
            d["__version"] = 2
        else:
            d["__version"] = 5
        return super().hash(d)

    @staticmethod
    def sharded(*, num_shards: int, llm_names: list | None = None, **kwargs):
        """Create sharded dialogue-gen jobs, optionally rotating across LLMs.

        llm_names: if provided, shard i uses llm_names[i % len(llm_names)].
        This distributes load and increases stylistic diversity.  If None,
        all shards use the single llm_name in kwargs.
        """
        if num_shards == 1:
            if llm_names:
                kwargs["llm_name"] = llm_names[0]
            return HfToDialogue(**kwargs).out_hf
        assert num_shards > 1
        base_llm = kwargs.pop("llm_name", "google/gemma-4-31B-it")
        shards = []
        for shard in range(num_shards):
            shard_llm = llm_names[shard % len(llm_names)] if llm_names else base_llm
            shards.append(
                HfToDialogue(
                    shard=shard,
                    num_shards=num_shards,
                    llm_name=shard_llm,
                    seed_offset=shard,  # different seed offset per shard for more variety
                    **kwargs,
                )
            )
        return HfMergeShards(shard_paths=[s.out_hf for s in shards]).out_hf

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        with vllm_server(self.llm_name) as llm_url:
            print("Now loading dataset")
            dataset = load_from_disk(self.dataset_split_path.get())
            if self.shard is not None and self.num_shards is not None:
                dataset = dataset.shard(num_shards=self.num_shards, index=self.shard)

            print(
                f"Dataset loaded ({len(dataset)} rows). Generating dialogues "
                f"(adapter={self.adapter_name!r}, llm={self.llm_name!r}, "
                f"shard={self.shard}/{self.num_shards}) …"
            )

            gen_fn = make_dialogue_gen(llm_url, self.llm_name, self.adapter_name)
            dataset = dataset.map(gen_fn, num_proc=32)
            print("Dialogues generated. Saving …")

            dataset.save_to_disk(self.out_hf.get())
            dataset.to_json(self.out_json.get())


class HfDialogueToJsonFile(Job):
    def __init__(
        self,
        *,
        hf_dataset_path: tk.Path,
        split: str | None = None,
        ignore_errors: bool = False,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.split = split
        self.ignore_errors = ignore_errors
        self.out_json = self.output_path("dialogue.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dataset = load_from_disk(self.hf_dataset_path.get())
        if self.split is not None:
            assert type(dataset) is DatasetDict
            dataset = dataset[self.split]
        else:
            assert type(dataset) is Dataset

        num_errors = 0
        with open(self.out_json.get(), "w") as f:
            for example in dataset:
                dialogue_str: str = example["dialogue"]
                dialogue_str = dialogue_str.strip()
                try:
                    if dialogue_str.startswith("```json"):
                        dialogue_str = dialogue_str[len("```json") :]
                    if dialogue_str.endswith("```"):
                        dialogue_str = dialogue_str[: -len("```")]
                    dialogue_json = json.loads(dialogue_str)
                    f.write(json.dumps(dialogue_json) + "\n")
                except Exception as e:
                    num_errors += 1
                    if not self.ignore_errors:
                        print("###")
                        print(dialogue_str)
                        print(f"Error parsing dialogue: {e} for {example}")
                        print("###")

        if num_errors > 0 and not self.ignore_errors:
            raise ValueError(f"Encountered {num_errors} errors while parsing dialogues.")
        if num_errors > len(dataset) * 0.1:
            raise ValueError(
                f"Encountered {num_errors} errors while parsing dialogues "
                f"({num_errors}/{len(dataset)}, > 10%). Something might be wrong."
            )


class HfDialogueCleaner(Job):
    """Parse + filter dialogue JSON from an HF dataset, save clean copy."""

    def __init__(
        self,
        *,
        hf_dataset_path: tk.Path,
        split: str | None = None,
        ignore_errors: bool = False,
    ):
        self.hf_dataset_path = hf_dataset_path
        self.split = split
        self.ignore_errors = ignore_errors
        self.out_hf = self.output_path("out_hf", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        dataset = load_from_disk(self.hf_dataset_path.get())
        if self.split is not None:
            assert type(dataset) is DatasetDict
            dataset = dataset[self.split]
        else:
            assert type(dataset) is Dataset

        def clean_dialogue(example):
            dialogue_str: str = example["dialogue"]
            dialogue_str = dialogue_str.strip()
            if dialogue_str.startswith("```json"):
                dialogue_str = dialogue_str[len("```json") :]
            if dialogue_str.endswith("```"):
                dialogue_str = dialogue_str[: -len("```")]
            example["dialogue"] = dialogue_str
            return example

        cleaned = dataset.map(clean_dialogue)

        def filter_dialogue(example):
            try:
                json.loads(example["dialogue"])
                return True
            except Exception as e:
                if not self.ignore_errors:
                    print(f"Error parsing dialogue: {e}")
                return False

        previous_len = len(cleaned)
        filtered = cleaned.filter(filter_dialogue)
        num_errors = previous_len - len(filtered)
        if num_errors > 0 and not self.ignore_errors:
            print(f"Encountered {num_errors} errors while parsing dialogues.")
        if num_errors > len(dataset) * 0.1:
            raise ValueError(
                f"Encountered {num_errors} errors while parsing dialogues "
                f"({num_errors}/{len(dataset)}, > 10%). Something might be wrong."
            )
        filtered.save_to_disk(self.out_hf.get())
