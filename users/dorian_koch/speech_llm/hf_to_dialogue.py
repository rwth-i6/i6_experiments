"""
hf_to_dialogue.py — convert HuggingFace datasets into spoken dialogue training examples.

Supports multiple source datasets (MMLU-Pro multiple-choice, TriviaQA open-ended, …)
via a DATASET_ADAPTER_REGISTRY, multiple LLMs round-robined per shard, and a pool of
dialogue-instruction prompt templates chosen deterministically per example to maximise
style diversity and avoid the "That would be …" monoculture.
"""

from sisyphus import Job, Task, tk
import os
import hashlib
from .common import vllm_server, write_progress
from datasets import (
    load_from_disk,
    Dataset,
    DatasetDict,
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
    # Template 6 — engaging teacher (PersonaPlex's QA persona: their fixed role prompt is
    # "You are a wise and friendly teacher. Answer questions or provide advice in a clear and
    # engaging way."). Deliberately asks for elaboration, since our corpus was too terse.
    (
        "You are a wise and friendly teacher.  Write a spoken dialogue where the user asks the "
        "question and you answer in a clear and engaging way: give the correct answer directly, "
        "then add one or two sentences of genuinely interesting context or a memorable detail.  "
        "Warm and vivid, never a lecture and never padded.  The user MUST speak first.  "
        "Total turns: 2-4, and the assistant's answers should be substantial (roughly 40-80 "
        "spoken words in total).\n" + _COMMON_SUFFIX
    ),
    # Template 7 — follow-up / topic change. PersonaPlex explicitly synthesises "two-turn
    # question-answering dialogs ... and second-question scenarios (topic change, follow up etc.)",
    # which our single-question corpus had none of.
    (
        "Write a spoken dialogue with TWO questions.  First the user asks the question and the "
        "assistant answers it clearly.  Then the user asks a SECOND question — either a natural "
        "follow-up about the same topic, or an unrelated change of subject — and the assistant "
        "answers that too, without being thrown by the switch.  The user MUST speak first.  "
        "Total turns: 4-6.\n" + _COMMON_SUFFIX
    ),
    # --- Templates 8-10: MANY SHORT TURNS ----------------------------------------------------
    # These raise total assistant words WITHOUT making individual turns long-winded. The terse
    # templates (qa_direct / casual_brief) were the problem not because short turns are wrong --
    # real speech is full of them, and a duplex model must handle rapid exchange -- but because a
    # 2-turn dialogue gives the model almost no assistant speech to learn from. Keeping turns short
    # while adding more of them preserves the natural conversational register and still gets the
    # per-example word count up.
    #
    # Template 8 — quickfire: several short related questions, brief answers to each.
    (
        "Write a spoken dialogue in which the user asks several short, quick questions about the "
        "topic, one at a time, and the assistant answers each one briefly — one or two sentences "
        "at most per turn, never a lecture.  The exchange should feel fast and easy, like two "
        "people talking quickly.  Make sure the correct answer to the main question is clearly "
        "stated in one of the assistant's turns.  The user MUST speak first.  Total turns: 6-8.\n" + _COMMON_SUFFIX
    ),
    # Template 9 — clarify then answer: a short clarification round before the answer.
    (
        "Write a spoken dialogue where the user asks the question a little vaguely, the assistant "
        "asks one short clarifying question back, the user clarifies, and the assistant then gives "
        "the correct answer and one brief remark about it.  Keep every turn short and natural — "
        "this is a quick back-and-forth, not an interview.  The user MUST speak first.  "
        "Total turns: 5-7.\n" + _COMMON_SUFFIX
    ),
    # Template 10 — reactive chat: the user reacts/backchannels, the assistant adds a bit each time.
    (
        "Write a casual spoken exchange where the assistant gives the answer in small pieces across "
        "several short turns, and the user reacts in between with brief responses like 'oh really', "
        "'huh', 'right', or a one-line follow-up.  Every assistant turn is short — one or two "
        "sentences — but together they cover the answer and an interesting detail or two.  "
        "The user MUST speak first.  Total turns: 6-8.\n" + _COMMON_SUFFIX
    ),
]


# Human-readable names for each template (same order as DIALOGUE_INSTRUCTION_TEMPLATES).
# Stored as provenance metadata per dialogue row.
DIALOGUE_INSTRUCTION_TEMPLATE_NAMES = [
    "qa_direct",
    "qa_user_guesses",
    "casual_brief",
    "multi_turn_explanation",
    "skeptical_user",
    "socratic",
    "engaging_teacher",
    "followup_topic_change",
    "quickfire",
    "clarify_then_answer",
    "reactive_chat",
]
assert len(DIALOGUE_INSTRUCTION_TEMPLATE_NAMES) == len(DIALOGUE_INSTRUCTION_TEMPLATES)


def dialogue_template_distribution(n: int = 100000) -> dict[str, float]:
    """Realised sampling probability per template name, for sanity-checking the weights.

    Uses the same uid->template path as generation, so it reflects what a corpus would actually get
    rather than the nominal weights.
    """
    import collections

    counts = collections.Counter(_pick_template(str(i))[1] for i in range(n))
    return {k: v / n for k, v in sorted(counts.items())}


#: Sampling weight per template (same order as DIALOGUE_INSTRUCTION_TEMPLATES). Relative, not
#: normalised. This is what makes the dialogue *shape* probabilistic: every example draws a template,
#: so the corpus mixes 2-turn one-liners with 4-6 turn exchanges.
#:
#: Weights exist because uniform sampling gave a **bimodal, too-terse corpus** (measured 2026-07-23
#: over 3,000 dialogues): `qa_direct` and `casual_brief` collapsed to exactly 2 turns and ~5
#: assistant words, and together they were a third of the data. Overall the corpus averaged 20.8
#: assistant words / 16.0 s per example, versus PersonaPlex's ~37.5 s two-turn QA. Since a finetune
#: learns response *length* as readily as content, a corpus that is one-third one-liners plausibly
#: teaches terseness -- which is the passivity / collapsed take_turn we already observe.
#:
#: So: keep short exchanges present (real conversation has them) but stop them dominating, and give
#: the richer templates more mass.
#: Three ways to get the assistant-word count up, deliberately balanced rather than picking one:
#:   long turns   — engaging_teacher, multi_turn_explanation, skeptical_user
#:   many turns   — quickfire, reactive_chat, clarify_then_answer  (short turns, more of them)
#:   more topics  — followup_topic_change
#: The "many short turns" group matters most for a DUPLEX model: it is the only group that trains
#: rapid exchange and frequent turn-taking, which is exactly the behaviour our finetunes lost
#: (collapsed take_turn). Long-turn templates alone would raise word count while making the model
#: MORE monologue-ish -- the opposite of what we want.
DIALOGUE_TEMPLATE_WEIGHTS = [
    1,  # qa_direct              — 2 turns, ~5 words; kept for variety only
    2,  # qa_user_guesses        — mid
    1,  # casual_brief           — 2 turns, ~5 words; kept for variety only
    3,  # multi_turn_explanation — long turns
    2,  # skeptical_user         — long turns
    2,  # socratic               — mid
    3,  # engaging_teacher       — long turns (PersonaPlex QA persona)
    2,  # followup_topic_change  — two questions
    3,  # quickfire              — many short turns
    2,  # clarify_then_answer    — several short turns
    3,  # reactive_chat          — many short turns w/ backchannels
]


def _pick_template(uid: str) -> tuple[str, str]:
    """Deterministically pick a template (and its name) for ``uid``, weighted by
    :data:`DIALOGUE_TEMPLATE_WEIGHTS`.

    Deterministic on purpose: the same example always gets the same template, so regenerating a
    shard reproduces the same corpus. Weighted rather than uniform -- see the note on the weights.
    """
    h = int(hashlib.md5(uid.encode()).hexdigest(), 16)
    total = sum(DIALOGUE_TEMPLATE_WEIGHTS)
    r = h % total
    for idx, w in enumerate(DIALOGUE_TEMPLATE_WEIGHTS):
        r -= w
        if r < 0:
            return DIALOGUE_INSTRUCTION_TEMPLATES[idx], DIALOGUE_INSTRUCTION_TEMPLATE_NAMES[idx]
    raise AssertionError("unreachable: weights must sum > 0")


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


# JSON schema enforced at decode time (vLLM structured output): a top-level
# array of {speaker, text} turns.  This is what the prompts ask for; without a
# schema, response_format json_object forces a wrapper dict around the array.
_DIALOGUE_JSON_SCHEMA = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "speaker": {"type": "string", "enum": ["user", "assistant"]},
            "text": {"type": "string", "minLength": 1},
        },
        "required": ["speaker", "text"],
        "additionalProperties": False,
    },
}


def make_dialogue_gen(
    llm_url: str,
    model_name: str,
    adapter_name: str,
    work_dir: str | None = None,
    total: int | None = None,
    template_name: str | None = None,
):
    """Return a datasets.map-compatible function that generates one dialogue.

    ``template_name``: if given, EVERY example uses that one template instead of drawing from the
    weighted mix. This is what lets each template be generated as its own job (see
    ``make_per_template_dialogues`` in the recipe): a per-template corpus can then be added, resized
    or regenerated on its own, and adding template N+1 does not touch the N that already exist.
    """
    if adapter_name == "service":
        from . import personaplex_service_data  # noqa: F401  (registers the "service" adapter)
    adapter = DATASET_ADAPTER_REGISTRY[adapter_name]
    _local_done = [0]

    def make_dialogue(example, seed_offset: int = 0):
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "nothing"),
            base_url=llm_url,
        )
        spec = adapter(example)
        if adapter_name == "service":
            # Service scenarios use their own role-grounded templates + message builder
            # (personaplex_service_data); the QA path below is unchanged -> hash-safe.
            from .personaplex_service_data import build_service_user_message, pick_service_template

            # `resolved_template_name` (not `template_name`) is what gets recorded, because
            # `template_name` is the enclosing *parameter* -- assigning to it here would make it a
            # local and break the single-template branch below with UnboundLocalError.
            template, resolved_template_name = pick_service_template(str(spec["uid"]))
            user_msg = build_service_user_message(spec, template)
        else:
            if template_name is not None:
                # Single-template mode: pinned by the job, not drawn per example.
                idx = DIALOGUE_INSTRUCTION_TEMPLATE_NAMES.index(template_name)
                template, resolved_template_name = (
                    DIALOGUE_INSTRUCTION_TEMPLATES[idx],
                    template_name,
                )
            else:
                template, resolved_template_name = _pick_template(str(spec["uid"]))
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
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    seed=(seed + attempt) % (2**31),
                    temperature=temp,
                    extra_body={"guided_json": _DIALOGUE_JSON_SCHEMA},
                    max_tokens=800,
                )
            except Exception as e:
                print(f"API error on attempt {attempt + 1} (uid={spec['uid']!r}): {e!r}")
                parsed = []
            else:
                finish_reason = response.choices[0].finish_reason
                raw = response.choices[0].message.content
                if finish_reason == "length":
                    print(f"Truncated by max_tokens on attempt {attempt + 1} (uid={spec['uid']!r})")
                try:
                    parsed = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    print(f"JSONDecodeError on attempt {attempt + 1} (uid={spec['uid']!r}): raw={raw!r}")
                    parsed = []
            if isinstance(parsed, list) and parsed and parsed[0].get("speaker") == "user":
                result = {
                    "dialogue": json.dumps(parsed),
                    "llm_name": model_name,
                    "template_name": resolved_template_name,
                    "truncated": finish_reason == "length",
                }
                break
            print(
                f"Attempt {attempt + 1} failed (uid={spec['uid']!r}): "
                f"first speaker={parsed[0].get('speaker') if isinstance(parsed, list) and parsed else repr(parsed)[:120]!r}"
            )
            print(f"  Generated: {json.dumps(parsed)[:300]}")
        else:
            # All attempts failed — mark row as None so HfToDialogue.run() can filter and count
            print(f"All {max_attempts} attempts failed for uid={spec['uid']!r}. Marking as failed.")
            result = {
                "dialogue": None,
                "llm_name": model_name,
                "template_name": resolved_template_name,
                "truncated": False,
            }
        _local_done[0] += 1
        if work_dir and total and _local_done[0] % 10 == 0:
            try:
                write_progress(_local_done[0], total, os.path.join(work_dir, f"progress_{os.getpid()}.json"))
            except Exception:
                pass
        return result

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
        templates_version: int = 1,
        template_name: str | None = None,
    ):
        self.dataset_split_path = dataset_split_path
        self.llm_name = llm_name
        # dialogue_instructions kept in hash for backward compat with existing jobs
        self.dialogue_instructions = dialogue_instructions or ""
        self.adapter_name = adapter_name
        self.shard = shard
        self.num_shards = num_shards
        self.seed_offset = seed_offset
        # DIALOGUE_INSTRUCTION_TEMPLATES / DIALOGUE_TEMPLATE_WEIGHTS are module constants, so editing
        # them does NOT change this job's hash and a finished corpus would silently NOT regenerate.
        # Bump `templates_version` to force a fresh corpus (and the TTS/annotate cascade below it)
        # after changing the prompt set. Hash-excluded at its default, so it is a no-op until bumped.
        self.templates_version = templates_version
        # None = draw from the weighted template mix (legacy behaviour, one job for the whole
        # corpus). A name = this job generates ONLY that template, which is what makes templates
        # independently addable: each becomes its own job with its own hash, so adding template N+1
        # leaves the N existing corpora finished and untouched. Merge them afterwards.
        self.template_name = template_name

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
        if d.get("templates_version", 1) == 1:
            d.pop("templates_version", None)  # no-op at the default: existing corpora keep their hash
        if d.get("template_name") is None:
            d.pop("template_name", None)  # no-op unless single-template mode is used
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

    def completed_fraction(self):
        import glob
        from sisyphus import global_settings as gs

        work_dir = self._sis_path(gs.JOB_WORK_DIR)
        files = glob.glob(os.path.join(work_dir, "progress_*.json"))
        if not files:
            return None
        total = None
        done = 0
        for f in files:
            try:
                d = json.load(open(f))
                done += d.get("done", 0)
                if total is None:
                    total = d.get("total")
            except (OSError, ValueError):
                pass
        if not total:
            return None
        return min(done / total, 1.0)

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

            work_dir = os.getcwd()
            gen_fn = make_dialogue_gen(
                llm_url,
                self.llm_name,
                self.adapter_name,
                work_dir=work_dir,
                total=len(dataset),
                template_name=self.template_name,
            )
            dataset = dataset.map(gen_fn, num_proc=32)

            # Count and remove failed rows; abort if failure rate > 1%
            num_failures = sum(1 for row in dataset if row["dialogue"] is None)
            if num_failures > 0:
                print(f"Dialogue generation: {num_failures}/{len(dataset)} rows failed.")
            if num_failures > len(dataset) * 0.01:
                raise ValueError(
                    f"Dialogue generation failure rate too high: "
                    f"{num_failures}/{len(dataset)} ({100 * num_failures / len(dataset):.1f}% > 1%). "
                    "Check vLLM server or reduce temperature."
                )
            dataset = dataset.filter(lambda row: row["dialogue"] is not None)

            # Check truncation rate on surviving rows
            num_truncated = sum(1 for row in dataset if row.get("truncated"))
            if num_truncated > 0:
                print(
                    f"Truncated by max_tokens: {num_truncated}/{len(dataset)} ({100 * num_truncated / len(dataset):.1f}%)"
                )
            if num_truncated > len(dataset) * 0.01:
                raise ValueError(
                    f"Too many responses truncated by max_tokens: "
                    f"{num_truncated}/{len(dataset)} ({100 * num_truncated / len(dataset):.1f}% > 1%). "
                    "Increase max_tokens or tighten the prompt."
                )
            print(f"Dialogues generated ({len(dataset)} rows after filtering). Saving …")

            dataset.save_to_disk(self.out_hf.get())
            dataset.to_json(self.out_json.get())


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
        if num_errors > len(dataset) * 0.01:
            raise ValueError(
                f"Encountered {num_errors} errors while parsing dialogues "
                f"({num_errors}/{len(dataset)}, > 1%). Something might be wrong."
            )
        filtered.save_to_disk(self.out_hf.get())
