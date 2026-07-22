"""MoshiRAG training-data construction -- DATA SPIKE (design + tiny sample, no real run).

MoshiRAG (arXiv 2604.12928) trains Moshi to emit a ``<ret>`` retrieval-trigger token before
the *lead* of a knowledge-intensive reply, so an asynchronous retrieval completes before the
reply's key content is spoken. The paper builds its data with three LLM roles (user, Moshi,
reference) and structures each grounded reply as **lead** (no external knowledge) / **body**
(reference-grounded) / optional **tail**, then "replaces the text token before the first text
token in the lead portion" with ``<ret>`` and attaches the retrieved reference text.

This module is the *spike*: it (1) documents how we reuse the existing dialogue machinery and
(2) provides a deterministic, LLM-free transformer + a tiny built-in sample so we can eyeball
the construction now -- no TTS / annotate / training. The two pieces:

* ``RAG_DIALOGUE_INSTRUCTIONS`` -- a paper-faithful prompt for the dialogue generator's
  "reference" role + lead/body/tail structure. It plugs into the *existing*
  ``HfToDialogue`` via its ``dialogue_instructions`` parameter (see
  ``make_dialogue_from_hf_dataset(..., dialogue_instructions=...)``) -- so the real RAG
  data-gen reuses all of the dialogue pipeline; only this prompt + the post-processor below
  are new.
* ``insert_ret`` / ``ragify_dialogue`` -- the post-processor that splits an assistant reply
  into lead/body/tail, prepends ``<ret>``, and attaches the reference. This is the lean,
  testable core of the construction; wire it as a cleaner step after dialogue generation.

Run ``python -m i6_experiments.users.dorian_koch.speech_llm.moshirag_data`` (login node, no
GPU) to print the sample rows. See ``projects/2026-01-speech-llm/moshirag.md``.
"""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk

# The retrieval-trigger token. VERIFY(moshirag): confirm the exact surface form / special-token
# id the released moshika-rag tokenizer uses; the training collator must map this to that id.
RET_TOKEN = "<ret>"


# Reusable prompt for the dialogue generator (plugs into HfToDialogue.dialogue_instructions).
# Asks for a lead/body/tail reply plus a separate reference snippet, mirroring the paper's
# user/Moshi/reference role design without adding any new job -- the generator LLM (our vLLM
# Gemma) produces both the dialogue and the reference text in one structured response.
RAG_DIALOGUE_INSTRUCTIONS = (
    "Write a short spoken dialogue where the user asks a knowledge-intensive question and the "
    "assistant answers. For each assistant turn that needs an external fact, structure the "
    "reply as three parts: a brief LEAD that buys time without stating the fact (e.g. 'Oh, "
    "good question - let me think'), a BODY that delivers the factual answer, and an optional "
    "short TAIL. Also output, per such turn, a one-sentence REFERENCE giving the supporting "
    "fact (as if retrieved from a knowledge source). Keep it natural and spoken. Return JSON: "
    'a list of {"speaker": "user"|"assistant", "text": ..., "lead": ..., "body": ..., '
    '"tail": ..., "reference_text": ...}; non-knowledge turns omit the lead/body/tail/reference '
    "fields."
)

# Heuristic lead/body split for replies that arrive as one flat string (e.g. legacy dialogues
# without explicit lead/body fields): the lead is everything up to and including the first
# sentence-ending punctuation; the rest is the body. Real data should carry explicit fields.
_SENT_END = re.compile(r"(.+?[.!?—-])\s+(.*)", re.DOTALL)


def split_reply(text: str) -> tuple[str, str]:
    """Split a flat assistant reply into (lead, body). Falls back to (\"\", text)."""
    m = _SENT_END.match(text.strip())
    if not m:
        return "", text.strip()
    return m.group(1).strip(), m.group(2).strip()


def insert_ret(turn: dict) -> dict:
    """Turn one assistant turn into a RAG training turn.

    If the turn carries explicit ``lead``/``body`` (the generator path), use them; else split
    the flat ``text``. Prepend ``<ret>`` before the first lead token (paper: <ret> replaces the
    token before the first text token of the lead) and keep the reference text alongside.
    Non-knowledge turns (no reference) are returned unchanged.
    """
    reference = turn.get("reference_text")
    if not reference or turn.get("speaker") != "assistant":
        return {"speaker": turn["speaker"], "text": turn["text"]}
    lead = turn.get("lead")
    body = turn.get("body")
    tail = turn.get("tail") or ""
    if lead is None or body is None:
        lead, body = split_reply(turn["text"])
    spoken = " ".join(p for p in [lead, body, tail] if p).strip()
    return {
        "speaker": "assistant",
        # <ret> at the very start of the spoken text -> triggers retrieval before the lead.
        "text": f"{RET_TOKEN} {spoken}".strip(),
        "reference_text": reference,
        "rag_trigger": True,
        "lead": lead,
        "body": body,
        "tail": tail,
    }


def ragify_dialogue(turns: list[dict]) -> list[dict]:
    """Apply :func:`insert_ret` to every turn of a dialogue."""
    return [insert_ret(t) for t in turns]


# A tiny built-in sample (LLM-free) to eyeball the construction now. Each base reply already
# carries an explicit lead/body so the transform is deterministic.
_SAMPLE = [
    {
        "question": "Who wrote the play Hamlet?",
        "lead": "Oh, good question -",
        "body": "Hamlet was written by William Shakespeare, around 1600.",
        "reference": "Hamlet is a tragedy written by William Shakespeare, believed composed between 1599 and 1601.",
    },
    {
        "question": "What is the capital of Australia?",
        "lead": "Let me think for a second.",
        "body": "The capital of Australia is Canberra, not Sydney as many assume.",
        "reference": "Canberra is the capital city of Australia, chosen as a compromise between Sydney and Melbourne in 1908.",
    },
    {
        "question": "How tall is Mount Kilimanjaro?",
        "lead": "Hmm, good one.",
        "body": "Mount Kilimanjaro rises to about 5,895 metres above sea level.",
        "reference": "Mount Kilimanjaro's highest point, Uhuru Peak, is 5,895 m (19,341 ft) above sea level.",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "lead": "Ah, that one I know -",
        "body": "the Mona Lisa was painted by Leonardo da Vinci in the early 1500s.",
        "reference": "The Mona Lisa is a portrait painted by Leonardo da Vinci, dated to circa 1503-1519.",
    },
]


def make_rag_dialogues_sample(n: int = 12) -> list[list[dict]]:
    """Build ~n sample RAG dialogues (user question + ragified assistant reply), LLM-free.

    Repeats the built-in examples to reach ``n`` rows so the transform is visible at a glance.
    """
    rows = []
    for i in range(n):
        ex = _SAMPLE[i % len(_SAMPLE)]
        base = [
            {"speaker": "user", "text": ex["question"]},
            {
                "speaker": "assistant",
                "text": f"{ex['lead']} {ex['body']}",
                "lead": ex["lead"],
                "body": ex["body"],
                "reference_text": ex["reference"],
            },
        ]
        rows.append(ragify_dialogue(base))
    return rows


class RagifyReferences(Job):
    """Attach a per-row ``reference_text`` to a dialogue dataset for MoshiRAG grounded training.

    MoshiRAG conditions generation on a *retrieved reference* (the ARC-Encoder encodes it; the model
    emits ``<ret>`` to trigger retrieval). Training needs, per dialogue, the grounding fact the
    assistant answer rests on. For QA datasets that fact is already in the source row (the gold
    answer carried through dialogue-gen), so we build the reference deterministically from
    ``question`` + ``answer`` -- no extra LLM call.

    The spoken text is left UNCHANGED: ``<ret>`` is a TEXT-stream token injected at *tokenization*
    (``moshi_family.moshirag_train_data.MoshiRagTokenizer``), never spoken, so it must NOT enter the
    TTS text. (The ``insert_ret``/``ragify_dialogue`` helpers above are a paper illustration of where
    ``<ret>`` lands in the token stream; the real pipeline does that injection at tokenization, not
    here.)

    Output = input dataset + a ``reference_text`` string column. Carry it through TTS + annotate via
    their opt-in ``keep_columns=["reference_text"]`` so it reaches the training tokenizer.
    """

    def __init__(
        self,
        *,
        dialogues: tk.Path,
        question_col: str = "question",
        answer_col: str = "answer",
        max_chars: int = 400,
    ):
        self.dialogues = dialogues
        self.question_col = question_col
        self.answer_col = answer_col
        self.max_chars = max_chars
        self.out_hf = self.output_path("out_hf", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from datasets import load_from_disk

        ds = load_from_disk(self.dialogues.get())
        assert self.question_col in ds.column_names, (self.question_col, ds.column_names)
        assert self.answer_col in ds.column_names, (self.answer_col, ds.column_names)

        def add_ref(example):
            q = str(example.get(self.question_col) or "").strip()
            a = example.get(self.answer_col)
            # TriviaQA-style nested answer {value, aliases}; else a plain string/value.
            if isinstance(a, dict):
                a = a.get("value") or ""
            ref = f"{q} {str(a).strip()}".strip()[: self.max_chars]
            example["reference_text"] = ref
            return example

        ds = ds.map(add_ref)
        n_empty = sum(1 for r in ds["reference_text"] if not r)
        assert n_empty < max(1, len(ds) * 0.01), f"too many empty references: {n_empty}/{len(ds)}"
        print(f"[ragify] attached reference_text to {len(ds)} rows ({n_empty} empty)", flush=True)
        ds.save_to_disk(self.out_hf.get())


def demo() -> None:
    rows = make_rag_dialogues_sample(12)
    n_trig = sum(1 for r in rows for t in r if t.get("rag_trigger"))
    print(f"[moshirag-data-spike] {len(rows)} sample dialogues, {n_trig} <ret>-triggered turns\n")
    for i, row in enumerate(rows[:4]):
        print(f"--- sample {i} ---")
        print(json.dumps(row, indent=2, ensure_ascii=False))
    # Sanity checks (turn the construction into guards):
    for row in rows:
        for t in row:
            if t.get("rag_trigger"):
                assert t["text"].startswith(RET_TOKEN), "ret token must lead the assistant text"
                assert t.get("reference_text"), "triggered turn must carry a reference"
    print("\n[moshirag-data-spike] OK: every triggered turn starts with <ret> + has a reference")


if __name__ == "__main__":
    demo()
