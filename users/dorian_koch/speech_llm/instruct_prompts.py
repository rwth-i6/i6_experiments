"""Spoken instruction prompts for instruction-following GRPO (user, 2026-10-03).

User: "implement one or more of those instruct datasets for use with RL training". GRPO needs PROMPTS (the reply is
sampled from the policy and judged), so only the user turn is synthesised; the dataset's own reply is kept as the
judge's REFERENCE (``moshi_family.rl.instruct_reward``). Sources: human-written requests with human replies and
permissive licences (checked on their dataset cards 2026-10-03):

  dolly15k   databricks/databricks-dolly-15k, CC BY-SA 3.0. Categories kept: open_qa, general_qa, brainstorming,
             creative_writing, classification -- WITHOUT a context passage (the others are reading tasks).
  oasst2_en  OpenAssistant/oasst2, Apache-2.0. First user turn of each English tree (not deleted, review passed) and
             its rank-0 assistant reply.

Pipeline: ``LoadInstructPrompts`` (normalise, rule filter) -> ``InstructPromptPrep`` (one LLM call per row: keep or
drop with a reason, rewrite the request as speech, condense the reference to spoken length) -> the knowledge
benchmark's TTS (``ChatterboxSingleSpeakerInference``, which reads ``question``). Output columns match
``LLMPreprocess`` (``question, answer, aliases, category``) plus ``source``, ``orig_question``, ``orig_answer``, so the
knowledge GRPO question source reads it unchanged. Every drop is counted by reason in ``stats.json``; verbatim
kept/dropped examples go to ``examples.jsonl``.
"""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk

from .common import map_concurrent, vllm_gpu_mem_gb

DOLLY_KEEP = ("open_qa", "general_qa", "brainstorming", "creative_writing", "classification")
INSTRUCT_SOURCES = ("dolly15k", "oasst2_en")
_WORD = re.compile(r"[A-Za-z0-9']+")
# Requests a voice reply cannot serve, caught before the LLM call (the LLM filter catches the rest)
_FORMAT_RE = re.compile(
    r"```|\bcode\b|\bpython\b|javascript|\bjava\b|c\+\+|\bsql\b|\bregex\b|\bhtml\b|\bcss\b|\bjson\b|\blatex\b|"
    r"\bscript\b|\bfunction\b|\btable\b|spreadsheet|https?://|www\.|\bessay\b|\b\d{3,} words\b",
    re.IGNORECASE,
)
MAX_Q_WORDS = 60
MAX_REF_CHARS = 3000


def n_words(text: str) -> int:
    return len(_WORD.findall(text or ""))


def rule_drop(question: str, reference: str) -> str | None:
    """-> the drop reason, or None to keep for the LLM pass."""
    if not question.strip() or not reference.strip():
        return "empty"
    if not 3 <= n_words(question) <= MAX_Q_WORDS:
        return "question_length"
    if n_words(reference) < 3:
        return "reference_short"
    if _FORMAT_RE.search(question):
        return "code_or_format"
    return None


def oasst_first_turns(rows) -> list[dict]:
    """oasst2 message rows -> ``[{"id", "question", "answer"}]``: each English tree's first user turn and its best
    assistant reply (rank 0; an unranked single reply counts as rank 0). Deleted or review-failed messages are skipped."""
    ok = [r for r in rows if not r.get("deleted") and r.get("review_result") is not False]
    prompts = {
        r["message_id"]: r for r in ok if r["role"] == "prompter" and r["parent_id"] is None and r["lang"] == "en"
    }
    best: dict[str, tuple] = {}
    for r in ok:
        if r["role"] == "assistant" and r["parent_id"] in prompts:
            rank = r["rank"] if r.get("rank") is not None else 0
            if r["parent_id"] not in best or rank < best[r["parent_id"]][0]:
                best[r["parent_id"]] = (rank, r["text"])
    return [{"id": pid, "question": prompts[pid]["text"], "answer": t} for pid, (_, t) in sorted(best.items())]


class LoadInstructPrompts(Job):
    """Normalise the instruction sources to ``question, answer, aliases ("[]"), category, source, id`` and apply
    ``rule_drop``. ``stats.json``: rows per source and drop counts per reason."""

    def __init__(self, *, sources: tuple = INSTRUCT_SOURCES):
        assert set(sources) <= set(INSTRUCT_SOURCES), sources
        self.sources = tuple(sources)
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from collections import Counter

        from datasets import Dataset, load_dataset

        rows, stats = [], {}
        for src in self.sources:
            if src == "dolly15k":
                ds = load_dataset("databricks/databricks-dolly-15k", split="train")
                cand = []
                for i, r in enumerate(ds):
                    if r["category"] not in DOLLY_KEEP:
                        cand.append((None, "category"))
                    elif (r["context"] or "").strip():
                        cand.append((None, "has_context"))
                    else:
                        cand.append(({"id": f"dolly_{i}", "question": r["instruction"], "answer": r["response"],
                                      "category": r["category"]}, None))  # fmt: skip
            else:
                ds = load_dataset("OpenAssistant/oasst2", split="train")
                cand = [({**t, "category": "oasst"}, None) for t in oasst_first_turns(ds)]
            drops = Counter()
            n_in = len(cand)
            for row, reason in cand:
                reason = reason or rule_drop(row["question"], row["answer"])
                if reason:
                    drops[reason] += 1
                    continue
                rows.append({
                    "id": row["id"],
                    "question": row["question"].strip(),
                    "answer": row["answer"].strip()[:MAX_REF_CHARS],
                    "aliases": "[]",
                    "category": row["category"],
                    "source": src,
                })  # fmt: skip
            stats[src] = {"candidates": n_in, "kept": n_in - sum(drops.values()), "dropped": dict(drops)}
            print(f"[instruct] {src}: {stats[src]}", flush=True)
        Dataset.from_list(rows).save_to_disk(self.out_hf.get_path())
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=1)


#: The keep/rewrite instruction (``InstructPromptPrep``). Prompt text is not hashed: bump ``prompt_version`` with it.
PREP_INSTRUCTIONS = (
    "You prepare prompts for training a VOICE assistant that talks with a user in real time. Below is a request a "
    "person typed to a text assistant, and a reply a person wrote.\n"
    "Decide whether it makes a good SPOKEN request: something a person could plausibly ask a voice assistant out "
    "loud, which the assistant can answer well in under about 20 seconds of speech. Reject it (keep=false) if it "
    "needs code, maths working, a table, a long list, or a long text (an essay, a full email or letter, a story "
    "longer than a few sentences); refers to a document, passage, link, image or earlier conversation that is not "
    "given; depends on today's date, the user's location or personal details, or on recent events; is unsafe, "
    "sexual or hateful; is not in English; or if the written reply is wrong or unhelpful and you cannot tell what a "
    "good answer is.\n"
    "If keep=true, also write:\n"
    "- question: the request as the user would say it aloud: natural spoken English, self-contained, no formatting, "
    "numbers and symbols written as words, at most 40 words. Keep its meaning and intent.\n"
    "- reference: what a good voice assistant would say back, based on the written reply (fix it if it is clearly "
    "wrong): plain spoken sentences, no lists or formatting, at most 70 words.\n"
    'Respond with JSON only: {"keep": true or false, "reason": "<a few words>", "question": "<or empty>", '
    '"reference": "<or empty>"}'
)


class InstructPromptPrep(Job):
    """One LLM call per ``LoadInstructPrompts`` row (``PREP_INSTRUCTIONS``): keep/drop with a reason, the spoken
    request (``question``) and the spoken-length reference (``answer``). The originals stay in ``orig_question`` /
    ``orig_answer``. ``stats.json``: kept/dropped per source and category, drop reasons (LLM ``reason`` strings,
    counted verbatim); ``examples.jsonl``: up to 25 kept and 25 dropped rows per source, verbatim."""

    def __init__(self, *, in_hf: tk.Path, llm_name: str = "google/gemma-4-31B-it", prompt_version: int = 1):
        self.in_hf = in_hf
        self.llm_name = llm_name
        self.prompt_version = prompt_version
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.out_examples = self.output_path("examples.jsonl")
        self.rqmt = {"gpu": 1, "cpu": 8, "mem": 32, "time": 6, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import time
        from collections import Counter, defaultdict

        from datasets import Dataset, load_from_disk
        from openai import OpenAI

        from .common import vllm_server

        rows = list(load_from_disk(str(self.in_hf.get())))
        print(f"[instruct-prep] {len(rows)} rows, model {self.llm_name}, prompt v{self.prompt_version}", flush=True)
        t0 = time.time()
        # 12288: LLMPreprocess's measured fit for gemma-4-31B on an 80 GB card; these prompts are < 2k tokens
        with vllm_server(self.llm_name, max_model_len=12288) as url:
            client = OpenAI(api_key="EMPTY", base_url=url, timeout=300)

            def call(r):
                user = f"Request: {r['question']}\n\nWritten reply: {r['answer']}\n\n{PREP_INSTRUCTIONS}"
                for _ in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=self.llm_name,
                            messages=[{"role": "user", "content": user}],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                            max_tokens=700,
                        )
                        d = json.loads(resp.choices[0].message.content)
                        keep = d.get("keep") is True
                        q, ref = str(d.get("question") or "").strip(), str(d.get("reference") or "").strip()
                        if keep and (not q or not ref):
                            return {"keep": False, "reason": "llm_empty_rewrite"}
                        return {"keep": keep, "reason": str(d.get("reason") or "").strip(), "question": q,
                                "reference": ref}  # fmt: skip
                    except Exception as e:  # noqa: BLE001 -- retried, then counted as a drop
                        err = repr(e)
                return {"keep": False, "reason": f"llm_fail: {err[:80]}"}

            outs = map_concurrent(call, rows, concurrency=96, progress_every=500)
        print(f"[instruct-prep] LLM pass {time.time() - t0:.0f} s", flush=True)

        kept, stats, reasons = [], defaultdict(Counter), defaultdict(Counter)
        examples = defaultdict(lambda: {"kept": [], "dropped": []})
        for r, o in zip(rows, outs):
            key = f"{r['source']}/{r['category']}"
            stats[key]["kept" if o["keep"] else "dropped"] += 1
            ex = examples[r["source"]]["kept" if o["keep"] else "dropped"]
            if len(ex) < 25:
                ex.append({"id": r["id"], "category": r["category"], "orig_question": r["question"],
                           "orig_answer": r["answer"][:600], **o})  # fmt: skip
            if not o["keep"]:
                reasons[r["source"]][o["reason"].lower()[:60]] += 1
                continue
            kept.append({
                "question": o["question"],
                "answer": o["reference"],
                "aliases": "[]",
                "category": r["category"],
                "source": r["source"],
                "id": r["id"],
                "orig_question": r["question"],
                "orig_answer": r["answer"],
            })  # fmt: skip
        Dataset.from_list(kept).save_to_disk(self.out_hf.get_path())
        summary = {
            "rows_in": len(rows),
            "kept": len(kept),
            "by_category": {k: dict(v) for k, v in sorted(stats.items())},
            "top_drop_reasons": {s: dict(c.most_common(25)) for s, c in reasons.items()},
        }
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(summary, f, indent=1)
        with open(self.out_examples.get_path(), "w") as f:
            for src, d in examples.items():
                for kind in ("kept", "dropped"):
                    for e in d[kind]:
                        f.write(json.dumps({"source": src, "kind": kind, **e}) + "\n")
        print(f"[instruct-prep] kept {len(kept)} / {len(rows)}", flush=True)
