"""Spoken instruction prompts for instruction-following GRPO (2026-10-03).

Instruct datasets as RL training data. GRPO needs PROMPTS (the reply is
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

Multi-turn threads (2026-10-11): ``LoadInstructThreads`` (oasst2 rank-0 paths with 2-3 user turns) ->
``InstructThreadPrep`` (one LLM call per THREAD: keep only threads whose follow-ups make sense after any reasonable
reply; every turn rewritten as speech with its own reference) -> the same TTS. One row per user turn, a thread's turns
consecutive, plus ``thread_id`` / ``turn`` / ``n_turns`` (``moshi_family.rl.knowledge_segments.InstructThreadSource``).
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


# --- multi-turn threads (2026-10-11, multi-turn instruct GRPO) ------------------------------------
#: user turns per kept thread (the rank-0 path of an oasst2 tree)
MT_MIN_TURNS, MT_MAX_TURNS = 2, 3


def _best_child(children: list[dict]) -> dict | None:
    """The rank-0 child (an unranked child counts as rank 0; ties: the first in data order)."""
    best = None
    for c in children:
        r = c["rank"] if c.get("rank") is not None else 0
        if best is None or r < best[0]:
            best = (r, c)
    return best[1] if best else None


def oasst_threads(rows, *, min_turns: int = MT_MIN_TURNS, max_turns: int = MT_MAX_TURNS) -> list[dict]:
    """oasst2 message rows -> ``[{"id", "turns": [{"question", "answer"}, ...]}]``: each English tree's rank-0 path
    (user -> best reply -> best user follow-up -> ...) cut to ``max_turns`` user turns, kept when it has at least
    ``min_turns`` user turns that each have a reply. Deleted or review-failed messages are skipped."""
    ok = [r for r in rows if not r.get("deleted") and r.get("review_result") is not False]
    kids: dict[str, list[dict]] = {}
    for r in ok:
        if r["parent_id"] is not None:
            kids.setdefault(r["parent_id"], []).append(r)
    out = []
    for root in ok:
        if root["role"] != "prompter" or root["parent_id"] is not None or root["lang"] != "en":
            continue
        turns, user = [], root
        while user is not None and len(turns) < max_turns:
            reply = _best_child([c for c in kids.get(user["message_id"], []) if c["role"] == "assistant"])
            if reply is None:
                break
            turns.append({"question": user["text"], "answer": reply["text"]})
            user = _best_child([c for c in kids.get(reply["message_id"], []) if c["role"] == "prompter"])
        if len(turns) >= min_turns:
            out.append({"id": root["message_id"], "turns": turns})
    return sorted(out, key=lambda t: t["id"])


def flatten_threads(threads: list[dict], source: str) -> list[dict]:
    """Threads -> one row per user turn, a thread's turns consecutive and in order: the knowledge GRPO question columns
    (``question, answer, aliases, category``) plus ``thread_id``, ``turn``, ``n_turns``, ``source``, ``id``."""
    rows = []
    for t in threads:
        n = len(t["turns"])
        for k, turn in enumerate(t["turns"]):
            rows.append({
                "id": f"{t['id']}_t{k}",
                "thread_id": t["id"],
                "turn": k,
                "n_turns": n,
                "question": turn["question"],
                "answer": turn["answer"],
                "aliases": "[]",
                "category": "oasst_mt",
                "source": source,
            })  # fmt: skip
    return rows


class LoadInstructThreads(Job):
    """oasst2 English multi-turn threads (``oasst_threads``), every user turn through ``rule_drop`` (one failing turn
    drops the thread), flattened (``flatten_threads``). ``stats.json``: trees, candidate threads, turn-count
    histogram, drops by reason."""

    def __init__(self, *, min_turns: int = MT_MIN_TURNS, max_turns: int = MT_MAX_TURNS):
        self.min_turns, self.max_turns = int(min_turns), int(max_turns)
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from collections import Counter

        from datasets import Dataset, load_dataset

        ds = load_dataset("OpenAssistant/oasst2", split="train")
        threads = oasst_threads(ds, min_turns=self.min_turns, max_turns=self.max_turns)
        drops, kept = Counter(), []
        for t in threads:
            t["turns"] = [{**x, "answer": x["answer"].strip()[:MAX_REF_CHARS]} for x in t["turns"]]
            reason = next((r for x in t["turns"] if (r := rule_drop(x["question"], x["answer"]))), None)
            if reason:
                drops[reason] += 1
            else:
                kept.append(t)
        rows = flatten_threads(kept, "oasst2_mt")
        stats = {
            "candidate_threads": len(threads),
            "kept_threads": len(kept),
            "turns_histogram": dict(Counter(len(t["turns"]) for t in kept)),
            "dropped": dict(drops),
            "rows": len(rows),
        }
        print(f"[instruct-mt] {stats}", flush=True)
        Dataset.from_list(rows).save_to_disk(self.out_hf.get_path())
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=1)


#: The thread keep/rewrite instruction (``InstructThreadPrep``). Prompt text is not hashed: bump ``prompt_version``.
THREAD_PREP_INSTRUCTIONS = (
    "You prepare multi-turn conversations for training a VOICE assistant that talks with a user in real time. Below "
    "is a conversation a person had with a text assistant: several user turns, each with the reply a person wrote.\n"
    "During training, the voice assistant will answer every user turn ITSELF, so its earlier replies will differ "
    "from the written ones. Keep the conversation (keep=true) only if every follow-up user turn still makes sense "
    "after ANY reasonable reply to the turns before it. Reject it (keep=false) if a follow-up refers to specific "
    "content only the written reply contains (for example 'tell me more about the second point', 'why did you say "
    "X', 'translate that'), corrects or complains about the written reply, or changes to an unrelated topic.\n"
    "Also reject it if any turn needs code, maths working, a table, a long list, or a long text (an essay, a full "
    "email or letter, a story longer than a few sentences); refers to a document, link or image that is not given; "
    "depends on today's date, the user's location or personal details, or on recent events; is unsafe, sexual or "
    "hateful; is not in English; or if a written reply is wrong or unhelpful and you cannot tell what a good answer "
    "is.\n"
    "If keep=true, write for EVERY user turn, in order:\n"
    "- question: the turn as the user would say it aloud: natural spoken English, no formatting, numbers and symbols "
    "written as words, at most 40 words. A follow-up may stay short and may refer back to the topic; it must not "
    "rely on details only the written reply contains. Light rewording to achieve that is allowed; keep its intent.\n"
    "- reference: what a good voice assistant would say back to that turn, based on the written reply (fix it if it "
    "is clearly wrong): plain spoken sentences, no lists or formatting, at most 70 words.\n"
    'Respond with JSON only: {"keep": true or false, "reason": "<a few words>", "turns": [{"question": "...", '
    '"reference": "..."}, ...]}  (an empty "turns" list when keep=false; when keep=true exactly one entry per user '
    "turn)."
)


class InstructThreadPrep(Job):
    """One LLM call per thread of ``LoadInstructThreads`` (``THREAD_PREP_INSTRUCTIONS``): keep/drop with a reason and
    every turn's spoken request + spoken-length reference. Output: the kept threads flattened as the input (turns
    consecutive, ``orig_question`` / ``orig_answer`` kept). ``stats.json``: threads in/kept by turn count, drop
    reasons verbatim (counted); ``examples.jsonl``: up to 40 kept and 40 dropped threads, verbatim."""

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

        flat = list(load_from_disk(str(self.in_hf.get())))
        threads: dict[str, list[dict]] = {}
        for r in flat:
            threads.setdefault(r["thread_id"], []).append(r)
        items = [sorted(v, key=lambda r: r["turn"]) for _, v in sorted(threads.items())]
        print(
            f"[instruct-mt-prep] {len(items)} threads, model {self.llm_name}, prompt v{self.prompt_version}", flush=True
        )
        t0 = time.time()
        with vllm_server(self.llm_name, max_model_len=12288) as url:
            client = OpenAI(api_key="EMPTY", base_url=url, timeout=300)

            def call(turns):
                conv = "\n\n".join(
                    f"User turn {k + 1}: {t['question']}\nWritten reply {k + 1}: {t['answer']}"
                    for k, t in enumerate(turns)
                )
                user = f"{conv}\n\n{THREAD_PREP_INSTRUCTIONS}"
                err = ""
                for _ in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=self.llm_name,
                            messages=[{"role": "user", "content": user}],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                            max_tokens=1500,
                        )
                        d = json.loads(resp.choices[0].message.content)
                        keep = d.get("keep") is True
                        out = [
                            (str(x.get("question") or "").strip(), str(x.get("reference") or "").strip())
                            for x in (d.get("turns") or [])
                        ]
                        if keep and (len(out) != len(turns) or not all(q and ref for q, ref in out)):
                            return {"keep": False, "reason": "llm_bad_turns", "turns": []}
                        return {
                            "keep": keep,
                            "reason": str(d.get("reason") or "").strip(),
                            "turns": out if keep else [],
                        }
                    except Exception as e:  # noqa: BLE001 -- retried, then counted as a drop
                        err = repr(e)
                return {"keep": False, "reason": f"llm_fail: {err[:80]}", "turns": []}

            outs = map_concurrent(call, items, concurrency=96, progress_every=500)
        print(f"[instruct-mt-prep] LLM pass {time.time() - t0:.0f} s", flush=True)

        kept_rows, by_n, reasons = [], defaultdict(Counter), Counter()
        examples = {"kept": [], "dropped": []}
        for turns, o in zip(items, outs):
            by_n[str(len(turns))]["kept" if o["keep"] else "dropped"] += 1
            ex = examples["kept" if o["keep"] else "dropped"]
            if len(ex) < 40:
                ex.append({
                    "thread_id": turns[0]["thread_id"],
                    "orig": [{"question": t["question"], "answer": t["answer"][:500]} for t in turns],
                    "keep": o["keep"],
                    "reason": o["reason"],
                    "turns": [{"question": q, "reference": ref} for q, ref in o["turns"]],
                })  # fmt: skip
            if not o["keep"]:
                reasons[o["reason"].lower()[:60]] += 1
                continue
            for t, (q, ref) in zip(turns, o["turns"]):
                kept_rows.append({
                    **{k: t[k] for k in ("id", "thread_id", "turn", "n_turns", "aliases", "category", "source")},
                    "question": q,
                    "answer": ref,
                    "orig_question": t["question"],
                    "orig_answer": t["answer"],
                })  # fmt: skip
        Dataset.from_list(kept_rows).save_to_disk(self.out_hf.get_path())
        summary = {
            "threads_in": len(items),
            "threads_kept": sum(v["kept"] for v in by_n.values()),
            "rows_kept": len(kept_rows),
            "by_turns": {k: dict(v) for k, v in sorted(by_n.items())},
            "top_drop_reasons": dict(reasons.most_common(30)),
        }
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(summary, f, indent=1)
        with open(self.out_examples.get_path(), "w") as f:
            for kind in ("kept", "dropped"):
                for e in examples[kind]:
                    f.write(json.dumps({"kind": kind, **e}) + "\n")
        print(f"[instruct-mt-prep] {summary}", flush=True)
