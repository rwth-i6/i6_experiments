"""Broad-spectrum knowledge-RL prompts beyond TriviaQA (user, 2026-10-03/-11; knowledge_broad_rl.md).

Knowledge GRPO's pool is TriviaQA only (pub-quiz facts about named entities). This adds further question categories
with permissive data licences (checked 2026-10-11):

  search   google-research-datasets/nq_open (CC BY-SA 3.0): real Google search queries with short Wikipedia answers.
  science  allenai/ai2_arc ARC-Easy + ARC-Challenge (CC BY-SA 4.0): grade-school science exam questions, asked OPEN
           (the multiple-choice options are dropped, the correct option is the reference).

Training rows come from each set's TRAIN split, held-out rows from its official dev/test splits (NQ-open validation;
ARC validation + test), so the two are disjoint by construction. Pipeline (each step counted, with verbatim examples):

  ``LoadBroadQuestions``   normalise to ``question, answer, aliases (JSON list), category, source, id`` + rule filter
                           (time-dependent NQ questions; ARC questions that need their options, ARC-DA's rules).
  ``VerifyBroadQuestions`` one LLM call per row: self-contained, unambiguous, stable over time and place, answerable
                           without options/images/context, reference plausible -> keep / drop + reason.
  ``SplitBroadQuestions``  decontaminate (normalised text and content-word near-duplicates) against the given
                           exclude sets (TriviaQA validation, MRQA's four sets, the TriviaQA RL pool) and across the
                           held-out/train boundary, then cap each split (seeded).

Then the knowledge benchmark's ``LLMPreprocess`` (spoken form) and ``ChatterboxSingleSpeakerInference`` per source;
``RLFinetune(knowledge_sources=...)`` takes the per-source (questions, audio) pairs labelled by category.
"""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk

from .common import map_concurrent, vllm_gpu_mem_gb

BROAD_SOURCES = {"nq_open": "search", "arc": "science"}
_WORD = re.compile(r"[a-z0-9']+")
# NQ-open: AmbigNQ found 13% of its ambiguity is time-dependence; these words mark questions whose answer moves
_TIME_RE = re.compile(
    r"\b(current|currently|now|today|latest|recent|recently|this year|last year|next year|upcoming|newest|so far|"
    r"to date|still|as of)\b",
    re.IGNORECASE,
)
# ARC: ARC-DA (Bhakthavatsalam et al. 2021) dropped questions that are ill-formed without their options
_OPTION_RE = re.compile(
    r"\b(which of|of the following|the following|which statement|which one|which choice|which option|"
    r"best|least|these)\b|_{2,}",
    re.IGNORECASE,
)
_STOP = set(
    "a an the of in on at to for from by with and or is are was were be been do does did what which who whom whose "
    "where when why how that this these those it its as into than then there their they he she his her you your "
    "can could would should will many much".split()
)


def norm(text: str) -> str:
    return " ".join(_WORD.findall(str(text).lower()))


def content_words(text: str) -> frozenset:
    return frozenset(w for w in _WORD.findall(str(text).lower()) if w not in _STOP)


def rule_drop(source: str, question: str, answer: str) -> str | None:
    """-> the drop reason, or None to keep for the LLM pass."""
    n = len(_WORD.findall(question.lower()))
    if not answer.strip():
        return "no_answer"
    if not 3 <= n <= 40:
        return "question_length"
    if source == "nq_open" and _TIME_RE.search(question):
        return "time_dependent"
    if source == "arc" and _OPTION_RE.search(question):
        return "needs_options"
    if source == "arc" and not question.rstrip().endswith("?"):  # a completion stem ("... the switch can")
        return "not_a_question"
    return None


def _arc_rows(split: str):
    from datasets import load_dataset

    for config in ("ARC-Easy", "ARC-Challenge"):
        for r in load_dataset("allenai/ai2_arc", config, split=split):
            labels, texts = r["choices"]["label"], r["choices"]["text"]
            if r["answerKey"] not in labels:
                yield {"id": r["id"], "question": r["question"], "answer": "", "aliases": [], "config": config}
                continue
            ans = texts[labels.index(r["answerKey"])]
            yield {"id": r["id"], "question": r["question"], "answer": ans, "aliases": [ans], "config": config}


def _nq_rows(split: str):
    from datasets import load_dataset

    for i, r in enumerate(load_dataset("google-research-datasets/nq_open", split=split)):
        ans = list(r["answer"])
        yield {"id": f"nq_{split}_{i}", "question": r["question"], "answer": ans[0] if ans else "", "aliases": ans}


class LoadBroadQuestions(Job):
    """One source's ``splits``, normalised and rule-filtered, then (``n_sample``) a seeded sample of the survivors.
    ``stats.json``: candidates, drops per reason, kept; ``examples.jsonl``: up to 20 verbatim rows per drop reason."""

    def __init__(self, *, source: str, splits: tuple, n_sample: int | None = None, seed: int = 0):
        assert source in BROAD_SOURCES, source
        self.source = source
        self.splits = tuple(splits)
        self.n_sample = n_sample
        self.seed = seed
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.out_examples = self.output_path("examples.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from collections import Counter, defaultdict

        import numpy as np
        from datasets import Dataset

        rows_fn = {"nq_open": _nq_rows, "arc": _arc_rows}[self.source]
        kept, drops, examples, n_in = [], Counter(), defaultdict(list), 0
        for split in self.splits:
            for r in rows_fn(split):
                n_in += 1
                reason = rule_drop(self.source, r["question"], r["answer"])
                if reason:
                    drops[reason] += 1
                    if len(examples[reason]) < 20:
                        examples[reason].append({"question": r["question"], "answer": r["answer"]})
                    continue
                kept.append({
                    "question": r["question"].strip(),
                    "answer": r["answer"].strip(),
                    "aliases": json.dumps(r["aliases"]),
                    "category": BROAD_SOURCES[self.source],
                    "source": self.source,
                    "id": r["id"],
                })  # fmt: skip
        n_rule = len(kept)
        if self.n_sample is not None and len(kept) > self.n_sample:
            idx = np.random.default_rng(self.seed).permutation(len(kept))[: self.n_sample]
            kept = [kept[i] for i in sorted(idx)]
        stats = {
            "source": self.source,
            "splits": list(self.splits),
            "candidates": n_in,
            "dropped": dict(drops),
            "kept_after_rules": n_rule,
            "sampled": len(kept),
        }
        print(f"[broad-load] {stats}", flush=True)
        Dataset.from_list(kept).save_to_disk(self.out_hf.get_path())
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=1)
        with open(self.out_examples.get_path(), "w") as f:
            for reason, ex in examples.items():
                for e in ex:
                    f.write(json.dumps({"reason": reason, **e}) + "\n")


#: The verifier's instruction (``VerifyBroadQuestions``). Prompt text is not hashed: bump ``prompt_version`` with it.
VERIFY_INSTRUCTIONS = (
    "You check questions for training a VOICE assistant to answer factual questions out loud. Below is a question and "
    "its reference answer from an existing dataset. The question will be read aloud to the assistant on its own: no "
    "answer options, no images, no passage, no earlier conversation.\n"
    "keep=true only if ALL hold:\n"
    "- the question is self-contained and makes sense spoken on its own (it does not refer to options, a figure, a "
    "table, a passage or 'the following');\n"
    "- it has one clear answer: not ambiguous, not a matter of opinion;\n"
    "- the answer does not depend on when or where it is asked (no 'current' office holders, latest records, recent "
    "events, or the listener's location);\n"
    "- the reference answer is plausibly correct for the question as written;\n"
    "- it is in English and is not unsafe, sexual or hateful.\n"
    'Respond with JSON only: {"keep": true or false, "reason": "<a few words>"}'
)


class VerifyBroadQuestions(Job):
    """One LLM call per row (``VERIFY_INSTRUCTIONS``): keep / drop with a reason. ``stats.json``: kept/dropped, top drop
    reasons (verbatim LLM strings); ``examples.jsonl``: 30 kept and 30 dropped rows, verbatim."""

    def __init__(self, *, in_hf: tk.Path, llm_name: str = "google/gemma-4-31B-it", prompt_version: int = 1):
        self.in_hf = in_hf
        self.llm_name = llm_name
        self.prompt_version = prompt_version
        self.out_hf = self.output_path("out_hf", directory=True)
        self.out_stats = self.output_path("stats.json")
        self.out_examples = self.output_path("examples.jsonl")
        self.rqmt = {"gpu": 1, "cpu": 8, "mem": 32, "time": 3, "gpu_mem_gb": vllm_gpu_mem_gb(llm_name)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import time
        from collections import Counter

        from datasets import Dataset, load_from_disk
        from openai import OpenAI

        from .common import vllm_server

        rows = list(load_from_disk(str(self.in_hf.get())))
        print(f"[broad-verify] {len(rows)} rows, model {self.llm_name}, prompt v{self.prompt_version}", flush=True)
        t0 = time.time()
        # 12288: LLMPreprocess's measured fit for gemma-4-31B on an 80 GB card; these prompts are < 1k tokens
        with vllm_server(self.llm_name, max_model_len=12288) as url:
            client = OpenAI(api_key="EMPTY", base_url=url, timeout=300)

            def call(r):
                user = f"Question: {r['question']}\nReference answer: {r['answer']}\n\n{VERIFY_INSTRUCTIONS}"
                err = ""
                for _ in range(3):
                    try:
                        resp = client.chat.completions.create(
                            model=self.llm_name,
                            messages=[{"role": "user", "content": user}],
                            response_format={"type": "json_object"},
                            temperature=0.0,
                            max_tokens=100,
                        )
                        d = json.loads(resp.choices[0].message.content)
                        return {"keep": d.get("keep") is True, "reason": str(d.get("reason") or "").strip()}
                    except Exception as e:  # noqa: BLE001 -- retried, then counted as a drop
                        err = repr(e)
                return {"keep": False, "reason": f"llm_fail: {err[:80]}"}

            outs = map_concurrent(call, rows, concurrency=96, progress_every=1000)
        print(f"[broad-verify] LLM pass {time.time() - t0:.0f} s", flush=True)
        kept, reasons, ex = [], Counter(), {"kept": [], "dropped": []}
        for r, o in zip(rows, outs):
            kind = "kept" if o["keep"] else "dropped"
            if len(ex[kind]) < 30:
                ex[kind].append({"id": r["id"], "question": r["question"], "answer": r["answer"], **o})
            if o["keep"]:
                kept.append(r)
            else:
                reasons[o["reason"].lower()[:60]] += 1
        Dataset.from_list(kept).save_to_disk(self.out_hf.get_path())
        stats = {"rows_in": len(rows), "kept": len(kept), "top_drop_reasons": dict(reasons.most_common(30))}
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(stats, f, indent=1)
        with open(self.out_examples.get_path(), "w") as f:
            for kind, es in ex.items():
                for e in es:
                    f.write(json.dumps({"kind": kind, **e}) + "\n")
        print(f"[broad-verify] kept {len(kept)} / {len(rows)}", flush=True)


def is_near_duplicate(words: frozenset, index: dict, sets: list, *, jaccard: float = 0.8) -> int | None:
    """-> the index into ``sets`` of a content-word set with Jaccard >= ``jaccard`` to ``words``, or None. ``index``:
    word -> ids of the sets containing it. Prefix filter: a set at Jaccard >= t with ``words`` must contain at least one
    of ``words``' ``len - ceil(t * len) + 1`` rarest words, so only those words' sets are candidates."""
    import math

    if not words:
        return None
    ordered = sorted(words, key=lambda w: (len(index.get(w, ())), w))
    cand = set()
    for w in ordered[: len(ordered) - math.ceil(jaccard * len(ordered)) + 1]:
        cand |= index.get(w, set())
    for j in cand:
        other = sets[j]
        if len(words & other) / len(words | other) >= jaccard:
            return j
    return None


class SplitBroadQuestions(Job):
    """Decontaminate and cap. ``heldout`` / ``train``: ``{source: verified out_hf}``; ``exclude``: HF datasets with a
    ``question`` column that no kept row may duplicate (normalised text, or content words at Jaccard >= 0.8). Held-out
    rows are decontaminated against ``exclude``; train rows against ``exclude`` AND every held-out row. Then seeded caps
    (``n_heldout`` per source; ``n_train[source]``, None = all). Outputs ``heldout`` (all sources, ``category`` column)
    and ``train_<source>``; ``stats.json`` (drops per source and reason), ``examples.jsonl`` (verbatim duplicates)."""

    def __init__(
        self,
        *,
        heldout: dict,
        train: dict,
        exclude: list,
        n_heldout: int = 200,
        n_train: dict | None = None,
        seed: int = 0,
    ):
        assert set(heldout) == set(train) and set(train) <= set(BROAD_SOURCES), (sorted(heldout), sorted(train))
        self.heldout = dict(heldout)
        self.train = dict(train)
        self.exclude = list(exclude)
        self.n_heldout = n_heldout
        self.n_train = dict(n_train or {})
        self.seed = seed
        self.out_heldout = self.output_path("heldout", directory=True)
        self.out_train = {s: self.output_path(f"train_{s}", directory=True) for s in sorted(train)}
        self.out_stats = self.output_path("stats.json")
        self.out_examples = self.output_path("examples.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from collections import Counter, defaultdict

        import numpy as np
        from datasets import Dataset, load_from_disk

        texts, sets, index = set(), [], defaultdict(set)

        def add(q):
            texts.add(norm(q))
            w = content_words(q)
            sets.append(w)
            for x in w:
                index[x].add(len(sets) - 1)

        exclude_qs = []
        for p in self.exclude:
            exclude_qs += list(load_from_disk(str(p.get() if hasattr(p, "get") else p))["question"])
        for q in exclude_qs:
            add(q)
        stats, examples = defaultdict(Counter), []

        def clean(rows, label):
            out = []
            for r in rows:
                q = r["question"]
                hit = None
                if norm(q) in texts:
                    hit = "exact"
                else:
                    j = is_near_duplicate(content_words(q), index, sets)
                    if j is not None:
                        hit = "near"
                if hit:
                    stats[label][f"contaminated_{hit}"] += 1
                    if len(examples) < 60:
                        examples.append({"split": label, "kind": hit, "question": q})
                    continue
                out.append(r)
            return out

        rng = np.random.default_rng(self.seed)
        held_all = []
        for s in sorted(self.heldout):
            rows = clean(list(load_from_disk(str(self.heldout[s].get()))), f"heldout/{s}")
            rows = [rows[i] for i in rng.permutation(len(rows))[: self.n_heldout]]
            stats[f"heldout/{s}"]["kept"] = len(rows)
            held_all += rows
        for r in held_all:  # train must not duplicate any held-out row (any source)
            add(r["question"])
        Dataset.from_list(held_all).save_to_disk(self.out_heldout.get_path())
        for s in sorted(self.train):
            rows = clean(list(load_from_disk(str(self.train[s].get()))), f"train/{s}")
            seen, uniq = set(), []
            for r in rows:  # within-split duplicates
                k = norm(r["question"])
                if k in seen:
                    stats[f"train/{s}"]["duplicate_within"] += 1
                    continue
                seen.add(k)
                uniq.append(r)
            cap = self.n_train.get(s)
            if cap is not None and len(uniq) > cap:
                uniq = [uniq[i] for i in sorted(rng.permutation(len(uniq))[:cap])]
            stats[f"train/{s}"]["kept"] = len(uniq)
            Dataset.from_list(uniq).save_to_disk(self.out_train[s].get_path())
        summary = {"exclude_questions": len(exclude_qs), **{k: dict(v) for k, v in sorted(stats.items())}}
        print(f"[broad-split] {summary}", flush=True)
        with open(self.out_stats.get_path(), "w") as f:
            json.dump(summary, f, indent=1)
        with open(self.out_examples.get_path(), "w") as f:
            for e in examples:
                f.write(json.dumps(e) + "\n")
