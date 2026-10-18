"""Rung B's question file from a user-text head's own transcripts (2026-10-18, the CTC-head variant of the gap ladder)."""

from __future__ import annotations

import json
import re

from sisyphus import Job, Task, tk


def dedup_repeats(text: str) -> str:
    """Collapse consecutive repeated words (compared lowercase, without punctuation; the first spelling is kept) and
    runs of the same punctuation mark: a framewise head's stutter ("is is a a student in in which year year") read
    offline. Real repeats ("New New York" does not occur in TriviaQA) would be lost too."""
    out, prev = [], None
    for w in text.split():
        key = re.sub(r"[^\w']", "", w.lower())
        if key and key == prev:
            continue
        out.append(w)
        prev = key or prev
    return re.sub(r"([?!.,])\1+", r"\1", " ".join(out))


class HeadTranscriptQuestions(Job):
    """``offline_inference --question_context`` rows (``{"index", "question"}``, as ``questions``) with each question
    replaced by the head's transcript of its clip: ``hypotheses`` = ``analysis/ppx_user_text_head_eval.py``'s
    ``per_unit.jsonl`` (rows ``step`` / ``pop`` / ``unit`` "q<index>" / ``hyp``), read at ``step`` for ``pop``.
    ``dedup``: :func:`dedup_repeats` on every transcript. Every index of ``questions`` must have a transcript; an empty
    one stays empty (the prompt then names no question). ``out_stats``: counts + examples (gold vs transcript)."""

    def __init__(
        self, *, hypotheses: tk.Path, questions: tk.Path, step: int, pop: str = "knowledge", dedup: bool = True
    ):
        self.hypotheses = hypotheses
        self.questions = questions
        self.step = step
        self.pop = pop
        self.dedup = dedup
        self.out_questions = self.output_path("questions.jsonl")
        self.out_stats = self.output_path("stats.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        hyp = {}
        for r in map(json.loads, open(self.hypotheses.get_path())):
            if r["step"] == self.step and r["pop"] == self.pop:
                assert r["unit"].startswith("q"), r["unit"]
                hyp[r["unit"][1:]] = r["hyp"] or ""
        gold = [json.loads(line) for line in open(self.questions.get_path())]
        missing = [str(r["index"]) for r in gold if str(r["index"]) not in hyp]
        assert not missing, f"no transcript at step {self.step} for {len(missing)} questions, e.g. {missing[:3]}"
        rows, examples = [], []
        for r in gold:
            h = hyp[str(r["index"])]
            q = dedup_repeats(h) if self.dedup else h
            rows.append({"index": r["index"], "question": q})
            if len(examples) < 10:
                examples.append({"index": r["index"], "gold": r["question"], "raw": h, "question": q})
        with open(self.out_questions.get_path(), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        n_words = [len(r["question"].split()) for r in rows]
        stats = {
            "n": len(rows),
            "n_empty": sum(1 for n in n_words if n == 0),
            "words_mean": sum(n_words) / len(rows),
            "gold_words_mean": sum(len(r["question"].split()) for r in gold) / len(gold),
            "raw_words_mean": sum(len(hyp[str(r["index"])].split()) for r in gold) / len(gold),
            "step": self.step,
            "dedup": self.dedup,
            "examples": examples,
        }
        json.dump(stats, open(self.out_stats.get_path(), "w"), indent=2)
        print(json.dumps({k: v for k, v in stats.items() if k != "examples"}))
