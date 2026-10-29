"""Rung B's question file from a user-text head's own transcripts (2026-10-18, the CTC-head variant of the gap ladder)."""

from __future__ import annotations

import hashlib
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


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class HeadTranscriptQuestions(Job):
    """``offline_inference --question_context`` rows (``{"index", "question"}``, as ``questions``) with each question
    replaced by the head's transcript of its clip: ``hypotheses`` = ``analysis/ppx_user_text_head_eval.py``'s
    ``per_unit.jsonl`` (rows ``step`` / ``pop`` / ``unit`` "q<index>" / ``hyp``), read at ``step`` for ``pop``.
    ``dedup``: :func:`dedup_repeats` on every transcript. Every index of ``questions`` must have a transcript; an empty
    one stays empty (the prompt then names no question). ``out_stats``: counts + examples (gold vs transcript).
    ``hypotheses_sha256``: the file's content hash, since a script (not a Job) writes it in place; ``run`` checks it,
    and it is hashed unless it is the content the first job read (``HYPOTHESES_PINNED_SHA256``)."""

    HYPOTHESES_PINNED_SHA256 = "2ed2d1942f3c5df2fc8762423255c75435acf6b4b1fbebd74c93a842906ba654"

    def __init__(
        self,
        *,
        hypotheses: tk.Path,
        questions: tk.Path,
        step: int,
        pop: str = "knowledge",
        dedup: bool = True,
        hypotheses_sha256: str | None = None,
    ):
        self.hypotheses = hypotheses
        self.hypotheses_sha256 = hypotheses_sha256
        self.questions = questions
        self.step = step
        self.pop = pop
        self.dedup = dedup
        self.out_questions = self.output_path("questions.jsonl")
        self.out_stats = self.output_path("stats.json")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(parsed_args)
        if d.get("hypotheses_sha256") in (None, cls.HYPOTHESES_PINNED_SHA256):
            d.pop("hypotheses_sha256", None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        if self.hypotheses_sha256 is not None:
            got = file_sha256(self.hypotheses.get_path())
            if got != self.hypotheses_sha256:
                raise RuntimeError(f"{self.hypotheses}: sha256 {got} != declared {self.hypotheses_sha256}")
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
