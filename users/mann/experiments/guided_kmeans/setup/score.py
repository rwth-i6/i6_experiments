__all__ = ["JiwerScoringJob"]
import re
from dataclasses import dataclass

import jiwer

from sisyphus import tk, Job, Task

wer_re = re.compile(r"wer=(\d+\.\d+)%")
edits_re = re.compile(r"substitutions=(\d+) deletions=(\d+) insertions=(\d+) hits=(\d+)")

class JiwerScoringJob(Job):
    def __init__(self, ref: tk.Path, hyp: tk.Path):
        self.ref = ref
        self.hyp = hyp
    
        self.out_alignment = self.output_path("alignment.txt")
        self.out_wer = self.output_var("wer")
        self.out_substitutions = self.output_var("substitutions")
        self.out_deletions = self.output_var("deletions")
        self.out_insertions = self.output_var("insertions")
    
    def tasks(self):
        yield Task("run_alignment")
        yield Task("summary")
    
    def run_alignment(self):
        with open(self.ref.get_path(), "r") as ref_fp:
            ref_sentences = [line.strip("\n") for line in ref_fp]

        with open(self.hyp.get_path(), "r") as hyp_fp:
            hyp_sentences = [line.strip("\n") for line in hyp_fp]
        
        out = jiwer.process_words(ref_sentences, hyp_sentences)

        with open(self.out_alignment.get_path(), "w+") as fp:
            fp.write(jiwer.visualize_alignment(out))

    def summary(self):
        with open(self.out_alignment.get_path()) as fp:
            for line in fp:
                m1 = wer_re.match(line)
                m2 = edits_re.match(line)

                if not m1 and not m2:
                    continue

                if m1 is not None:
                    wer_raw = m1.group(1)
                    wer = float(wer_raw)
                    self.out_wer.set(wer)
                    continue

                if m2 is not None:
                    s, d, i, h = [int(m2.group(k)) for k in range(1, 5)]
                    total = s + d + i + h
                    self.out_substitutions.set(s / total)
                    self.out_deletions.set(d / total)
                    self.out_insertions.set(i / total)
                    continue

@dataclass
class ScoreResult:
    score_job: Job
    wer: tk.Variable
    substitutions: tk.Variable
    deletions: tk.Variable
    insertions: tk.Variable

    @classmethod
    def from_job(cls, score_job):
        return cls(
            score_job,
            score_job.out_wer,
            score_job.out_substitutions,
            score_job.out_deletions,
            score_job.out_insertions,
        )
