__all__ = ["JiwerScoringJob", "TaggedCorpusToTxtJob"]
import re
from dataclasses import dataclass

import jiwer

from sisyphus import tk, Job, Task

wer_re = re.compile(r"wer=(\d+\.\d+)%")
edits_re = re.compile(r"substitutions=(\d+) deletions=(\d+) insertions=(\d+) hits=(\d+)")


class TaggedCorpusToTxtJob(Job):
    # Write one 'segment_name<tab>text' line per segment from a Bliss corpus

    def __init__(self, corpus: tk.Path):
        self.corpus = corpus
        self.out_txt = self.output_path("ref.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from i6_core.lib.corpus import Corpus
        corpus = Corpus()
        corpus.load(self.corpus.get_path())
        with open(self.out_txt.get_path(), "w") as f:
            for segment in corpus.segments():
                f.write(f"{segment.fullname()}\t{segment.orth}\n")


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

    @staticmethod
    def _read_tagged(path: str) -> dict:
        result = {}
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                tag, text = line.split("\t", 1)
                result[tag] = text
        return result

    def run_alignment(self):
        ref_dict = self._read_tagged(self.ref.get_path())
        hyp_dict = self._read_tagged(self.hyp.get_path())

        common_tags = sorted(set(ref_dict) & set(hyp_dict))

        ref_only = set(ref_dict) - set(hyp_dict)
        hyp_only = set(hyp_dict) - set(ref_dict)
        if ref_only:
            print(f"WARNING: {len(ref_only)} segments in ref but not in hyp")
        if hyp_only:
            print(f"WARNING: {len(hyp_only)} segments in hyp but not in ref")

        # iterate common_tags list in the same order
        ref_sentences = [ref_dict[t] for t in common_tags]
        hyp_sentences = [hyp_dict[t] for t in common_tags]

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
                    total = s + d + h  # reference length: substitutions + deletions + hits
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
