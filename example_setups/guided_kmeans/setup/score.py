__all__ = ["JiwerScoringJob", "TaggedCorpusToTxtJob"]
import re
from dataclasses import dataclass
from typing import TextIO, cast

from i6_core.lib.corpus import Corpus
from i6_core.util import uopen

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
        corpus = Corpus()
        corpus.load(self.corpus.get_path())
        with open(self.out_txt.get_path(), "w") as f:
            for segment in corpus.segments():
                f.write(f"{segment.fullname()}\t{segment.orth}\n")


class JiwerScoringJob(Job):
    """
    Score a tagged hypothesis file against a tagged reference with jiwer.

    Both files are ``tag<tab>text`` per line and may be gzipped. Only the tags
    present in both are scored; the rest are reported as a warning.

    :param ref: reference, e.g. ``TaggedCorpusToTxtJob.out_txt``
    :param hyp: hypotheses, e.g. a decode's ``hyp.txt`` or a chunked clustering
        epoch's ``out_hypotheses``
    :param write_alignment: write the per-sentence alignment visualization. Set
        this to False for full-corpus hypotheses, where the visualization runs
        to hundreds of megabytes of text nobody reads. Hash-excluded at its
        default, so existing jobs keep their hash.
    """

    __sis_hash_exclude__ = {"write_alignment": True}

    #: Sentences per jiwer call. The edit counts are additive over batches, so
    #: this only bounds how many alignments are held in memory at once - which
    #: for a full corpus (~28 k sequences of a few hundred labels) is what
    #: decides whether the job fits in its memory requirement.
    BATCH_SIZE = 2000

    def __init__(self, ref: tk.Path, hyp: tk.Path, write_alignment: bool = True):
        self.ref = ref
        self.hyp = hyp
        self.write_alignment = write_alignment

        # Declared only when it is actually going to be written: an output that
        # never appears would leave any consumer of it with a missing file
        # (sisyphus marks the job finished on task completion, not on outputs).
        self.out_alignment = self.output_path("alignment.txt") if write_alignment else None
        self.out_wer = self.output_var("wer")
        self.out_substitutions = self.output_var("substitutions")
        self.out_deletions = self.output_var("deletions")
        self.out_insertions = self.output_var("insertions")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 1})

    @staticmethod
    def _read_tagged(path: str) -> dict:
        result = {}
        # uopen picks gzip by extension; it is annotated as returning a binary
        # IOBase, hence the cast for a text-mode read.
        with cast(TextIO, uopen(path, "rt")) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                tag, text = line.split("\t", 1)
                result[tag] = text
        return result

    def run(self):
        ref_dict = self._read_tagged(self.ref.get_path())
        hyp_dict = self._read_tagged(self.hyp.get_path())

        common_tags = sorted(set(ref_dict) & set(hyp_dict))

        ref_only = set(ref_dict) - set(hyp_dict)
        hyp_only = set(hyp_dict) - set(ref_dict)
        if ref_only:
            print(f"WARNING: {len(ref_only)} segments in ref but not in hyp")
        if hyp_only:
            print(f"WARNING: {len(hyp_only)} segments in hyp but not in ref")
        if not common_tags:
            raise ValueError("no segment tag occurs in both the reference and the hypotheses")

        counts = {"substitutions": 0, "deletions": 0, "insertions": 0, "hits": 0}
        alignment = open(self.out_alignment.get_path(), "w") if self.out_alignment else None
        try:
            for start in range(0, len(common_tags), self.BATCH_SIZE):
                batch = common_tags[start : start + self.BATCH_SIZE]
                out = jiwer.process_words(
                    [ref_dict[t] for t in batch], [hyp_dict[t] for t in batch]
                )
                for key in counts:
                    counts[key] += getattr(out, key)
                if alignment is not None:
                    # Measures per batch would be misleading; the summary of
                    # the whole file is appended once, below.
                    alignment.write(jiwer.visualize_alignment(out, show_measures=False))

            s, d, i, h = (counts[k] for k in ("substitutions", "deletions", "insertions", "hits"))
            total = s + d + h  # reference length
            if total == 0:
                raise ValueError(
                    f"the {len(common_tags)} scored references are all empty, "
                    "so there is no error rate to report"
                )
            # Taken from the counts rather than parsed back out of the
            # visualization, as this job used to do - same numbers, but at full
            # precision instead of the two decimals jiwer prints, and it works
            # when no visualization is written at all.
            self.out_wer.set(100.0 * (s + d + i) / total)
            self.out_substitutions.set(s / total)
            self.out_deletions.set(d / total)
            self.out_insertions.set(i / total)

            if alignment is not None:
                # Same shape as jiwer's own summary block, so wer_re/edits_re
                # still parse these files.
                alignment.write(
                    f"\n=== SUMMARY ===\nnumber of sentences: {len(common_tags)}\n"
                    f"substitutions={s} deletions={d} insertions={i} hits={h}\n\n"
                    f"wer={100.0 * (s + d + i) / total:.2f}%\n"
                )
        finally:
            if alignment is not None:
                alignment.close()

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
