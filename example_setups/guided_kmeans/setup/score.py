__all__ = ["JiwerScoringJob", "TaggedCorpusToTxtJob", "FrameErrorRateJob"]
import re
from collections import Counter
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
        self.out_confusion_pairs = self.output_path("confusion_pairs.tsv")
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
        # Substituted (reference, hypothesis) label pairs, accumulated over all
        # batches - which label the recognizer confuses for which is the main
        # thing this job is inspected for beyond the plain error rate.
        confusion: Counter = Counter()
        alignment = open(self.out_alignment.get_path(), "w") if self.out_alignment else None
        try:
            for start in range(0, len(common_tags), self.BATCH_SIZE):
                batch = common_tags[start : start + self.BATCH_SIZE]
                ref_sentences = [ref_dict[t] for t in batch]
                hyp_sentences = [hyp_dict[t] for t in batch]
                out = jiwer.process_words(ref_sentences, hyp_sentences)
                for key in counts:
                    counts[key] += getattr(out, key)
                self._collect_confusions(out, ref_sentences, hyp_sentences, confusion)
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

        with open(self.out_confusion_pairs.get_path(), "w") as fp:
            fp.write("ref\thyp\tcount\n")
            for (r, h), count in sorted(confusion.items(), key=lambda x: -x[1]):
                fp.write(f"{r}\t{h}\t{count}\n")

    @staticmethod
    def _collect_confusions(out, ref_sentences: list, hyp_sentences: list, confusion: Counter) -> None:
        """Tally the substituted (reference, hypothesis) label pairs of one jiwer batch."""
        for chunks, ref_sent, hyp_sent in zip(out.alignments, ref_sentences, hyp_sentences):
            ref_words = ref_sent.split()
            hyp_words = hyp_sent.split()
            for chunk in chunks:
                if chunk.type == "substitute":
                    for r, h in zip(
                        ref_words[chunk.ref_start_idx:chunk.ref_end_idx],
                        hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx],
                    ):
                        confusion[(r, h)] += 1


class FrameErrorRateJob(Job):
    def __init__(self, frame_labels: tk.Path, alignment: tk.Path, lexicon: tk.Path):
        self.frame_labels = frame_labels
        self.alignment = alignment
        self.lexicon = lexicon
        self.out_fer = self.output_var("fer")
        self.out_frame_confusion_pairs = self.output_path("frame_confusion_pairs.tsv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import gzip, json, pickle
        import numpy as np
        from xml.etree import ElementTree as ET

        lexicon_path = self.lexicon.get_path()
        open_fn = gzip.open if lexicon_path.endswith(".gz") else open
        with open_fn(lexicon_path, "rb") as f:
            root = ET.parse(f).getroot()
        phonemes = [e.findtext("symbol") for e in root.findall(".//phoneme-inventory/phoneme")]
        phoneme_to_idx = {p: i for i, p in enumerate(phonemes)}

        with open(str(self.alignment), "rb") as f:
            ref_alignment = pickle.load(f)

        hyp_frames = {}
        with open(str(self.frame_labels)) as f:
            for line in f:
                entry = json.loads(line)
                tag = "/".join(entry["seq_tag"].split("/")[-2:])
                hyp_frames[tag] = entry["frames"]

        idx_to_phoneme = {i: p for p, i in phoneme_to_idx.items()}

        total = 0
        errors = 0
        confusion: Counter = Counter()
        for ref_tag, ref in ref_alignment.items():
            short_tag = "/".join(ref_tag.split("/")[-2:])
            if short_tag not in hyp_frames:
                continue
            hyp = np.array([phoneme_to_idx.get(p, -1) for p in hyp_frames[short_tag]])
            n = min(len(ref), len(hyp))
            total += n
            errors += int(np.sum(ref[:n] != hyp[:n]))
            for r_idx, h_idx in zip(ref[:n], hyp[:n]):
                confusion[(idx_to_phoneme.get(int(r_idx), "?"), idx_to_phoneme.get(int(h_idx), "?"))] += 1

        self.out_fer.set(round(errors / total * 100, 2) if total > 0 else float("nan"))

        with open(self.out_frame_confusion_pairs.get_path(), "w") as fp:
            fp.write("ref\thyp\tcount\n")
            for (r, h), count in sorted(confusion.items(), key=lambda x: -x[1]):
                fp.write(f"{r}\t{h}\t{count}\n")


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
