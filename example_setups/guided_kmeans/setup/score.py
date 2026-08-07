__all__ = ["JiwerScoringJob", "TaggedCorpusToTxtJob", "FrameErrorRateJob"]
import re
from collections import Counter
from dataclasses import dataclass
from i6_core.lib.corpus import Corpus

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
    def __init__(self, ref: tk.Path, hyp: tk.Path):
        self.ref = ref
        self.hyp = hyp
    
        self.out_alignment = self.output_path("alignment.txt")
        self.out_confusion_pairs = self.output_path("confusion_pairs.tsv")
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

        confusion: Counter = Counter()
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

        with open(self.out_confusion_pairs.get_path(), "w") as fp:
            fp.write("ref\thyp\tcount\n")
            for (r, h), count in sorted(confusion.items(), key=lambda x: -x[1]):
                fp.write(f"{r}\t{h}\t{count}\n")

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
