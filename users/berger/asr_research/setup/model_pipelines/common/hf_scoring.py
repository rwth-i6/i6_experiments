"""Updated HF text normalization for STM/CTM scoring with Sclite.

Published ASR leaderboard scores may use an older normalizer; see
:mod:`.hf_normalizer` for the version history and the Granite 2.61% vs 2.67%
example. We deliberately keep the updated normalization rules here.
"""

from difflib import SequenceMatcher
from typing import Iterator

from sisyphus import Job, Task, tk

from .hf_normalizer.normalizer import EnglishTextNormalizer


def normalize_hf_english(text: str) -> str:
    return EnglishTextNormalizer()(text)


def normalize_compound_pair(reference: str, hypothesis: str) -> tuple[str, str]:
    """Apply the leaderboard's compound-boundary equivalence rule."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    new_ref, new_hyp = [], []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, ref_words, hyp_words).get_opcodes():
        if tag == "equal":
            new_ref.extend(ref_words[i1:i2])
            new_hyp.extend(hyp_words[j1:j2])
            continue
        ref_part = "".join(ref_words[i1:i2])
        hyp_part = "".join(hyp_words[j1:j2])
        if ref_part and ref_part == hyp_part:
            new_ref.append(ref_part)
            new_hyp.append(hyp_part)
        else:
            new_ref.extend(ref_words[i1:i2])
            new_hyp.extend(hyp_words[j1:j2])
    return " ".join(new_ref), " ".join(new_hyp)


class NormalizeStmCtmJob(Job):
    """Normalize an STM reference and CTM hypothesis before Sclite scoring."""

    def __init__(self, stm: tk.Path, ctm: tk.Path):
        self.stm = stm
        self.ctm = ctm
        self.out_stm = self.output_path("reference.stm")
        self.out_ctm = self.output_path("hypothesis.ctm")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self) -> None:
        normalizer = EnglishTextNormalizer()
        references = {}
        comments = []
        with open(self.stm.get_path(), encoding="utf-8") as src:
            for line in src:
                if line.startswith(";") or not line.strip():
                    if line.startswith(";"):
                        comments.append(line)
                    continue
                fields = line.rstrip("\n").split(maxsplit=6)
                if len(fields) == 7:
                    references.setdefault(fields[0], []).append((fields[:6], fields[6], float(fields[3]), float(fields[4])))

        grouped = {}
        with open(self.ctm.get_path(), encoding="utf-8") as src:
            for line in src:
                if line.startswith(";") or not line.strip():
                    continue
                fields = line.split()
                if len(fields) >= 5:
                    grouped.setdefault(fields[0], []).append(fields)
        with open(self.out_stm.get_path(), "w", encoding="utf-8") as ref_dst, open(
            self.out_ctm.get_path(), "w", encoding="utf-8"
        ) as hyp_dst:
            ref_dst.writelines(comments)
            for rec in sorted(set(references) | set(grouped)):
                ref_entries = references.get(rec, [])
                rows = grouped.get(rec, [])
                for prefix, raw_ref, seg_start, seg_end in ref_entries:
                    ref_text = normalizer(raw_ref)
                    seg_rows = [
                        row
                        for row in rows
                        if seg_start <= float(row[2]) + 0.5 * float(row[3]) < seg_end
                    ]
                    hyp_text = normalizer(" ".join(row[4] for row in seg_rows))
                    ref_text, hyp_text = normalize_compound_pair(ref_text, hyp_text)
                    ref_dst.write(" ".join(prefix) + f" {ref_text}\n")
                    words = hyp_text.split()
                    if words and seg_rows:
                        start = float(seg_rows[0][2])
                        end = max(float(row[2]) + float(row[3]) for row in seg_rows)
                        duration = (end - start) / len(words)
                        for i, word in enumerate(words):
                            hyp_dst.write(
                                f"{rec} {seg_rows[0][1]} {start + i * duration:.3f} {duration:.3f} {word} 0.99\n"
                            )
