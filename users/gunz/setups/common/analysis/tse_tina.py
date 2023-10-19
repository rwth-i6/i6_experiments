import logging

import numpy as np
from typing import Dict, List, Iterator, Tuple

from i6_core.lib.rasr_cache import FileArchiveBundle
from sisyphus import Job, Path, Task

from .tse import cache


def get_word_boundaries(allos: List[str]) -> Tuple[List[int], List[int]]:
    word_starts = []
    word_ends = []
    for t, allo in enumerate(allos):
        is_sil = "[SILENCE]" in allo

        if not is_sil and "@i" in allo and (t == 0 or allos[t - 1] != allo):
            word_starts.append(t)
        if not is_sil and "@f" in allo and (t == len(allos) - 1 or allos[t + 1] != allo):
            word_ends.append(t)

    return word_starts, word_ends


def compute_tse(
    ref_word_bounds: Dict[str, Tuple[List[int], List[int], int]], alignments: FileArchiveBundle, ss_factor: int
):
    distances = []
    len_mismatches = 0

    for seq_tag, (ref_word_starts, ref_word_ends, ref_len) in ref_word_bounds.items():
        align = alignments.read(seq_tag, "align")
        if len(align) > 0:
            allos = [alignments.files[seq_tag].allophones[item[1]] for item in align]
            if ss_factor > 1:
                allos = sum([[allo] * ss_factor for allo in allos], [])

            word_starts, word_ends = get_word_boundaries(allos)
            if word_ends:
                word_ends[-1] = min(word_ends[-1], ref_len - 1)
            distances += [abs(start_1 - start_2) for start_1, start_2 in zip(word_starts, ref_word_starts)]
            distances += [abs(end_1 - end_2) for end_1, end_2 in zip(word_ends, ref_word_ends)]

            if len(word_starts) != len(ref_word_starts) or len(word_ends) != len(ref_word_ends):
                len_mismatches += 1

    return np.mean(distances), len_mismatches


def compute_ref_word_bounds(ref_alignments: FileArchiveBundle) -> Dict[str, Tuple[List[int], List[int], int]]:
    ref_word_bounds = {}

    for seq_tag in ref_alignments.file_list():
        if seq_tag.endswith(".attribs"):
            continue
        align = ref_alignments.read(seq_tag, "align")
        allos = [ref_alignments.files[seq_tag].allophones[item[1]] for item in align]
        word_starts, word_ends = get_word_boundaries(allos)
        ref_word_bounds[seq_tag] = (word_starts, word_ends, len(align))

    return ref_word_bounds


class ComputeTinaTseJob(Job):
    def __init__(
        self,
        ref_alignment_bundle: Path,
        ref_allophones: Path,
        ref_t_step: float,
        alignment_bundle: Path,
        allophones: Path,
        ss_factor: int,
    ):
        assert ref_t_step > 0
        assert ss_factor > 0

        self.ref_alignment_bundle = ref_alignment_bundle
        self.ref_allophones = ref_allophones
        self.ref_t_step = ref_t_step

        self.alignment_bundle = alignment_bundle
        self.allophones = allophones

        self.ss_factor = ss_factor

        self.out_tse = self.output_var("tse")
        self.out_num_len_mismatches = self.output_var("num_len_mismatches")

        self.rqmt = {"cpu": 1, "time": 4, "mem": 8}

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        ref_alignments = FileArchiveBundle(cache(self.ref_alignment_bundle))
        ref_alignments.setAllophones(self.ref_allophones.get_path())
        logging.info("computing ref timestamps")
        ref_word_bounds = compute_ref_word_bounds(ref_alignments)

        alignments = FileArchiveBundle(cache(self.alignment_bundle))
        alignments.setAllophones(self.allophones.get_path())
        logging.info("computing TSE")
        tse, num_len_mismatches = compute_tse(ref_word_bounds, alignments, self.ss_factor)

        self.out_tse.set(tse * self.ref_t_step)
        self.out_num_len_mismatches.set(num_len_mismatches)
