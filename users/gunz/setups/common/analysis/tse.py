from difflib import SequenceMatcher
import logging
import numpy as np
import subprocess
from typing import Iterator, List, Tuple, TypeVar
import sys

from sisyphus import Job, Path, Task

from i6_core.lib.rasr_cache import FileArchiveBundle


NUM_TASKS = 16

T = TypeVar("T")


def cache(path: Path) -> str:
    return subprocess.check_output(["cf", path.get_path()]).decode(sys.stdout.encoding).strip()


def get_mixture_indices(a_cache: FileArchiveBundle, segment: str) -> np.ndarray:
    a_data: List[Tuple[int, int, int, int]] = a_cache.read(segment, "align")
    return np.array(a_data)[:, 1] if len(a_data) > 0 else np.array([])  # return only mixture indices


def compute_begins_ends(mix_indices: np.ndarray, silence_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    def get_boundaries(a: np.ndarray) -> np.ndarray:
        change_mask = np.logical_and(a[1:] != a[:-1], a[1:] != silence_idx)
        boundaries = np.where(change_mask)[0] + 1
        return boundaries

    begins = get_boundaries(mix_indices)
    ends = len(mix_indices) - get_boundaries(mix_indices[::-1])[::-1]

    return begins, ends


class ComputeTimestampErrorJob(Job):
    """
    A job that computes a time stamp error given a reference alignment.

    The reference alignment can be more fine-grained on the timescale
    than the alignment to be tested.
    """

    __sis_hash_exclude__ = {"fuzzy_match_mismatching_phoneme_sequences": False}

    def __init__(
        self,
        *,
        allophones: Path,
        alignment: Path,
        t_step: float,
        reference_allophones: Path,
        reference_alignment: Path,
        reference_t_step: float,
        fuzzy_match_mismatching_phoneme_sequences: bool = True,
    ):
        assert t_step >= reference_t_step > 0

        super().__init__()

        self.allophones = allophones
        self.alignment = alignment
        self.t_step = t_step

        self.reference_allophones = reference_allophones
        self.reference_alignment = reference_alignment
        self.reference_t_step = reference_t_step

        self.fuzzy_match_mismatching_phoneme_sequences = fuzzy_match_mismatching_phoneme_sequences

        self.out_num_processed = self.output_var("num_processed")
        self.out_num_skipped = self.output_var("num_skipped")
        self.out_tse = self.output_var("out_tse")
        self.out_tse_per_seq = self.output_var("out_tse_per_seq")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self) -> Iterator[Task]:
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        alignment = FileArchiveBundle(cache(self.alignment))
        alignment.setAllophones(self.allophones.get_path())

        ref_alignment = FileArchiveBundle(cache(self.reference_alignment))
        ref_alignment.setAllophones(self.reference_allophones.get_path())

        segments = [f for f in alignment.file_list() if not f.endswith(".attribs")]
        self.out_num_processed.set(len(segments))

        s_idx = next(iter(alignment.files.values())).allophones.index("[SILENCE]{#+#}@i@f")
        ref_s_idx = next(iter(ref_alignment.files.values())).allophones.index("[SILENCE]{#+#}@i@f")

        skipped = 0
        total_dist = 0
        total_num = 0
        tse = {}

        for seg in segments:
            mix_indices = get_mixture_indices(alignment, seg)
            mix_indices_ref = get_mixture_indices(ref_alignment, seg)

            if len(mix_indices) == 0 or len(mix_indices_ref) == 0:
                logging.warning(f"empty alignment for {seg}, skipping")
                skipped += 1
                continue

            begins, ends = compute_begins_ends(mix_indices, s_idx)
            begins_ref, ends_ref = compute_begins_ends(mix_indices_ref, ref_s_idx)

            if len(begins) == len(begins_ref) and len(ends) == len(ends_ref):
                data = [(begins, ends, begins_ref, ends_ref)]
            elif self.fuzzy_match_mismatching_phoneme_sequences:
                b_matcher = SequenceMatcher(None, a=begins, b=begins_ref)
                b_matches = [b for b in b_matcher.get_matching_blocks() if b.size > 3]

                e_matcher = SequenceMatcher(None, a=ends, b=ends_ref)
                e_matches = [b for b in e_matcher.get_matching_blocks() if b.size > 3]

                if len(b_matches) != len(e_matches):
                    logging.info(
                        f"Diff between begins and ends gave different number of matching blocks, skipping: {len(b_matches)} vs. {len(e_matches)}"
                    )
                    skipped += 1
                    continue

                data = [
                    (
                        begins[b_match.a : b_match.a + b_match.size],
                        ends[e_match.a : e_match.a + e_match.size],
                        begins_ref[b_match.b : b_match.b + b_match.size],
                        ends_ref[e_match.b : e_match.b + e_match.size],
                    )
                    for b_match, e_match in zip(b_matches, e_matches)
                ]
            else:
                logging.info(
                    f"len mismatch in {seg} of {len(begins)}/{len(ends)} vs. {len(begins_ref)}/{len(ends_ref)}, skipping due to different pronunciation. {len(tse)} alignments already diffed."
                )
                skipped += 1
                continue

            distances = [self._compute_distance(b, e, b_ref, e_ref) for b, e, b_ref, e_ref in data]
            dists, nums = list(zip(*distances))

            total_num += sum(nums)
            total_dist += sum(dists)
            tse[seg] = (sum(dists) / sum(nums)) * self.t_step

        self.out_num_skipped.set(skipped)
        self.out_tse.set((total_dist / total_num) * self.t_step)
        self.out_tse_per_seq.set(tse)

    def _compute_distance(
        self, begins: np.ndarray, ends: np.ndarray, ref_begins: np.ndarray, ref_ends: np.ndarray
    ) -> Tuple[float, int]:
        factor = self.t_step / self.reference_t_step
        begins_err = np.abs(begins * factor - ref_begins)
        ends_err = np.abs(ends * factor - ref_ends)
        distance = np.sum(begins_err) + np.sum(ends_err)
        num_boundaries = len(begins) + len(ends)

        return distance, num_boundaries
