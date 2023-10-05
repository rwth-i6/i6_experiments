import itertools
import logging
import numpy as np
import subprocess
from typing import Any, Dict, Iterator, List, Tuple, TypeVar, Union
import sys

from sisyphus import Job, Path, Task

from i6_core.lib.rasr_cache import FileArchiveBundle


NUM_TASKS = 16

T = TypeVar("T")


def cache(path: Path) -> str:
    return subprocess.check_output(["cf", path.get_path()]).decode(sys.stdout.encoding).strip()


def get_mixture_indices(a_cache: FileArchiveBundle, segment: str) -> np.ndarray:
    a_data: List[Tuple[int, int, int, int]] = a_cache.read(segment, "align")
    return np.array(a_data)[:, 1]  # return only mixture indices


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

    def __init__(
        self,
        *,
        allophones: Path,
        alignment: Path,
        t_step: float,
        reference_allophones: Path,
        reference_alignment: Path,
        reference_t_step: float,
    ):
        assert t_step >= reference_t_step > 0

        super().__init__()

        self.allophones = allophones
        self.alignment = alignment
        self.t_step = t_step

        self.reference_allophones = reference_allophones
        self.reference_alignment = reference_alignment
        self.reference_t_step = reference_t_step

        self.out_num_processed = self.output_var("num_processed")
        self.out_num_skipped = self.output_var("num_skipped")
        self.out_tse = self.output_var("out_tse")
        self.out_tse_per_seq = self.output_var("out_tse_per_seq")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt=self.rqmt)

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

            # Debug logging
            # logging.info(begins)
            # logging.info(ends)
            # logging.info(begins_ref)
            # logging.info(ends_ref)

            # quick escape hatch for phoneme sequence (in)equality, avoids having to fetch the allophones
            if len(begins) != len(begins_ref):
                logging.info(
                    f"len mismatch in {seg} of {len(begins)} vs. {len(begins_ref)}, skipping due to different pronunciation. {len(tse)} alignments already diffed."
                )
                skipped += 1
                continue

            dist, num = self._compute_distance(begins, ends, begins_ref, ends_ref)

            total_num += num
            total_dist += dist
            tse[seg] = (dist / num) * self.t_step

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
