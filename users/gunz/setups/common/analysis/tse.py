import itertools
import logging
import numpy as np
import subprocess
from typing import Dict, Iterator, List, Tuple
import sys

from sisyphus import Job, Path, Task

from i6_core.lib.rasr_cache import FileArchiveBundle
from i6_core.util import chunks


NUM_TASKS = 300


def cache(path: Path) -> str:
    return subprocess.check_output(["cf", path.get_path()]).decode(sys.stdout.encoding).strip()


def compute_begins_ends(durations: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(durations)
    begins = ends - durations
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
        n_states_per_phone: int,
        t_step: float,
        reference_allophones: Path,
        reference_alignment: Path,
        reference_n_states_per_phone: int,
        reference_t_step: float,
    ):
        assert (reference_n_states_per_phone == n_states_per_phone) or (
            reference_n_states_per_phone >= n_states_per_phone == 1
        )
        assert t_step >= reference_t_step > 0

        super().__init__()

        self.allophones = allophones
        self.alignment = alignment
        self.n_states_per_phone = n_states_per_phone
        self.t_step = t_step

        self.reference_allophones = reference_allophones
        self.reference_alignment = reference_alignment
        self.reference_n_states_per_phone = reference_n_states_per_phone
        self.reference_t_step = reference_t_step

        self.out_tse = self.output_var("merged.tse")

        self.out_seq_lens = {i: self.output_var(f"{i}.len") for i in range(NUM_TASKS)}
        self.out_num_segs = {i: self.output_var(f"{i}.processed") for i in range(NUM_TASKS)}
        self.out_num_skipped = {i: self.output_var(f"{i}.skipped") for i in range(NUM_TASKS)}
        self.out_tses = {i: self.output_var(f"{i}.tse") for i in range(NUM_TASKS)}

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self) -> Iterator[Task]:
        yield Task("run", args=list(range(NUM_TASKS)), rqmt=self.rqmt)
        yield Task("merge", mini_task=True)

    def run(self, task_id: int):
        alignment = FileArchiveBundle(cache(self.alignment))
        alignment.setAllophones(self.allophones.get_path())

        ref_alignment = FileArchiveBundle(cache(self.reference_alignment))
        ref_alignment.setAllophones(self.reference_allophones.get_path())

        segments = list(chunks([s for s in alignment.file_list() if not s.endswith(".attribs")], NUM_TASKS))[task_id]

        logging.info(f"processing {len(segments)} segments")

        s_idx = next(iter(alignment.files.values())).allophones.index("[SILENCE]{#+#}@i@f")
        ref_s_idx = next(iter(ref_alignment.files.values())).allophones.index("[SILENCE]{#+#}@i@f")

        skipped = 0
        seq_lens = {}
        tse = {}

        for seg in segments:
            a_states = [alignment.files[seg].allophones[mix] for _, mix, _, _ in alignment.read(seg, "align")]
            a_states_ref = [
                ref_alignment.files[seg].allophones[mix] for _, mix, _, _ in ref_alignment.read(seg, "align")
            ]

            if len(a_states) == 0:
                logging.warning(f"empty alignment for {seg}, skipping")
                continue

            a_states_dedup = [(k, len(list(g))) for k, g in itertools.groupby(a_states) if k != s_idx]
            a_states_ref_dedup = [(k, len(list(g))) for k, g in itertools.groupby(a_states_ref) if k != ref_s_idx]

            if len(a_states_dedup) != len(a_states_ref_dedup):
                logging.info(
                    f"len mismatch in {seg} of {len(a_states_dedup)} vs. {len(a_states_ref_dedup)}, skipping due to different pronunciation. {len(tse)} alignments already diffed."
                )

                skipped += 1
                continue

            # unzip the list
            a_states_dedup, a_states_duration = list(zip(*a_states_dedup))
            a_states_begins, a_states_ends = compute_begins_ends(a_states_duration)
            a_states_ref_dedup, a_states_ref_duration = list(zip(*a_states_ref_dedup))
            a_states_ref_begins, a_states_ref_ends = compute_begins_ends(a_states_ref_duration)

            seq_lens[seg] = len(a_states)
            tse[seg] = self._compute_distance(
                np.array(a_states_begins) * self.t_step,
                np.array(a_states_ends) * self.t_step,
                np.array(a_states_ref_begins) * self.reference_t_step,
                np.array(a_states_ref_ends) * self.reference_t_step,
            )

        self.out_seq_lens[task_id].set(seq_lens)
        self.out_num_segs[task_id].set(len(segments))
        self.out_num_skipped[task_id].set(skipped)
        self.out_tses[task_id].set(tse)

    def _compute_distance(
        self, begins: np.ndarray, ends: np.ndarray, ref_begins: np.ndarray, ref_ends: np.ndarray
    ) -> float:
        begins_err = np.abs(begins - ref_begins)
        ends_err = np.abs(ends - ref_ends)
        return (np.sum(begins_err) + np.sum(ends_err)) / begins.size

    def merge(self):
        seq_lens: Dict[int, Dict[str, int]] = {k: v.get() for k, v in self.out_seq_lens.items()}
        tses: Dict[int, Dict[str, float]] = {k: v.get() for k, v in self.out_tses.items()}

        all_tse = [tses[i][key] * seq_lens[i][key] for i in tses.keys() for key in tses[i].keys()]
        total_lens = [seq_len for len_map in seq_lens.values() for seq_len in len_map.values()]

        self.out_tse.set(sum(all_tse) / sum(total_lens))
