from typing import Iterator

from i6_core.util import chunks
from sisyphus import Job, Path, Task

from .processor import AlignmentProcessor


class ComputeAlignmentSamplingStatisticsJob(Job):
    def __init__(
        self,
        alignment_bundle: Path,
        allophone_file: Path,
        sample_rate: int,
        sil_allophone: str = "[SILENCE]",
        n_tasks: int = 20,
    ):
        assert sample_rate > 0
        assert n_tasks > 0

        self.alignment_bundle = alignment_bundle
        self.allophone_file = allophone_file
        self.sample_rate = sample_rate
        self.sil_allophone = sil_allophone

        self.out_skipped_per_task = [self.output_var(f"skipped_{i}") for i in range(n_tasks)]
        self.out_total_per_task = [self.output_var(f"total_{i}") for i in range(n_tasks)]
        self.out_segments_with_sampling_per_task = [
            self.output_var(f"segments_with_sampling_{i}") for i in range(n_tasks)
        ]

        self.out_total_phones = self.output_var("total_phones")
        self.out_total_skipped = self.output_var("total_skipped")
        self.out_ratio_skipped = self.output_var("ratio_skipped")
        self.out_segments_with_sampling = self.output_var("segments_with_sampling")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", args=list(range(len(self.out_total_per_task))), rqmt={"cpu": 1, "time": 1, "mem": 6})
        yield Task("merge", mini_task=True)

    def run(self, i: int):
        processor = AlignmentProcessor(
            alignment_bundle_path=self.alignment_bundle.get_path(),
            allophones_path=self.allophone_file.get_path(),
            sil_allophone=self.sil_allophone,
            monophone=False,
        )
        segs = list(chunks(processor.segments, len(self.out_total_per_task)))[i]

        total_phones = 0
        total_skipped = 0
        segment_with_sampling = []

        for segment in segs:
            idx = 0
            cur = None
            with_running_index = []

            # split off state ID, add running index
            for hmm_state in processor.get_alignment_states(segment):
                state = hmm_state.split(".")[0]
                if state != cur:
                    idx += 1
                    cur = state
                with_running_index.append(f"{state}.{idx}")

            # measure how many phones remain after sampling
            after_slice = len(set(with_running_index[:: self.sample_rate]))
            num_skipped = idx - after_slice

            if num_skipped > 0:
                segment_with_sampling.append(segment)

            total_skipped += num_skipped
            total_phones += idx

        self.out_total_per_task[i].set(total_phones)
        self.out_skipped_per_task[i].set(total_skipped)
        self.out_segments_with_sampling_per_task[i].set(segment_with_sampling)

    def merge(self):
        total_phones = sum((v.get() for v in self.out_total_per_task))
        total_skipped = sum((v.get() for v in self.out_skipped_per_task))

        self.out_total_phones.set(total_phones)
        self.out_total_skipped.set(total_skipped)
        self.out_ratio_skipped.set(total_skipped / total_phones)
        self.out_segments_with_sampling.set([s for var in self.out_segments_with_sampling.values() for s in var.get()])
