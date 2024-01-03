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
        n_tasks: int = 1,
    ):
        assert sample_rate > 0
        assert n_tasks > 0

        self.alignment_bundle = alignment_bundle
        self.allophone_file = allophone_file
        self.sample_rate = sample_rate
        self.sil_allophone = sil_allophone

        self.out_skipped_per_task = [self.output_var(f"skipped_{i}") for i in range(n_tasks)]
        self.out_total_per_task = [self.output_var(f"total_{i}") for i in range(n_tasks)]

        self.out_total_phones = self.output_var("total_phones")
        self.out_total_skipped = self.output_var("total_skipped")
        self.out_ratio_skipped = self.output_var("ratio_skipped")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", args=list(range(len(self.out_total_per_task))), rqmt={"cpu": 1, "time": 1, "mem": 6}, mini_task=True)
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

        for segment in segs:
            alignment = [s.split(".")[0] for s in processor._get_alignment_states(segment)]

            idx = 0
            cur = None
            with_running_index = []
            for state in alignment:
                if state != cur:
                    idx += 1
                    cur = state
                with_running_index.append(f"{state}.{idx}")

            after_slice = len(set(with_running_index[:: self.sample_rate]))
            num_skipped = idx - after_slice

            total_skipped += num_skipped
            total_phones += idx

            print(alignment)
            print(with_running_index)
            print(after_slice)
            print(num_skipped)
            print(idx)

            break

        self.out_total_per_task[i].set(total_phones)
        self.out_skipped_per_task[i].set(total_skipped)

    def merge(self):
        total_phones = sum((v.get() for v in self.out_total_per_task))
        total_skipped = sum((v.get() for v in self.out_skipped_per_task))

        self.out_total_phones.set(total_phones)
        self.out_total_skipped.set(total_skipped)
        self.out_ratio_skipped.set(total_skipped / total_phones)
