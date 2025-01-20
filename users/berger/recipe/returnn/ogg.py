__all__ = [
    "FilterOggzipSegmentsByLengthRatioJob",
]

from typing import List, Union

from sisyphus import Job, Task, setup_path, tk

assert __package__ is not None
Path = setup_path(__package__)


class FilterOggzipSegmentsByLengthRatioJob(Job):
    def __init__(
        self,
        oggzip_files: Union[tk.Path, List[tk.Path]],
        comparison_hdfs: List[tk.Path],
        min_ratio: float = 0,  # at least this many labels per second
        max_ratio: float = float("inf"),  # at most this many labels per second
    ) -> None:
        self.oggzip_files = oggzip_files
        if not isinstance(self.oggzip_files, list):
            self.oggzip_files = [self.oggzip_files]
        self.comparison_hdfs = comparison_hdfs
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.out_segments = self.output_path("segments")

    def tasks(self):
        yield Task("run", resume="run", rqmt={"cpu": 1, "mem": 16})

    def run(self) -> None:
        import h5py
        import zipfile

        metadata = []
        for oggzip_file in self.oggzip_files:
            with zipfile.ZipFile(oggzip_file.get(), "r") as zip_ref:
                with zip_ref.open("out.ogg.txt") as file:
                    lines = file.read().decode().split("\n")
                    for line in lines:
                        if line.startswith("{") and line.endswith("},"):
                            metadata.append(eval(line[:-1]))

        comparison_hdf_files = [h5py.File(comparison_hdf) for comparison_hdf in self.comparison_hdfs]
        comparison_length_map = {}
        for comparison_hdf_file in comparison_hdf_files:
            comparison_length_map.update(
                dict(
                    zip(
                        comparison_hdf_file["seqTags"],
                        [length[0] for length in comparison_hdf_file["seqLengths"]],
                    )
                )
            )

        result_segments = []

        num_too_short = 0
        num_too_long = 0
        for segment_metadata in metadata:
            seq_tag = segment_metadata["seq_name"]
            duration = segment_metadata["duration"]
            comparison_length = comparison_length_map[seq_tag.encode()]

            if comparison_length / duration >= self.max_ratio:
                print(
                    f"Sequence {seq_tag} is too short with duration {duration}s and {comparison_length} labels, should be at least {comparison_length / self.max_ratio:.1f}s."
                )
                num_too_short += 1
            elif comparison_length / duration <= self.min_ratio:
                print(
                    f"Sequence {seq_tag} is too long with duration {duration}s and {comparison_length} labels, should be at most {comparison_length / self.min_ratio:.1f}s."
                )
                num_too_long += 1
            else:
                result_segments.append(seq_tag)

        print(
            f"Finished processing. Out of a total of {len(metadata)} sequences, {num_too_short} were too short and {num_too_long} were too long, for a total of {num_too_short + num_too_long} filtered-out sequences."
        )

        with open(self.out_segments, "w") as file:
            for segment in result_segments:
                file.write(segment + "\n")
