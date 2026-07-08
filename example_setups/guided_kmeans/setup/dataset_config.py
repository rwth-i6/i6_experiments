__all__ = ["DatasetConfig", "SamplingMethod"]

from dataclasses import dataclass
from typing import Sequence

from sisyphus import tk, Job, Task
from i6_core.returnn import ReturnnConfig
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.text import PipelineJob
from i6_core.lib.corpus import Corpus

@dataclass(frozen=True)
class _All:
    pass

All = _All()

@dataclass(frozen=True)
class RandomFraction:
    fraction: float

@dataclass(frozen=True)
class RandomNumber:
    num: int

@dataclass(frozen=True)
class SegmentFile:
    path: tk.Path

SamplingMethod = _All | RandomFraction | RandomNumber | SegmentFile


@dataclass
class DatasetConfig:
    audio_hdf_path: tk.Path | list[tk.Path]
    sampling_method: SamplingMethod = All
    precomputed: bool = False


def get_dataset_config(
    hdf_path: str | tk.Path | Sequence[str | tk.Path],
    sampled_segments: tk.Path | None = None,
) -> ReturnnConfig:
    files = hdf_path
    if not isinstance(files, list):
        files = [files]
    dataset_config = {
        "class": "HDFDataset",
        "files": files,
        "partition_epoch": 1,
        "use_cache_manager": True,
    }
    if sampled_segments:
        dataset_config["seq_list_filter_file"] = sampled_segments

    config = dict(
        forward_data = dataset_config
    )
    
    return ReturnnConfig(config)


def sample_segments_by_fraction(all_segments: tk.Path, fraction: float = 0.05) -> tk.Path:
    split_segments = ShuffleAndSplitSegmentsJob(
        all_segments,
        split={
            "relevant": fraction,
            "remainder": 1.0 - fraction,
        }
    )
    segment_file = split_segments.out_segments["relevant"]
    return segment_file


def sample_segments_by_number(all_segments: tk.Path, num_segments: int = 20) -> tk.Path:
    pipe = PipelineJob(all_segments, [f"shuf -n {num_segments}"], zip_output=False)
    return pipe.out

def select_segments(method: SamplingMethod, segments: tk.Path) -> tk.Path | None:
    sampled_segments = tk.Path("")
    match method:
        case RandomFraction(f):
            sampled_segments = sample_segments_by_fraction(segments, fraction=f)
        case RandomNumber(n):
            sampled_segments = sample_segments_by_number(segments, num_segments=n)
        case _All():
            sampled_segments = None
        case SegmentFile(path):
            sampled_segments = path
    
    return sampled_segments


class CreateSequenceWhitelistJob(Job):
    def __init__(self, corpus):

        self.corpus = corpus
        self.out_whitelist = self.output_path("segments_whitelist.txt")


    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        corpus = Corpus()
        corpus.load(self.corpus.get_path())
        sampled_segments = [rec.fullname() for rec in corpus.recordings]
        with open(self.out_whitelist, "wt") as f:
            f.write("\n".join(sampled_segments))