__all__ = ["DatasetConfig", "SamplingMethod"]

from dataclasses import dataclass

from sisyphus import tk
from i6_core.returnn import ReturnnConfig
from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.text import PipelineJob

from .corpus_setup import py as setup_corpus

@dataclass(frozen=True)
class All:
    pass

@dataclass(frozen=True)
class RandomFraction:
    fraction: float

@dataclass(frozen=True)
class RandomNumber:
    num: int

SamplingMethod = All | RandomFraction | RandomNumber


@dataclass
class DatasetConfig:
    audio_hdf_path: tk.Path
    sampling_method: SamplingMethod = All


def get_dataset_config(
    hdf_path: str | tk.Path,
    sampled_segments: tk.Path | None = None,
) -> ReturnnConfig:
    dataset_config = {
        "class": "HDFDataset",
        "files": [ hdf_path ],
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
        case All():
            sampled_segments = None
        case RandomFraction(f):
            sampled_segments = sample_segments_by_fraction(segments, fraction=f)
        case RandomNumber(n):
            sampled_segments = sample_segments_by_number(segments, num_segments=n)
    
    return sampled_segments


def create_config(dataset_config: DatasetConfig):
    corpus_res = setup_corpus()

    match dataset_config.sampling_method:
        case All():
            sampled_segments = None
        case RandomFraction(f):
            sampled_segments = sample_segments_by_fraction(corpus_res.segments, fraction=f)
        case RandomNumber(n):
            sampled_segments = sample_segments_by_number(corpus_res.segments, num_segments=n)
    
    return get_dataset_config(dataset_config.audio_hdf_path, sampled_segments), sampled_segments
