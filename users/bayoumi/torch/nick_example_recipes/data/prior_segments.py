from sisyphus import tk

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict


def get_prior_segments(data_share: float = 0.3) -> tk.Path:
    """
    Create a mixed cross-validation set containing
    1500 lines of dev-clean and 1500 lines of dev-other

    :return: line based segment file
    :rtype: Path
    """
    bliss_corpus_dict = get_bliss_corpus_dict(output_prefix="datasets")
    train_other = bliss_corpus_dict["train-other-960"]

    train_other_segments = SegmentCorpusJob(train_other, 1).out_single_segment_files[1]

    def shuffle_and_head(name, segment_file):
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file, split={"shuffle": 1.0}, shuffle=True
        )
        shuffle_segment_file_job.add_alias(f"datasets/LibriSpeech/prior_shuffle_{name}")
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, ratio=data_share).out

    train_other_prior_subset = shuffle_and_head("train_other", train_other_segments)

    return train_other_prior_subset
