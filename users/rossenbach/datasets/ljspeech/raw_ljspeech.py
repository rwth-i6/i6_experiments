"""
This is the pipeline code for the "raw" ljspeech corpus, which can be used for TTS models with character input.
"""
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob, TailJob

from i6_experiments.users.rossenbach.datasets.common import DatasetGroup
from i6_experiments.common.datasets.ljspeech import get_16khz_corpus_object, get_22khz_corpus_object


def get_raw_ljspeech_dataset_group(sample_rate, alias_path=None):
    """
    Uses fixed shuffling and fixed sizes for dev (500) test (600) and train (12500) to have a standardized LJSpeech corpus

    No oggzip is added.

    :param int sample_rate: 16000 or 22050
    :param str alias_path:
    :return: a dataset group with the "raw" corpus divided into "ljspeech-dev" and "ljspeech-train"
    :rtype: DatasetGroup
    """
    corpus_object = None
    if sample_rate == 16000:
        corpus_object = get_16khz_corpus_object(alias_path)
    elif sample_rate == 22050:
        corpus_object = get_22khz_corpus_object(alias_path)
    else:
        assert sample_rate in [16000, 22050]

    dataset_group = DatasetGroup("ljspeech.raw")
    dataset_group.add_corpus_object("ljspeech", corpus_object)

    # generate a single segment file from corpus
    create_segment_file_job = SegmentCorpusJob(corpus_object.corpus_file, 1)
    segment_file = create_segment_file_job.out_single_segment_files[1]

    # only shuffle, this is deterministic
    shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
        segment_file=segment_file,
        split={"ljspeech": 1.0},
        shuffle=True
    )
    segment_file = shuffle_segment_file_job.out_segments["ljspeech"]

    devtest_segments = HeadJob(segment_file, num_lines=1100).out
    dev_segments = HeadJob(devtest_segments, num_lines=500).out
    test_segments = TailJob(devtest_segments, num_lines=600).out
    train_segments = TailJob(segment_file, num_lines=12000).out

    dataset_group.add_segmented_dataset("ljspeech-train", "ljspeech", train_segments)
    dataset_group.add_segmented_dataset("ljspeech-dev", "ljspeech", dev_segments)
    dataset_group.add_segmented_dataset("ljspeech-test", "ljspeech", test_segments)

    return dataset_group