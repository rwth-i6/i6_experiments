from sisyphus import tk

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict


def get_dev_segments():
    dev = get_bliss_corpus_dict()["dev.all"]
    dev_all_segments = SegmentCorpusJob(dev, 1).out_single_segment_files[1]

    def shuffle_and_head(segment_file, num_lines):
        # only shuffle, this is deterministic
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file,
            split={"shuffle": 1.0},
            shuffle=True
        )
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, num_lines=num_lines).out

    dev_all_subset = shuffle_and_head(dev_all_segments, 3000)
    return dev_all_subset
