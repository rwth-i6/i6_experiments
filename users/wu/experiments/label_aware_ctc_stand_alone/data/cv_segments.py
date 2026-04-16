from sisyphus import tk

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict


def get_mixed_cv_segments(output_prefix: str = "datasets") -> tk.Path:
    """
    Create a mixed cross-validation set containing
    1500 lines of dev-clean and 1500 lines of dev-other

    :return: line based segment file
    :rtype: Path
    """
    bliss_corpus_dict = get_bliss_corpus_dict(output_prefix=output_prefix)
    dev_clean = bliss_corpus_dict["dev-clean"]
    dev_other = bliss_corpus_dict["dev-other"]

    dev_clean_segments = SegmentCorpusJob(dev_clean, 1).out_single_segment_files[1]
    dev_other_segments = SegmentCorpusJob(dev_other, 1).out_single_segment_files[1]

    def shuffle_and_head(name, segment_file, num_lines):
        # only shuffle, this is deterministic
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file, split={"shuffle": 1.0}, shuffle=True
        )
        shuffle_segment_file_job.add_alias(output_prefix + f"/LibriSpeech/cv_shuffle_{name}")
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, num_lines=num_lines).out

    dev_clean_subset = shuffle_and_head("dev_clean", dev_clean_segments, 1500)
    dev_other_subset = shuffle_and_head("dev_other", dev_other_segments, 1500)

    dev_cv_segments = PipelineJob([dev_clean_subset, dev_other_subset], [], mini_task=True).out

    return dev_cv_segments
