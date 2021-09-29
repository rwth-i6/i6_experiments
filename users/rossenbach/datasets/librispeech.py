from functools import lru_cache
import os

from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.text.processing import HeadJob, PipelineJob

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.bpe import get_bpe_settings, get_returnn_subword_nmt


@lru_cache()
def get_librispeech_bpe(corpus_key="train-other-960", bpe_size=10000, unk_label="<unk>", output_prefix=""):
    """
    Get the BPE tokens via the subword-nmt fork for a librispeech setup.
    When using the default settings this will give a 100% compatible BPE settings to
    Albert Zeyers and Kazuki Iries setups.

    :param str corpus_key:
    :param int bpe_size:
    :param str output_prefix
    :return:
    :rtype: BPESettings
    """

    output_prefix = os.path.join(output_prefix, "librispeech_%s_bpe_%i" % (corpus_key, bpe_size))

    subword_nmt_commit_hash = "6ba4515d684393496502b79188be13af9cad66e2"
    subword_nmt_repo = get_returnn_subword_nmt(commit_hash=subword_nmt_commit_hash, output_prefix=output_prefix)
    train_other_960 = get_bliss_corpus_dict("flac", "corpora")[corpus_key]
    bpe_settings = get_bpe_settings(
        train_other_960,
        bpe_size=bpe_size,
        unk_label=unk_label,
        output_prefix=output_prefix,
        subword_nmt_repo_path=subword_nmt_repo)
    return bpe_settings


@lru_cache()
def get_mixed_cv_segments(output_prefix="datasets"):
    """
    Create a mixed crossvalidation set containing
    1500 lines of dev-clean and 1500 lines of dev-other

    :return:
    """
    bliss_corpus_dict = get_bliss_corpus_dict(output_prefix=output_prefix)
    dev_clean = bliss_corpus_dict['dev-clean']
    dev_other = bliss_corpus_dict['dev-other']

    dev_clean_segments = SegmentCorpusJob(dev_clean, 1).out_single_segment_files[1]
    dev_other_segments = SegmentCorpusJob(dev_other, 1).out_single_segment_files[1]

    def shuffle_and_head(segment_file, num_lines):
        # only shuffle, this is deterministic
        shuffle_segment_file_job = ShuffleAndSplitSegmentsJob(
            segment_file=segment_file,
            split={"shuffle": 1.0},
            shuffle=True
        )
        segment_file = shuffle_segment_file_job.out_segments["shuffle"]
        return HeadJob(segment_file, num_lines=num_lines).out

    dev_clean_subset = shuffle_and_head(dev_clean_segments, 1500)
    dev_other_subset = shuffle_and_head(dev_other_segments, 1500)

    dev_cv_segments = PipelineJob([dev_clean_subset, dev_other_subset], [], mini_task=True).out

    return dev_cv_segments