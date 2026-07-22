from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob

from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset

from ....data.librispeech import audio, text
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings


def build_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    # we don't pass sil_prob here, because we just want to get the lexicon here
    # we don't use the text-only data for training here
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_clean_hdfs, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    phoneme_dev_other_hdfs, _, _, dev_other_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    dev_seq_tags = ConcatenateJob([dev_clean_seq_tags, dev_other_seq_tags], zip_out=False).out

    devtrain_seq_tags = TakeNRandomLinesJob(text_file=train_seq_tags, num_lines=3000).out
    dev_seq_tags = TakeNRandomLinesJob(text_file=dev_seq_tags, num_lines=3000).out

    return TrainingDatasets(
        # HDFDataset alone does not work with partition epoch...
        train=MetaDataset(
            datasets={
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=train_seq_tags,
                    # set here because this controls which seqs are loaded
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                "data": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
        eval_datasets={
            "devtrain": HdfDataset(
                files=phoneme_960_hdfs,
                segment_file=devtrain_seq_tags,
                seq_ordering="sorted_reverse",
            ),
            "dev": HdfDataset(
                files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                segment_file=dev_seq_tags,
                seq_ordering="sorted_reverse",
            ),
        },
        datastreams={
            "data": LabelDatastream(
                available_for_inference=True,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


def build_test_datasets(
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_dev_hdfs, _, _, dev_seq_tags = text.get_phonemized_text(
        "dev-other",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )

    return {
        "dev-other": HdfDataset(
            files=phoneme_dev_hdfs,
            segment_file=dev_seq_tags,
            seq_ordering="sorted_reverse",
        )
    }
