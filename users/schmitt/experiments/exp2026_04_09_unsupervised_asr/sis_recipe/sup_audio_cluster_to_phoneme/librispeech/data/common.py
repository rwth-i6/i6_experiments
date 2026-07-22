from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob

from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset

from ....data.librispeech import audio, text
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings


def build_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    _, clusters_960, pca_960, clusters_960_hdfs = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_other_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_clean_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-clean",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )

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
        train=MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=train_seq_tags,
                    # set here because this controls which seqs are loaded
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
        eval_datasets={
            "devtrain": MetaDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                },
                data_map={
                    "data": ("feature_clusters", "data"),
                    "target": ("phon_indices", "data"),
                },
                seq_order_control_dataset="phon_indices",
            ),
            "dev": MetaDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_dev_other_hdfs + clusters_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": HdfDataset(
                        files=phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                },
                data_map={
                    "data": ("feature_clusters", "data"),
                    "target": ("phon_indices", "data"),
                },
                seq_order_control_dataset="phon_indices",
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=128,
            ),
            "target": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
        add_opts={"line_based_lexicon_file": lexicon_file},
    )


def build_test_datasets(
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    _, clusters_960, pca_960, _ = audio.get_featurized_audio(
        librispeech_key="train-other-960",
        dump_hdf_concurrent=10,
        featurize_concurrent=10,
        remove_cluster_repetitions=True,
    )
    _, _, _, clusters_dev_other_hdfs = audio.get_featurized_audio(
        librispeech_key="dev-other",
        existing_clusters=clusters_960,
        existing_pca=pca_960,
        dump_hdf_concurrent=1,
        featurize_concurrent=1,
        remove_cluster_repetitions=True,
    )

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
        "dev-other": MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_dev_other_hdfs,
                    segment_file=dev_seq_tags,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_dev_hdfs,
                    segment_file=dev_seq_tags,
                ),
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }
