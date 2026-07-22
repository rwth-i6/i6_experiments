from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob
from i6_core.serialization import CallImport

from i6_experiments.common.datasets.librispeech.corpus import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datasets import OggZipDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset
from i6_experiments.users.schmitt.datasets.postprocessing import PostprocessingDataset
from i6_experiments.users.schmitt.datasets.lm import LmDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset
from i6_experiments.users.schmitt.datasets.utils.hdf import DumpCorpusTextAsUtf8ToHdfJob

from ....data.librispeech import audio, text
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings, _wrap_in_post_proc

from sisyphus import tk


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
        train=CombinedDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
                "phon_indices": HdfDataset(
                    files=phoneme_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
            },
            data_map={
                ("phon_indices", "data"): "phon_indices",
                ("feature_clusters", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
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
                    ("phon_indices", "data"): "phon_indices",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
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
                    ("phon_indices", "data"): "phon_indices",
                    ("feature_clusters", "data"): "data",
                },
                seq_ordering="sorted",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=128,
            ),
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )


def build_training_datasets_w_silence_in_input(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    min_num_sil: int = 1,
    max_num_sil: int = 3,
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

    # we use these functions only to get the correct seq tags because some seqs are filtered out in the process
    # because the words cannot be phonemized by the lexicon
    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    _, _, _, train_seq_tags = text.get_phonemized_text(
        "train-other-960",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=10,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    _, _, _, dev_clean_seq_tags = text.get_phonemized_text(
        "dev-clean",
        lexicon_file=lexicon_file,
        dump_hdf_concurrent=1,
        vocab_file=phoneme_vocab,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
    _, _, _, dev_other_seq_tags = text.get_phonemized_text(
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

    phoneme_datastream = LabelDatastream(
        available_for_inference=False,
        vocab=phoneme_vocab,
        vocab_size=41,
    )
    ogg_zip_dict = get_ogg_zip_dict()
    bliss_corpus_dict = get_bliss_corpus_dict("ogg")

    train_bytes_hdfs = DumpCorpusTextAsUtf8ToHdfJob(
        bliss_corpus=bliss_corpus_dict["train-other-960"], concurrent=10
    ).out_hdfs
    dev_other_hdf = DumpCorpusTextAsUtf8ToHdfJob(bliss_corpus=bliss_corpus_dict["dev-other"], concurrent=1).out_hdfs[0]
    dev_clean_hdf = DumpCorpusTextAsUtf8ToHdfJob(bliss_corpus=bliss_corpus_dict["dev-clean"], concurrent=1).out_hdfs[0]

    phonemize_func = CallImport(
        code_object_path="i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.models.post_proc.phonemize.PhonemizeAndInsertSilence",
        hashed_arguments={
            "sil_prob": sil_prob,
            "surround_w_sil": surround_w_sil,
            "lexicon_file": lexicon_file,
            "target_key": "data",
            "new_target_key": "data_w_sil",
            "vocab_opts": phoneme_datastream.as_returnn_targets_opts(),
            "min_num_sil": min_num_sil,
            "max_num_sil": max_num_sil,
        },
        unhashed_package_root=None,
        unhashed_arguments={},
    )

    data_map = {
        ("phon_indices", "data"): "phon_indices",
        ("phon_indices", "data_w_sil"): "phon_indices_w_sil",
        ("feature_clusters", "data"): "data",
    }
    map_outputs = {
        "data": phoneme_datastream.as_returnn_extern_data_opts(),
        "data_w_sil": phoneme_datastream.as_returnn_extern_data_opts(),
    }

    return TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                    partition_epoch=settings.train_partition_epoch,
                    seq_ordering=settings.train_seq_ordering,
                ),
                "phon_indices": _wrap_in_post_proc(
                    dataset=HdfDataset(
                        files=list(train_bytes_hdfs.values()),
                        segment_file=train_seq_tags,
                        partition_epoch=settings.train_partition_epoch,
                        seq_ordering=settings.train_seq_ordering,
                    ),
                    # dataset=OggZipDataset(
                    #     files=[
                    #         # ogg_zip_dict["train-clean-100"],
                    #         # ogg_zip_dict["train-clean-360"],
                    #         # ogg_zip_dict["train-other-500"],
                    #         ogg_zip_dict["train-other-960"],
                    #     ],
                    #     partition_epoch=settings.train_partition_epoch,
                    #     seq_ordering=settings.train_seq_ordering,
                    #     target_options={"class": "Utf8ByteTargets"},
                    #     segment_file=train_seq_tags,
                    # ),
                    # dataset=LmDataset(
                    #     corpus_file=bliss_corpus_dict["train-other-960"],
                    #     seq_list_file=train_seq_tags,
                    #     orth_vocab={"class": "Utf8ByteTargets"},
                    #     partition_epoch=settings.train_partition_epoch,
                    #     seq_ordering=settings.train_seq_ordering,
                    # ),
                    map_seq=phonemize_func,
                    map_outputs=map_outputs,
                ),
            },
            data_map=data_map,
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        eval_datasets={
            "devtrain": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_960_hdfs,
                        segment_file=devtrain_seq_tags,
                    ),
                    "phon_indices": _wrap_in_post_proc(
                        dataset=HdfDataset(
                            files=list(train_bytes_hdfs.values()),
                            segment_file=devtrain_seq_tags,
                            partition_epoch=1,
                        ),
                        map_seq=phonemize_func,
                        map_outputs=map_outputs,
                    ),
                },
                data_map=data_map,
                # sorted does not work/ is not implemented
                seq_ordering="interleave",
                partition_epoch=1,
            ),
            "dev": CombinedDataset(
                datasets={
                    "feature_clusters": HdfDataset(
                        files=clusters_dev_other_hdfs + clusters_dev_clean_hdfs,
                        segment_file=dev_seq_tags,
                    ),
                    "phon_indices": _wrap_in_post_proc(
                        dataset=HdfDataset(
                            files=[dev_clean_hdf, dev_other_hdf],
                            segment_file=dev_seq_tags,
                            partition_epoch=1,
                        ),
                        map_seq=phonemize_func,
                        map_outputs=map_outputs,
                    ),
                },
                data_map=data_map,
                seq_ordering="interleave",
                partition_epoch=1,
            ),
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=True,
                vocab_size=128,
            ),
            "phon_indices": phoneme_datastream,
            "phon_indices_w_sil": phoneme_datastream,
        },
    )


def build_text_only_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
):
    """
    Text-only (single-task) training data: just the phoneme indices (exposed under the
    ``phon_indices`` key), no audio clusters and no alternate batching. Used as a single-task
    reference for :func:`build_training_datasets` (multi-task text+audio). Reuses the exact same
    phoneme HDFs as the multi-task setup (same Sisyphus jobs), so the text data is identical.
    """
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

    def _phon_dataset(hdfs, seq_tags, partition_epoch=None, seq_ordering=None):
        # wrap in a MetaDataset only to expose the phoneme HDF under the "phon_indices" key
        # (a raw HdfDataset exposes "data"), so train + recog use the same key.
        return MetaDataset(
            datasets={
                "phon_indices": HdfDataset(
                    files=hdfs,
                    segment_file=seq_tags,
                    partition_epoch=partition_epoch,
                    seq_ordering=seq_ordering,
                ),
            },
            data_map={"phon_indices": ("phon_indices", "data")},
            seq_order_control_dataset="phon_indices",
        )

    return TrainingDatasets(
        train=_phon_dataset(
            phoneme_960_hdfs,
            train_seq_tags,
            partition_epoch=settings.train_partition_epoch,
            seq_ordering=settings.train_seq_ordering,
        ),
        eval_datasets={
            "devtrain": _phon_dataset(phoneme_960_hdfs, devtrain_seq_tags, seq_ordering="sorted"),
            "dev": _phon_dataset(phoneme_dev_clean_hdfs + phoneme_dev_other_hdfs, dev_seq_tags, seq_ordering="sorted"),
        },
        datastreams={
            "phon_indices": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
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
                "phon_indices": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        ),
    }
