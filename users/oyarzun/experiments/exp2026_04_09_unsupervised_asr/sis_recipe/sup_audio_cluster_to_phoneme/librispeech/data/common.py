from i6_core.text.processing import TakeNRandomLinesJob, ConcatenateJob

from i6_experiments.common.setups.returnn.datasets.base import MetaDataset, Dataset
class CombinedDataset(Dataset):
    def __init__(self, datasets, data_map, seq_ordering=None, additional_options=None):
        super().__init__(additional_options=additional_options)
        self.datasets = datasets
        self.data_map = data_map
        self.seq_ordering = seq_ordering
    def as_returnn_opts(self):
        opts = super().as_returnn_opts()
        opts["class"] = "CombinedDataset"
        opts["datasets"] = {k: v if isinstance(v, dict) else v.as_returnn_opts() for k, v in self.datasets.items()}
        opts["data_map"] = self.data_map
        if self.seq_ordering:
            opts["seq_ordering"] = self.seq_ordering
        return opts

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset

from ....data.librispeech import audio, text
from ....data.common import TrainingDatasets, LabelDatastreamWoVocab, DatasetSettings


def build_training_datasets(
    settings: DatasetSettings,
    sil_prob: float = 0.25,
    surround_w_sil: bool = True,
    include_lm_data: bool = True,
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

    # pass sil_prob here so we can use the text-only data for LM adversarial training
    lm_phoneme_hdfs, phoneme_vocab, lexicon_file, lm_seq_tags = text.get_phonemized_text(
        "lm_minus_librivox", 
        dump_hdf_concurrent=100,
        sil_prob=sil_prob,
        surround_w_sil=surround_w_sil,
    )
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

    datasets = {
        "acoustic": MetaDataset(
            datasets={
                "feature_clusters": HdfDataset(
                    files=clusters_960_hdfs,
                    segment_file=train_seq_tags,
                ),
                "phon_indices": {
                    **HdfDataset(
                        files=phoneme_960_hdfs,
                        segment_file=train_seq_tags,
                        # set here because this controls which seqs are loaded
                        partition_epoch=settings.train_partition_epoch,
                        seq_ordering=settings.train_seq_ordering,
                    ).as_returnn_opts(),
                    "cache_byte_size": 0,
                },
            },
            data_map={
                "data": ("feature_clusters", "data"),
                "target": ("phon_indices", "data"),
            },
            seq_order_control_dataset="phon_indices",
        )
    }
    
    data_map = {
        ("acoustic", "data"): "data",
        ("acoustic", "target"): "target",
    }
    
    if include_lm_data:
        datasets["lm"] = {
            **HdfDataset(
                files=lm_phoneme_hdfs,
                segment_file=lm_seq_tags,
                # Set to 2800 to yield ~14k sequences per epoch, balancing exactly with the acoustic data
                partition_epoch=2800,
                seq_ordering="random",
            ).as_returnn_opts(),
            "cache_byte_size": 0,
        }
        data_map[("lm", "data")] = "lm_text"

    if include_lm_data:
        train_dataset = CombinedDataset(
            datasets=datasets,
            data_map=data_map,
            seq_ordering="interleave",
        )
    else:
        train_dataset = datasets["acoustic"]

    return TrainingDatasets(
        add_opts={"line_based_lexicon_file": lexicon_file},
        train=train_dataset,
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
    )


def build_test_datasets():
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
