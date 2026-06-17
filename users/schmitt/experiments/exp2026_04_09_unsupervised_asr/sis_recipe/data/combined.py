from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset, get_subepoch_dataset
from i6_experiments.users.schmitt.datasets.distrib_files import DistributedFilesDataset
from i6_experiments.users.schmitt.datasets.combine import CombinedDataset

from ..data.common import DatasetSettings, TrainingDatasets
from ..data.wav2vec import run_meta_experiments
from ..data.text import get_phonemized_lm_data, get_dev_text
from ..pipeline import training
from ...default_tools import RETURNN_EXE, RETURNN_ROOT


def build_training_datasets():
    return TrainingDatasets(
        train=CombinedDataset(
            datasets={
                "phonemes": DistributedFilesDataset(
                    files=phoneme_train_hdfs if num_phoneme_hdfs is None else phoneme_train_hdfs[:num_phoneme_hdfs],
                    partition_epoch=phoneme_partition_epoch,
                    get_subepoch_dataset=functools.partial(
                        get_subepoch_dataset,
                        multi_proc=False,
                        postprocessing_opts={
                            "map_seq_stream": label_noiser,
                            "buf_size": 2,
                            "num_workers": 0,
                            # "map_outputs": {"data": {"dims": [Dim(None, name="T")], "dtype": "int64", "sparse_dim": Dim(sparse_dim, name="classes")}},
                        }
                    )
                ),
                "audio_features": DistributedFilesDataset(
                    # upsample audio data to match phoneme data size
                    # phoneme data has 18x more frames than audio data
                    # set partition_epoch 4x smaller and use 5x replication -> 20x
                    files=(cluster_ids_train_hdf if num_cluster_id_hdfs is None else cluster_ids_train_hdf[:num_cluster_id_hdfs]) * audio_upsampling_factor,
                    partition_epoch=audio_partition_epoch,
                    get_subepoch_dataset=functools.partial(
                        get_subepoch_dataset,
                        multi_proc=False,
                        postprocessing_opts={
                            "map_seq_stream": label_noiser,
                            "buf_size": 2,
                            "num_workers": 0,
                            # "map_outputs": {"data": {"dims": [Dim(None, name="T")], "dtype": "int64", "sparse_dim": Dim(sparse_dim, name="classes")}},
                        }
                    )
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="interleave",
            partition_epoch=1,
        ),
        cv=CombinedDataset(
            datasets={
                "phonemes": HdfDataset(
                    files=phoneme_dev_hdfs
                ),
                "audio_features": HdfDataset(
                    files=cluster_ids_dev_hdf
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="sorted",
            partition_epoch=1,
        ),
        devtrain=CombinedDataset(
            datasets={
                "phonemes": HdfDataset(
                    files=phoneme_devtrain_hdfs
                ),
                "audio_features": HdfDataset(
                    files=cluster_ids_devtrain_hdf
                ),
            },
            data_map={
                ("phonemes", "data"): "phon_indices",
                ("audio_features", "data"): "data",
            },
            seq_ordering="sorted",
            partition_epoch=1,
        ),
        datastreams={
            "features": LabelDatastreamWoVocab(
                available_for_inference=False,
                vocab_size=128,
            ),
            "labels": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        }
    )
