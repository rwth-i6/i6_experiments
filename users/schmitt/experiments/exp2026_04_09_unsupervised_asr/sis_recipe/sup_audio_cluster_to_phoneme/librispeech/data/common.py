from typing import Union, Dict, Optional, Any
from dataclasses import dataclass

from sisyphus import tk

from i6_core.text.processing import TakeNRandomLinesJob
from i6_core.serialization import Import

from i6_experiments.common.setups.returnn.datasets.base import MetaDataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datastreams.audio import AudioRawDatastream, ReturnnAudioRawOptions
from i6_experiments.users.schmitt.datasets.hdf import HdfDataset

from . import audio, text
from ....data.common import TrainingDatasets


class LabelDatastreamWoVocab(Datastream):
    """
    Defines a datastream for labels represented by indices using the default `Vocabulary` class of RETURNN

    This defines a word-(unit)-based vocabulary
    """

    def __init__(
        self,
        available_for_inference: bool,
        vocab_size: Union[tk.Variable, int],
    ):
        """

        :param available_for_inference:
        :param vocab: word vocab file path (pickle containing dictionary)
        :param vocab_size: used for the actual dimension
        :param unk_label: unknown label
        """
        super().__init__(available_for_inference)
        self.vocab_size = vocab_size

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(available_for_inference=available_for_inference),
            "shape": (None,),
            "dim": self.vocab_size,
            "sparse": True,
        }
        d.update(kwargs)
        return d


@dataclass()
class DatasetSettings:
    """
    A helper structure for the dataset settings that are configurable in RETURNN

    Args:
        custom_prcessing_function: the name of a python function added to the config
            this function can be used to process the input waveform
        partition_epoch: use this split of the data for one training epoch
        epoch_wise_filters: can be used to limit e.g. the sequence lengths at the beginning of the training
        seq_ordering: see RETURNN settings on sequence sorting
        preemphasis: filter scale for high-pass z-filter
        peak_normalization: normalize input utterance to unit amplitude peak
    """

    # training settings
    train_partition_epoch: int
    train_seq_ordering: str

    train_postprocessing_func: Optional[Import] = None

    # multiproc settings
    num_workers: int = 2


def get_audio_raw_datastream(
    preemphasis: Optional[float] = None, peak_normalization: bool = False
) -> AudioRawDatastream:
    """
    Return the datastream for raw-audio input settings for RETURNN

    :param preemphasis: set the pre-emphasis filter factor
    :param peak_normalization: normalize every utterance to peak amplitude 1
    """
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=peak_normalization, preemphasis=preemphasis),
    )
    return audio_datastream


def build_training_datasets():
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

    _, phoneme_vocab, lexicon_file, _ = text.get_phonemized_text("lm_minus_librivox", dump_hdf_concurrent=100)
    phoneme_960_hdfs, _, _, train_seq_tags = text.get_phonemized_text(
        "960h", lexicon_file=lexicon_file, dump_hdf_concurrent=10, phoneme_vocab=phoneme_vocab
    )
    phoneme_dev_hdfs, _, _, dev_seq_tags = text.get_phonemized_text(
        "dev", lexicon_file=lexicon_file, dump_hdf_concurrent=1, phoneme_vocab=phoneme_vocab
    )

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
                    partition_epoch=20,
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
        },
        datastreams={
            "data": LabelDatastreamWoVocab(
                available_for_inference=False,
                vocab_size=128,
            ),
            "target": LabelDatastream(
                available_for_inference=False,
                vocab=phoneme_vocab,
                vocab_size=41,
            ),
        },
    )
