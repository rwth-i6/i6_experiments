"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from i6_core.returnn import CodeWrapper
from i6_core.corpus.convert import CorpusToTextDictJob

from i6_experiments.common.datasets.librispeech import get_subword_nmt_bpe, get_ogg_zip_dict, get_bliss_corpus_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioRawDatastream, \
    ReturnnAudioRawOptions
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from .default_tools import RETURNN_ROOT, RETURNN_EXE

# -------------- Dataclasses for configuration and data passing -------------------

# here: (<from-epoch> , <to-epoch>, <max-mean-length>)
EpochWiseFilter = Tuple[int, int, int]

@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]


@dataclass()
class TrainingDatasetSettings:
    # features settings
    custom_processing_function: Optional[str]

    # training settings
    partition_epoch: int
    epoch_wise_filters: List[EpochWiseFilter]
    seq_ordering: str

# --------------------------- Helper functions  -----------------------------------

@lru_cache()
def get_bpe_datastream(librispeech_key: str, bpe_size: int, is_recog: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    """
    bpe_settings = get_subword_nmt_bpe(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream(preemphasis: Optional[float] = None):
    """
    :param preemphasis: set the pre-emphasis filter factor
    :return:
    """
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=True, preemphasis=preemphasis)
    )
    return audio_datastream

# --------------------------- Dataset functions  -----------------------------------

def build_training_datasets(
        librispeech_key: str,
        bpe_size,
        preemphasis,
        settings: TrainingDatasetSettings,
    ) -> TrainingDatasets:
    """

    :param settings:
    :param output_path:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict['dev-clean']
    dev_other_ogg = ogg_zip_dict['dev-other']

    train_bpe_datastream = get_bpe_datastream(librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=False)
    audio_datastream = get_audio_raw_datastream(preemphasis)

    datastreams = {
        'audio_features': audio_datastream,
        'bpe_labels': train_bpe_datastream,
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if settings.custom_processing_function:
        training_audio_opts["pre_process"] = CodeWrapper(settings.custom_processing_function)

    additional_opts = {}
    if settings.epoch_wise_filters:
        additional_opts['epoch_wise_filter'] = {}
        for fr, to, max_mean_len in settings.epoch_wise_filters:
            additional_opts['epoch_wise_filter'][(fr, to)] = {"max_mean_len": max_mean_len}

    def make_meta(dataset: OggZipDataset):
        return MetaDataset(
            data_map=data_map,
            datasets={"zip_dataset": dataset},
            seq_order_control_dataset="zip_dataset"
        )

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = make_meta(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = make_meta(cv_zip_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_meta(devtrain_zip_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams
    )


@lru_cache()
def build_test_dataset(librispeech_key:str, dataset_key: str, bpe_size: int, preemphasis: Optional[float] = None):
    """

    :param dataset_key:
    :param bpe_size:
    :param preemphasis:
    :return:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key]).out_dictionary

    train_bpe_datastream = get_bpe_datastream(librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=True)

    audio_datastream = get_audio_raw_datastream(preemphasis)

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse"
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file