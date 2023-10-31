"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from i6_core.returnn import CodeWrapper
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioRawDatastream, \
    ReturnnAudioRawOptions
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

DATA_PREFIX = "experiments/librispeech/2023_standalone/data/"

# -------------- Dataclasses for configuration and data passing -------------------

# here: (<from-epoch> , <to-epoch>, <max-mean-length>)
EpochWiseFilter = Tuple[int, int, int]

@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]
    prior: Optional[Dataset]



@dataclass()
class TrainingDatasetSettings:
    # features settings
    custom_processing_function: Optional[str]

    # training settings
    partition_epoch: int
    epoch_wise_filters: List[EpochWiseFilter]
    seq_ordering: str
    preemphasis: float
    peak_normalization: bool

# --------------------------- Helper functions  -----------------------------------


@lru_cache()
def get_audio_raw_datastream(preemphasis: Optional[float] = None, peak_normalization: bool = False) -> AudioRawDatastream:
    """
    :param preemphasis: set the pre-emphasis filter factor
    :param peak_normalization: normalize every utterance to peak amplitude 1
    """
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=peak_normalization, preemphasis=preemphasis)
    )
    return audio_datastream


def get_zip(name: str, bliss_dataset: tk.Path):
    """

    :param name:
    :param bliss_dataset:
    :return:
    """
    zip_dataset_job = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=True,  # for Librispeech we are already having ogg
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    zip_dataset_job.add_alias(DATA_PREFIX + name)

    return zip_dataset_job.out_ogg_zip


# --------------------------- Dataset functions  -----------------------------------

def build_training_datasets(
        train_ogg: tk.Path,
        dev_clean_ogg: tk.Path,
        dev_other_ogg: tk.Path,
        label_datastream: Datastream,
        settings: TrainingDatasetSettings,
    ) -> TrainingDatasets:
    """
    :param train_ogg:
    :param dev_clean_ogg:
    :param dev_other_ogg:
    :param label_datastream:
    :param settings:
    """
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    datastreams = {
        'raw_audio': audio_datastream,
        'labels': label_datastream,
    }

    data_map = {"raw_audio": ("zip_dataset", "data"),
                "labels": ("zip_dataset", "classes")}

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
        target_options=label_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = make_meta(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = make_meta(cv_zip_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_meta(devtrain_zip_dataset)
    
    prior_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=label_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted_reverse",
        additional_options=additional_opts,
    )
    prior_dataset = make_meta(prior_zip_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
        prior=prior_dataset,
    )


@lru_cache()
def build_test_dataset(
        dataset_key: str,
        preemphasis: Optional[float] = None,
        peak_normalization: bool = False,
    ):
    """

    :param dataset_key: e.g. dev-other, which test set to create
    :param preemphasis:
    :param peak_normalization:
    :return:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]

    audio_datastream = get_audio_raw_datastream(preemphasis, peak_normalization)

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse"
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, bliss_dict[dataset_key]