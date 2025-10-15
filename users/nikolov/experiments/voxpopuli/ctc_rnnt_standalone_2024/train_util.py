"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups

Here are (or rather, should be) the definitions for Tedlium-V2 data and RETURNN datasets that
are consistent across Phon/BPE as well as CTC/RNN-T/Attention systems
"""
from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from i6_core.returnn import CodeWrapper, BlissToOggZipJob

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream, BpeDatastream

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import ReturnnAudioRawOptions, AudioRawDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, MetaDataset

from .default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

DATA_PREFIX = "rescale/tedlium2_standalone_2023/data/"

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

# --------------------------- Helper functions  -----------------------------------


def get_zip(name: str, bliss_dataset: tk.Path):
    """

    :param name:
    :param bliss_dataset:
    :return:
    """
    zip_dataset_job = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=False,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    zip_dataset_job.add_alias(DATA_PREFIX + name)

    return zip_dataset_job.out_ogg_zip



@lru_cache()
def get_audio_raw_datastream():
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=False, preemphasis=0.97)
    )
    return audio_datastream

# --------------------------- Dataset functions  -----------------------------------




@lru_cache()
def build_test_dataset(dataset_key: str):
    """
    :param dataset_key: test dataset to generate ("eval" or "test")
    """

    #_, test_ogg = get_test_bliss_and_zip(dataset_key)
    #bliss_dict = get_bliss_corpus_dict()  # unmodified bliss

    #audio_datastream = get_audio_raw_datastream()

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
       # files=[test_ogg],
        #audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset#, bliss_dict[dataset_key]
