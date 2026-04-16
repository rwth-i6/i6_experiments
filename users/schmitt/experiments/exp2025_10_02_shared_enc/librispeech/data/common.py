"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk, Task
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.audio import AudioRawDatastream, ReturnnAudioRawOptions
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, MetaDataset

from .cv_segments import get_mixed_cv_segments
from .multi_proc import MultiProcDataset

from ...default_tools import RETURNN_ROOT, RETURNN_EXE

# -------------- Dataclasses for configuration and data passing -------------------


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]


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

    # general settings
    preemphasis: Optional[float]
    peak_normalization: bool

    # training settings
    train_partition_epoch: int
    train_seq_ordering: str
    train_additional_options: Optional[Dict[str, Any]] = None


# --------------------------- Helper functions  -----------------------------------


@lru_cache()
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


def get_zip(alias_name: str, bliss_dataset: tk.Path, no_audio: bool = False) -> tk.Path:
    """
    Helper function to generate an ogg-zips from a bliss corpus already containing ogg files

    :param alias_name: for the job alias
    :param bliss_dataset: path to the bliss corpus xml
    :param no_audio: do not include audio files in the zip
    :return: path to ogg-zip file
    """
    zip_dataset_job = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=True,  # for Librispeech we are already having ogg
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        no_audio=no_audio,
    )
    zip_dataset_job.add_alias(alias_name)

    return zip_dataset_job.out_ogg_zip


def remove_audio_from_oggzip(oggzip: tk.Path):
    job = RemoveAudioFromOggZipJob(oggzip)
    tk.register_output("data/librispeech/no_audio_oggzip", job.out_ogg_zip)
    return job.out_ogg_zip

# --------------------------- Dataset functions  -----------------------------------


def make_multi_proc(dataset: OggZipDataset):
    return MultiProcDataset(
        dataset=dataset,
        buffer_size=10,
        num_workers=4
    )


def build_training_datasets(
    train_ogg: Union[tk.Path, List[tk.Path]],
    dev_clean_ogg: tk.Path,
    dev_other_ogg: tk.Path,
    label_datastream: LabelDatastream,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """
    generic dataset construction helper to be used by the phon/bpe specific variants

    :param train_ogg: path to the train zip, potentially containing altered transcriptions
    :param dev_clean_ogg: path to the ls dev-clean zip, potentially containing altered transcriptions
    :param dev_other_ogg: path to the ls dev-other zip, potentially containing altered transcriptions
    :param label_datastream: label datastream (e.g. phoneme or bpe related)
    :param settings: settings object for the RETURNN data pipeline
    """
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    datastreams = {
        "raw_audio": audio_datastream,
        "labels": label_datastream,
    }

    data_map = {"raw_audio": ("zip_dataset", "data"), "labels": ("zip_dataset", "classes")}

    training_audio_opts = audio_datastream.as_returnn_audio_opts()

    def make_meta(dataset: OggZipDataset):
        return MetaDataset(
            data_map=data_map, datasets={"zip_dataset": dataset}, seq_order_control_dataset="zip_dataset"
        )

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=label_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
        additional_options=settings.train_additional_options,
    )
    train_dataset = make_multi_proc(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
    )
    cv_dataset = make_multi_proc(cv_zip_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_multi_proc(devtrain_zip_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
    )


def build_test_dataset(
    dataset_key: str,
    settings: DatasetSettings,
) -> Tuple[Dataset, tk.Path]:
    """
    Create ASR test set that only contains the audio stream

    :param dataset_key: e.g. dev-other, which test set to create
    :param settings: settings object for the RETURNN data pipeline
    :return: tuple of the test dataset and a path to the corresponding bliss corpus file
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]

    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    # data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg], audio_options=audio_datastream.as_returnn_audio_opts(), seq_ordering="sorted_reverse"
    )
    test_dataset = make_multi_proc(test_zip_dataset)
    # test_dataset = MetaDataset(
    #     data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    # )

    return test_dataset, bliss_dict[dataset_key]


class RemoveAudioFromOggZipJob(tk.Job):
    def __init__(self, oggzip_path: tk.Path):
        self.oggzip_path = oggzip_path

        self.out_ogg_zip = self.output_path("out.zip")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import zipfile

        with zipfile.ZipFile(self.oggzip_path.get_path(), "r") as zip_ref:
            for file in zip_ref.namelist():
                with zip_ref.open(file) as f:
                    print(file)
                    print(f.read())
                    exit()
