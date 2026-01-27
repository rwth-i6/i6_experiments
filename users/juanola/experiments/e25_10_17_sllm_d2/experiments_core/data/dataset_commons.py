"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from functools import lru_cache
from typing import List, Optional, Tuple, Union

from sisyphus import tk

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset
from i6_experiments.common.setups.returnn.datastreams.audio import AudioRawDatastream, ReturnnAudioRawOptions
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.juanola.data.cross_validation import get_mixed_cv_segments
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.multi_proc_dataset import MultiProcDataset
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from ...default_tools import RETURNN_ROOT, RETURNN_EXE


@lru_cache()
def get_audio_raw_datastream(
    preemphasis: Optional[float] = None, peak_normalization: bool = False
) -> AudioRawDatastream:
    """
    Return the datastream for raw-audio input settings for RETURNN

    :param preemphasis: set the pre-emphasis filter factor
    :param peak_normalization: normalize every utterance to peak amplitude 1
    """
    return AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=peak_normalization, preemphasis=preemphasis),
    )


def make_multi_proc(dataset: Dataset, num_workers: int) -> MultiProcDataset:
    return MultiProcDataset(dataset=dataset, buffer_size=10, num_workers=num_workers)


def build_training_datasets(
    train_ogg: Union[tk.Path, List[tk.Path]],
    dev_clean_ogg: tk.Path,
    dev_other_ogg: tk.Path,
    label_datastream: LabelDatastream,
    returnn_settings: ReturnnDatasetSettings,
        datasets_num_workers:int,
) -> TrainingDatasets:
    """
    generic dataset construction helper to be used by the phon/bpe specific variants

    :param train_ogg: path to the train zip, potentially containing altered transcriptions
    :param dev_clean_ogg: path to the ls dev-clean zip, potentially containing altered transcriptions
    :param dev_other_ogg: path to the ls dev-other zip, potentially containing altered transcriptions
    :param label_datastream: label datastream (e.g. phoneme or bpe related)
    :param returnn_settings: settings object for the RETURNN data pipeline
    """
    audio_datastream = get_audio_raw_datastream(returnn_settings.preemphasis, returnn_settings.peak_normalization)

    datastreams = {
        "raw_audio": audio_datastream,
        "labels": label_datastream,
    }

    training_audio_opts = audio_datastream.as_returnn_audio_opts()

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=label_datastream.as_returnn_targets_opts(),
        partition_epoch=returnn_settings.train_partition_epoch,
        seq_ordering=returnn_settings.train_seq_ordering,
        additional_options=returnn_settings.train_additional_options,
    )
    train_dataset = make_multi_proc(train_zip_dataset, num_workers=datasets_num_workers)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
    )
    cv_dataset = make_multi_proc(cv_zip_dataset, num_workers=datasets_num_workers)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_multi_proc(devtrain_zip_dataset, num_workers=datasets_num_workers)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
    )


def build_test_dataset(
    dataset_key: str,
    settings: ReturnnDatasetSettings,
        datasets_num_workers
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

    test_zip_dataset = OggZipDataset(
        files=[test_ogg], audio_options=audio_datastream.as_returnn_audio_opts(), seq_ordering="sorted_reverse"
    )
    test_dataset = make_multi_proc(
        test_zip_dataset,
        datasets_num_workers
    )

    return test_dataset, bliss_dict[dataset_key]
