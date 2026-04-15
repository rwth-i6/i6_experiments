"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.audio import AudioRawDatastream, ReturnnAudioRawOptions
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, MetaDataset
from i6_experiments.common.setups.returnn.datasets.generic import HDFDataset

from .cv_segments import get_mixed_cv_segments

from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

# -------------- Dataclasses for configuration and data passing -------------------


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]
    prior: Optional[Dataset]


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


def get_zip(alias_name: str, bliss_dataset: tk.Path) -> tk.Path:
    """
    Helper function to generate an ogg-zips from a bliss corpus already containing ogg files

    :param alias_name: for the job alias
    :param bliss_dataset: path to the bliss corpus xml
    :return: path to ogg-zip file
    """
    zip_dataset_job = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=True,  # for Librispeech we are already having ogg
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    zip_dataset_job.add_alias(alias_name)

    return zip_dataset_job.out_ogg_zip


# --------------------------- Dataset functions  -----------------------------------


def build_oggzip_dataset_with_optional_hdf(
    *,
    ogg_files: Union[tk.Path, List[tk.Path]],
    audio_datastream: AudioRawDatastream,
    label_datastream: Optional[LabelDatastream] = None,
    hdf_file: Optional[Union[tk.Path, List[tk.Path]]] = None,
    hdf_datastream: Optional[Datastream] = None,
    hdf_stream_name: str = "alignments",
    hdf_data_key: str = "data",
    partition_epoch: Optional[int] = None,
    segment_file: Optional[tk.Path] = None,
    seq_ordering: Optional[str] = None,
    random_subset: Optional[int] = None,
    additional_options: Optional[Dict[str, Any]] = None,
    control_dataset: str = "zip_dataset",
) -> Tuple[Dataset, Dict[str, Datastream]]:
    """
    Build a MetaDataset around an OggZipDataset and an optional HDF side stream.

    The HDF dataset is expected to share sequence tags with the OggZip dataset.
    If ``hdf_file`` is not provided, the resulting MetaDataset only contains the OggZip dataset.
    """
    if (hdf_file is None) != (hdf_datastream is None):
        raise ValueError("hdf_file and hdf_datastream must either both be set or both be None.")

    datastreams: Dict[str, Datastream] = {"raw_audio": audio_datastream}
    data_map: Dict[str, Tuple[str, str]] = {"raw_audio": ("zip_dataset", "data")}
    target_options = None
    if label_datastream is not None:
        datastreams["labels"] = label_datastream
        data_map["labels"] = ("zip_dataset", "classes")
        target_options = label_datastream.as_returnn_targets_opts()

    zip_dataset = OggZipDataset(
        files=ogg_files,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=target_options,
        partition_epoch=partition_epoch,
        segment_file=segment_file,
        seq_ordering=seq_ordering,
        random_subset=random_subset,
        additional_options=additional_options,
    )
    datasets: Dict[str, Dataset] = {"zip_dataset": zip_dataset}

    if hdf_file is not None:
        datastreams[hdf_stream_name] = hdf_datastream
        data_map[hdf_stream_name] = ("hdf_dataset", hdf_data_key)
        datasets["hdf_dataset"] = HDFDataset(
            files=hdf_file,
            partition_epoch=partition_epoch,
            segment_file=segment_file,
            seq_ordering=seq_ordering,
            random_subset=random_subset,
        )

    dataset = MetaDataset(
        data_map=data_map,
        datasets=datasets,
        seq_order_control_dataset=control_dataset,
    )
    return dataset, datastreams


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
    train_dataset = make_meta(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
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
        additional_options=None,
    )
    prior_dataset = make_meta(prior_zip_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
        prior=prior_dataset,
    )


def build_training_datasets_with_hdf(
    train_ogg: Union[tk.Path, List[tk.Path]],
    dev_clean_ogg: tk.Path,
    dev_other_ogg: tk.Path,
    label_datastream: LabelDatastream,
    settings: DatasetSettings,
    *,
    hdf_file: Union[tk.Path, List[tk.Path]],
    hdf_datastream: Datastream,
    hdf_stream_name: str = "alignments",
    hdf_data_key: str = "data",
    train_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
    cv_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
    devtrain_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
    prior_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
) -> TrainingDatasets:
    """
    Dataset construction helper that combines raw audio + text labels from OggZip with an extra HDF stream.

    The HDF stream is assumed to share sequence tags with the corresponding OggZip entries.
    If only ``hdf_file`` is provided, the same HDF source is used for train/cv/devtrain/prior.
    """
    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    train_hdf = train_hdf or hdf_file
    cv_hdf = cv_hdf or hdf_file
    devtrain_hdf = devtrain_hdf or train_hdf
    prior_hdf = prior_hdf or train_hdf

    datastreams = {
        "raw_audio": audio_datastream,
        "labels": label_datastream,
        hdf_stream_name: hdf_datastream,
    }

    data_map = {
        "raw_audio": ("zip_dataset", "data"),
        "labels": ("zip_dataset", "classes"),
        hdf_stream_name: ("hdf_dataset", hdf_data_key),
    }

    training_audio_opts = audio_datastream.as_returnn_audio_opts()

    def make_meta(zip_dataset: OggZipDataset, hdf_dataset: HDFDataset, *, control_dataset: str = "zip_dataset"):
        return MetaDataset(
            data_map=data_map,
            datasets={"zip_dataset": zip_dataset, "hdf_dataset": hdf_dataset},
            seq_order_control_dataset=control_dataset,
        )

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=label_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
        additional_options=settings.train_additional_options,
    )
    train_hdf_dataset = HDFDataset(
        files=train_hdf,
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
    )
    train_dataset = make_meta(train_zip_dataset, train_hdf_dataset)

    cv_zip_dataset = OggZipDataset(
        files=[dev_clean_ogg, dev_other_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
    )
    cv_hdf_dataset = HDFDataset(
        files=cv_hdf,
        seq_ordering="sorted_reverse",
    )
    cv_dataset = make_meta(cv_zip_dataset, cv_hdf_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=label_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_hdf_dataset = HDFDataset(
        files=devtrain_hdf,
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_meta(devtrain_zip_dataset, devtrain_hdf_dataset)

    prior_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=label_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted_reverse",
        additional_options=None,
    )
    prior_hdf_dataset = HDFDataset(
        files=prior_hdf,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )
    prior_dataset = make_meta(prior_zip_dataset, prior_hdf_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
        prior=prior_dataset,
    )


def build_training_datasets_with_optional_hdf(
    train_ogg: Union[tk.Path, List[tk.Path]],
    dev_clean_ogg: tk.Path,
    dev_other_ogg: tk.Path,
    label_datastream: LabelDatastream,
    settings: DatasetSettings,
    *,
    hdf_file: Optional[Union[tk.Path, List[tk.Path]]] = None,
    hdf_datastream: Optional[Datastream] = None,
    hdf_stream_name: str = "alignments",
    hdf_data_key: str = "data",
    train_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
    cv_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
    devtrain_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
    prior_hdf: Optional[Union[tk.Path, List[tk.Path]]] = None,
) -> TrainingDatasets:
    """
    Like ``build_training_datasets`` but can optionally add an HDF-backed stream to each MetaDataset.
    """
    if hdf_file is None and hdf_datastream is None:
        return build_training_datasets(
            train_ogg=train_ogg,
            dev_clean_ogg=dev_clean_ogg,
            dev_other_ogg=dev_other_ogg,
            label_datastream=label_datastream,
            settings=settings,
        )
    if hdf_file is None or hdf_datastream is None:
        raise ValueError("hdf_file and hdf_datastream must either both be set or both be None.")

    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)
    training_audio_opts = audio_datastream.as_returnn_audio_opts()

    train_hdf = train_hdf or hdf_file
    cv_hdf = cv_hdf or hdf_file
    devtrain_hdf = devtrain_hdf or train_hdf
    prior_hdf = prior_hdf or train_hdf

    train_dataset, datastreams = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=label_datastream,
        hdf_file=train_hdf,
        hdf_datastream=hdf_datastream,
        hdf_stream_name=hdf_stream_name,
        hdf_data_key=hdf_data_key,
        partition_epoch=settings.train_partition_epoch,
        seq_ordering=settings.train_seq_ordering,
        additional_options=settings.train_additional_options,
    )

    cv_dataset, _ = build_oggzip_dataset_with_optional_hdf(
        ogg_files=[dev_clean_ogg, dev_other_ogg],
        audio_datastream=audio_datastream,
        label_datastream=label_datastream,
        hdf_file=cv_hdf,
        hdf_datastream=hdf_datastream,
        hdf_stream_name=hdf_stream_name,
        hdf_data_key=hdf_data_key,
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
    )

    devtrain_dataset, _ = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=label_datastream,
        hdf_file=devtrain_hdf,
        hdf_datastream=hdf_datastream,
        hdf_stream_name=hdf_stream_name,
        hdf_data_key=hdf_data_key,
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )

    prior_dataset, _ = build_oggzip_dataset_with_optional_hdf(
        ogg_files=train_ogg,
        audio_datastream=audio_datastream,
        label_datastream=label_datastream,
        hdf_file=prior_hdf,
        hdf_datastream=hdf_datastream,
        hdf_stream_name=hdf_stream_name,
        hdf_data_key=hdf_data_key,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
        prior=prior_dataset,
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
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]

    audio_datastream = get_audio_raw_datastream(settings.preemphasis, settings.peak_normalization)

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg], audio_options=audio_datastream.as_returnn_audio_opts(), seq_ordering="sorted_reverse"
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, bliss_dict[dataset_key]
