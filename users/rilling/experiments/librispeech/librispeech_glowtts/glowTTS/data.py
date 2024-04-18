from dataclasses import dataclass
import os
from functools import lru_cache
from sisyphus import tk
from typing import Dict, List, Optional, Tuple

from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob
from i6_core.returnn import CodeWrapper, BlissToOggZipJob

from returnn_common.datasets import Dataset, OggZipDataset, HDFDataset, MetaDataset
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments

from ..data import (
    get_tts_log_mel_datastream,
    get_audio_raw_datastream,
    get_train_bliss_and_zip,
    get_vocab_datastream,
    get_mixed_cv_segments,
    get_bliss_corpus_dict,
    get_lexicon,
)
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_PYTORCH_EXE

EpochWiseFilter = Tuple[int, int, int]

@dataclass(frozen=True)
class TrainingDataset:
    """
    Dataclass for Alignment Datasets
    """

    train: Dataset
    cv: Dataset
    joint: Dataset
    datastreams: Dict[str, Datastream]

@dataclass()
class TrainingDatasetSettings:
    # features settings
    custom_processing_function: Optional[str]

    # training settings
    partition_epoch: int
    epoch_wise_filters: List[EpochWiseFilter]
    seq_ordering: str

def make_meta_dataset(audio_dataset, speaker_dataset, duration_dataset=None, xvector_dataset=None):
    """
    Shared function to create a metadatset with joined audio and speaker information

    :param datasets.OggZipDataset audio_dataset:
    :param datasets.HDFDataset speaker_dataset:
    :return:
    :rtype: MetaDataset
    """
    data_map = {
        "audio_features": ("audio", "data"),
        "phonemes": ("audio", "classes"),
        "speaker_labels": ("speaker", "data"),
    }

    ds = {
        "audio": audio_dataset.as_returnn_opts(), 
        "speaker": speaker_dataset.as_returnn_opts()
    }

    if duration_dataset:
        data_map["durations"] = ("durations", "data")
        ds["durations"] = duration_dataset.as_returnn_opts()

    if xvector_dataset is not None:
        data_map["xvectors"] = ("xvectors", "data")
        ds["xvectors"] = xvector_dataset.as_returnn_opts()

    meta_dataset = MetaDataset(
        data_map=data_map,
        datasets=ds,
        seq_order_control_dataset="audio",
    )
    return meta_dataset

def build_training_dataset(
    librispeech_key: str,
    settings: TrainingDatasetSettings,
    silence_preprocessing=False,
    xvectors_file=None,
) -> TrainingDataset:
    """

    :param settings:
    :param output_path:
    """
    train_bliss, train_ogg = get_train_bliss_and_zip("train-clean-100", silence_preprocessed=silence_preprocessing)
    # _, dev_clean_ogg = get_train_bliss_and_zip("dev-clean", silence_preprocessed=silence_preprocessing, remove_unk_seqs=True)
    # _, dev_other_ogg = get_train_bliss_and_zip("dev-other", silence_preprocessed=silence_preprocessing, remove_unk_seqs=True)

    train_bpe_datastream = get_vocab_datastream(corpus_key=librispeech_key, with_blank=True)
    audio_datastream = get_audio_raw_datastream()

    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key=librispeech_key)

    speaker_label_job = SpeakerLabelHDFFromBlissJob(
        bliss_corpus=train_bliss,
        returnn_root=MINI_RETURNN_ROOT,
    )
    joint_speaker_hdf = speaker_label_job.out_speaker_hdf

    joint_speaker_dataset = HDFDataset(
        files=[joint_speaker_hdf]
    )
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )

    datastreams = {
        "audio_features": audio_datastream,
        "phonemes": train_bpe_datastream,
        "speaker_labels": speaker_datastream,
    }

    if xvectors_file is not None:
        xvector_dataset = HDFDataset(files=[xvectors_file])
        xvector_dataset_train = HDFDataset(files=[xvectors_file], segment_file=train_segments)
        xvector_dataset_cv = HDFDataset(files=[xvectors_file], segment_file=cv_segments)
    else:
        xvector_dataset, xvector_dataset_train, xvector_dataset_cv = (None, None, None)

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if settings.custom_processing_function:
        training_audio_opts["pre_process"] = CodeWrapper(settings.custom_processing_function)

    additional_opts = {}
    if settings.epoch_wise_filters:
        additional_opts["epoch_wise_filter"] = {}
        for fr, to, max_mean_len in settings.epoch_wise_filters:
            additional_opts["epoch_wise_filter"][(fr, to)] = {"max_mean_len": max_mean_len}

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        segment_file=train_segments,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = make_meta_dataset(train_zip_dataset, joint_speaker_dataset, xvector_dataset=xvector_dataset_train)

    cv_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        seq_ordering="sorted_reverse",
    )
    cv_dataset = make_meta_dataset(cv_zip_dataset, joint_speaker_dataset, xvector_dataset=xvector_dataset_cv)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        # random_subset=3000,
    )
    devtrain_dataset = make_meta_dataset(devtrain_zip_dataset, joint_speaker_dataset, xvector_dataset=xvector_dataset)

    return TrainingDataset(train=train_dataset, cv=cv_dataset, joint=devtrain_dataset, datastreams=datastreams)

def build_training_dataset2(
        settings: TrainingDatasetSettings,
        ls_corpus_key="train-clean-100",
        durations_file=None,
        silence_preprocessed=True,
        ) -> TrainingDataset:
    """

    :param center: do feature centering
    """
    # bliss_dataset, zip_dataset = get_train_bliss_and_zip(ls_corpus_key=ls_corpus_key, silence_preprocessed=silence_preprocessed)

    train_bliss, train_ogg = get_train_bliss_and_zip(ls_corpus_key=ls_corpus_key, silence_preprocessed=silence_preprocessed)
    _, dev_clean_ogg = get_train_bliss_and_zip("dev-clean", silence_preprocessed=False, remove_unk_seqs=True)
    _, dev_other_ogg = get_train_bliss_and_zip("dev-other", silence_preprocessed=False, remove_unk_seqs=True)

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key=ls_corpus_key)

    vocab_datastream = get_vocab_datastream(with_blank=True, corpus_key=ls_corpus_key)
    # log_mel_datastream = get_tts_log_mel_datastream(center=center)

    audio_datastream = get_audio_raw_datastream()

    # we currently assume that train and cv share the same corpus file
    speaker_label_job = SpeakerLabelHDFFromBlissJob(
        bliss_corpus=train_bliss,
        returnn_root=MINI_RETURNN_ROOT,
    )
    joint_speaker_hdf = speaker_label_job.out_speaker_hdf

    joint_speaker_dataset = HDFDataset(
        files=[joint_speaker_hdf]
    )
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )

    if durations_file:
        duration_dataset = HDFDataset(
            files=[durations_file]
        )

    # ----- Ogg and Meta datasets
    training_audio_opts = audio_datastream.as_returnn_audio_opts()

    if settings.custom_processing_function:
        training_audio_opts["pre_process"] = CodeWrapper(settings.custom_processing_function)

    additional_opts = {}
    if settings.epoch_wise_filters:
        additional_opts["epoch_wise_filter"] = {}
        for fr, to, max_mean_len in settings.epoch_wise_filters:
            additional_opts["epoch_wise_filter"][(fr, to)] = {"max_mean_len": max_mean_len}

    train_ogg_dataset = OggZipDataset(
        path=train_ogg,
        audio_options=training_audio_opts,
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=settings.partition_epoch,
        seq_ordering="laplace:.1000"
    )
    if durations_file:
        train_dataset = make_meta_dataset(train_ogg_dataset, joint_speaker_dataset, duration_dataset=duration_dataset)
    else:
        train_dataset = make_meta_dataset(train_ogg_dataset, joint_speaker_dataset)

    cv_ogg_dataset = OggZipDataset(
        path=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    if durations_file:
        cv_dataset = make_meta_dataset(cv_ogg_dataset, joint_speaker_dataset, duration_dataset=duration_dataset)
    else:
        cv_dataset = make_meta_dataset(cv_ogg_dataset, joint_speaker_dataset)

    joint_ogg_zip = OggZipDataset(
        path=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted",
    )
    joint_metadataset = make_meta_dataset(joint_ogg_zip, joint_speaker_dataset)

    # ----- final outputs

    datastreams = {
        "audio_features": audio_datastream,
        "phonemes": vocab_datastream,
        "speaker_labels": speaker_datastream,
    }

    align_datasets = TrainingDataset(
        train=train_dataset,
        cv=cv_dataset,
        joint=joint_metadataset,
        datastreams=datastreams,
    )

    return align_datasets

def build_swer_test_dataset(synthetic_bliss, preemphasis: Optional[float] = None, peak_normalization: bool = False):
    """

    :param synthetic_bliss:
    :param preemphasis:
    :param peak_normalization:
    """
    zip_dataset_job = BlissToOggZipJob(
        bliss_corpus=synthetic_bliss,
        no_conversion=True,  # for Librispeech we are already having ogg
        returnn_python_exe=RETURNN_PYTORCH_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )

    audio_datastream = get_audio_raw_datastream(preemphasis, peak_normalization)

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[zip_dataset_job.out_ogg_zip],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset


@lru_cache()
def build_tts_forward_dataset(librispeech_key: str, dataset_key: str, xvectors_file: tk.Path = None):
    """

    :param librispeech_key: base librispeech training set for vocab generation
    :param dataset_key: test dataset to generate
    :param silence_preprocessing: use a setup with silence preprocessing
    """

    _, test_ogg = get_train_bliss_and_zip(ls_corpus_key=dataset_key, silence_preprocessed=False, remove_unk_seqs=True)
    bliss_dict = get_bliss_corpus_dict()
    # audio_datastream = get_audio_raw_datastream()
    tts_lexicon = get_lexicon(with_blank=True)
    phonemes_datastream = get_vocab_datastream(corpus_key=librispeech_key, with_blank=True)

    bliss_corpus = bliss_dict[dataset_key]

    data_map = {"audio": ("zip_dataset", "data"), "phonemes": ("zip_dataset", "classes")}

    audio_datastream = get_audio_raw_datastream()
    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=phonemes_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
    )

    if xvectors_file is not None:
        x_vectors_dataset = HDFDataset(files=[xvectors_file])
        data_map["xvectors"] = ("xvectors", "data")
        test_dataset = MetaDataset(
            data_map=data_map,
            datasets={"zip_dataset": test_zip_dataset, "xvectors": x_vectors_dataset},
            seq_order_control_dataset="zip_dataset",
        )
    else:
        speaker_label_job = SpeakerLabelHDFFromBlissJob(
            bliss_corpus=bliss_corpus,
            returnn_root=MINI_RETURNN_ROOT,
        )
        joint_speaker_hdf = speaker_label_job.out_speaker_hdf

        joint_speaker_dataset = HDFDataset(files=[joint_speaker_hdf])
        speaker_datastream = LabelDatastream(
            available_for_inference=True,
            vocab_size=speaker_label_job.out_num_speakers,
            vocab=speaker_label_job.out_speaker_dict,
        )

        data_map["speaker_labels"] = ("speakers", "data")
        test_dataset = MetaDataset(
            data_map=data_map,
            datasets={"zip_dataset": test_zip_dataset, "speakers": joint_speaker_dataset},
            seq_order_control_dataset="zip_dataset",
        )

    return test_dataset, bliss_corpus
