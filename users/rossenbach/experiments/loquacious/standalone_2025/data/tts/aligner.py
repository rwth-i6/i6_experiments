"""
Dataset helpers for aligner system or TTS systems that perform internal alignment

Basically means that audio, phonemes and speaker is provided, but no extra duration HDF
"""
from sisyphus import tk
from dataclasses import dataclass
from typing import Optional, Union

from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, HDFDataset
from i6_experiments.common.setups.returnn.datastreams.base import FeatureDatastream

from i6_experiments.users.rossenbach.datasets.loquacious import get_loquacious_tts_segments

from ..common import get_audio_raw_datastream, TrainingDatasets

from .tts_phon import (
    get_vocab_datastream,
    get_tts_bliss_and_zip,
    make_tts_meta_dataset,
)

from ...default_tools import MINI_RETURNN_ROOT


@dataclass(frozen=True)
class AlignmentTrainingDatasets(TrainingDatasets):
    """
    Dataclass for Alignment Datasets
    """

    # train: Dataset
    # cv: Dataset
    joint: Dataset


def build_training_dataset(
        loq_corpus_key:str = "train-small",
        partition_epoch: int = 1,
        dynamic_speaker_embeddings: Optional[tk.Path] = None,
        dynamic_speaker_embedding_size: Optional[Union[int, tk.Variable]] = None,
    ) -> AlignmentTrainingDatasets:
    """

    :param loq_corpus_key: which Loquacious part to use
    :param partition_epoch: partition factor for the training data
    """
    bliss_dataset, zip_dataset = get_tts_bliss_and_zip(loq_corpus_key=loq_corpus_key)

    # 1% of the data for cv
    train_segments, cv_segments = get_loquacious_tts_segments(loq_corpus_key=loq_corpus_key)

    vocab_datastream = get_vocab_datastream(with_blank=True, corpus_key=loq_corpus_key)
    audio_datastream = get_audio_raw_datastream()

    # we currently assume that train and cv share the same corpus file
    if dynamic_speaker_embeddings is None:
        # assert False, "Loquacious only supports dynamic embeddings"
        # we allow this now for vocoder training
        joint_speaker_dataset = None
    else:
        assert dynamic_speaker_embedding_size is not None
        joint_speaker_dataset = HDFDataset(
            files=[dynamic_speaker_embeddings],
        )
        speaker_datastream = FeatureDatastream(
            available_for_inference=True,
            feature_size=dynamic_speaker_embedding_size
        )


    # ----- Ogg and Meta datasets

    train_ogg_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=partition_epoch,
        seq_ordering="laplace:.1000"
    )
    train_dataset = make_tts_meta_dataset(train_ogg_dataset, joint_speaker_dataset, include_speakers=dynamic_speaker_embeddings)

    cv_ogg_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = make_tts_meta_dataset(cv_ogg_dataset, joint_speaker_dataset, include_speakers=dynamic_speaker_embeddings)
    
    devtrain_zip_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_tts_meta_dataset(devtrain_zip_dataset, joint_speaker_dataset, include_speakers=dynamic_speaker_embeddings)

    joint_ogg_zip = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted",
    )
    joint_metadataset = make_tts_meta_dataset(joint_ogg_zip, joint_speaker_dataset, include_speakers=dynamic_speaker_embeddings)

    # ----- final outputs

    datastreams = {
        "audio_features": audio_datastream,
        "phonemes": vocab_datastream,
    }
    if dynamic_speaker_embeddings:
        datastreams["speaker_embeddings"] = speaker_datastream

    align_datasets = AlignmentTrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        joint=joint_metadataset,
        datastreams=datastreams,
        prior=None,
    )

    return align_datasets
