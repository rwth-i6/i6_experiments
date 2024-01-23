from dataclasses import dataclass
import os
from sisyphus import tk
from typing import Dict

from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, HDFDataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments

from .common import get_audio_raw_datastream, TrainingDatasets

from .tts_phon import (
    get_vocab_datastream,
    get_tts_bliss_and_zip,
    make_tts_meta_dataset,
)

from ..default_tools import MINI_RETURNN_ROOT

@dataclass(frozen=True)
class AlignmentTrainingDatasets(TrainingDatasets):
    """
    Dataclass for Alignment Datasets
    """

    # train: Dataset
    # cv: Dataset
    joint: Dataset
    # datastreams: Dict[str, Datastream]


def build_training_dataset(
        ls_corpus_key="train-clean-100",
        silence_preprocessed=False,
        partition_epoch=1
    ) -> AlignmentTrainingDatasets:
    """

    :param center: do feature centering
    """
    bliss_dataset, zip_dataset = get_tts_bliss_and_zip(ls_corpus_key=ls_corpus_key, silence_preprocessed=silence_preprocessed)

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key=ls_corpus_key)

    vocab_datastream = get_vocab_datastream(with_blank=True, corpus_key=ls_corpus_key)
    audio_datastream = get_audio_raw_datastream()

    # we currently assume that train and cv share the same corpus file
    speaker_label_job = SpeakerLabelHDFFromBlissJob(
        bliss_corpus=bliss_dataset,
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

    # ----- Ogg and Meta datasets

    train_ogg_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=partition_epoch,
        seq_ordering="laplace:.1000"
    )
    train_dataset = make_tts_meta_dataset(train_ogg_dataset, joint_speaker_dataset)

    cv_ogg_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = make_tts_meta_dataset(cv_ogg_dataset, joint_speaker_dataset)
    
    devtrain_zip_dataset = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_tts_meta_dataset(devtrain_zip_dataset, joint_speaker_dataset)

    joint_ogg_zip = OggZipDataset(
        files=zip_dataset,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted",
    )
    joint_metadataset = make_tts_meta_dataset(joint_ogg_zip, joint_speaker_dataset)

    # ----- final outputs

    datastreams = {
        "audio_features": audio_datastream,
        "phonemes": vocab_datastream,
        "speaker_labels": speaker_datastream,
    }

    align_datasets = AlignmentTrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        joint=joint_metadataset,
        datastreams=datastreams,
        prior=None,
    )

    return align_datasets
