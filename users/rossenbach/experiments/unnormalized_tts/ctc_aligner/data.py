from dataclasses import dataclass
import os
from sisyphus import tk
from typing import Dict

from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import GenericDataset, OggZipDataset, HDFDataset
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments

from ..data import (
    get_audio_raw_datastream,
    get_ls100_silence_preprocess_ogg_zip,
    get_ls100_silence_preprocessed_bliss,
    get_vocab_datastream,
    make_meta_dataset
)
from ..default_tools import RETURNN_ROOT

@dataclass(frozen=True)
class AlignmentTrainingDatasets:
    """
    Dataclass for Alignment Datasets
    """

    train: GenericDataset
    cv: GenericDataset
    joint: GenericDataset
    datastreams: Dict[str, Datastream]


def build_training_dataset(center : bool = False) -> AlignmentTrainingDatasets:
    """

    :param center: do feature centering
    """

    bliss_dataset = get_ls100_silence_preprocessed_bliss()
    zip_dataset = get_ls100_silence_preprocess_ogg_zip()

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments()

    vocab_datastream = get_vocab_datastream(with_blank=True)
    raw_datastream = get_audio_raw_datastream()

    # we currently assume that train and cv share the same corpus file
    speaker_label_job = SpeakerLabelHDFFromBlissJob(
        bliss_corpus=bliss_dataset,
        returnn_root=RETURNN_ROOT,
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
        path=zip_dataset,
        audio_options=raw_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=1,
        seq_ordering="laplace:.1000"
    )
    train_dataset = make_meta_dataset(train_ogg_dataset, joint_speaker_dataset)

    cv_ogg_dataset = OggZipDataset(
        path=zip_dataset,
        audio_options=raw_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = make_meta_dataset(cv_ogg_dataset, joint_speaker_dataset)

    joint_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_options=raw_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted",
    )
    joint_metadataset = make_meta_dataset(joint_ogg_zip, joint_speaker_dataset)

    # ----- final outputs

    datastreams = {
        "audio_samples": raw_datastream,
        "phon_labels": vocab_datastream,
        "speaker_labels": speaker_datastream,
    }

    align_datasets = AlignmentTrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        joint=joint_metadataset,
        datastreams=datastreams,
    )

    return align_datasets