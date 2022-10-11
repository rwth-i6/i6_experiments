from dataclasses import dataclass
from typing import Dict, Any, Optional

from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob

from i6_experiments.common.setups.returnn_common.serialization import DimInitArgs, DataInitArgs

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import MetaDataset, HDFDataset, OggZipDataset
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments
from i6_experiments.users.rossenbach.tts.duration_extraction import ViterbiAlignmentToDurationsJob


from ..data import (
    get_tts_log_mel_datastream,
    get_ls100_silence_preprocess_ogg_zip,
    get_ls100_silence_preprocessed_bliss,
    get_vocab_datastream,
    get_lexicon,
)

from ..default_tools import RETURNN_DATA_ROOT

@dataclass(frozen=True)
class TTSTrainingDatasets:
    """
    Dataclass for TTS Datasets
    """
    train: MetaDataset
    cv: MetaDataset
    datastreams: Dict[str, Datastream]


def get_tts_data_from_ctc_align(alignment):
    """
    Build the datastreams for TTS training
    :param tk.Path alignment: alignment hdf
    :return:
    """
    bliss_dataset = get_ls100_silence_preprocessed_bliss()
    zip_dataset = get_ls100_silence_preprocess_ogg_zip()

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments()

    vocab_datastream = get_vocab_datastream()
    log_mel_datastream = get_tts_log_mel_datastream(center=True)  # CTC setup is with window/frame centering

    speaker_label_job = SpeakerLabelHDFFromBlissJob(
        bliss_corpus=bliss_dataset,
        returnn_root=RETURNN_DATA_ROOT,
    )
    train_speakers = speaker_label_job.out_speaker_hdf
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )

    viterbi_job = ViterbiAlignmentToDurationsJob(
        alignment, bliss_lexicon=get_lexicon(with_blank=True), returnn_root=RETURNN_DATA_ROOT, time_rqmt=4, mem_rqmt=16
    )
    durations = viterbi_job.out_durations_hdf
    duration_datastream = DurationDatastream(available_for_inference=True)

    datastreams = {
        "audio_features": log_mel_datastream,
        "phon_labels": vocab_datastream,
        "speaker_labels": speaker_datastream,
        "phon_durations": duration_datastream,
    }

    train_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_options=log_mel_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=1,
        seq_ordering="laplace:.1000",
    )
    speaker_hdf_dataset = HDFDataset(files=[train_speakers])
    duration_hdf_dataset = HDFDataset(files=[durations])
    train_dataset = make_meta_dataset(
        train_ogg_zip, speaker_hdf_dataset, duration_hdf_dataset
    )

    cv_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_options=log_mel_datastream.as_returnn_audio_opts(),
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = make_meta_dataset(
        cv_ogg_zip, speaker_hdf_dataset, duration_hdf_dataset
    )

    training_datasets = TTSTrainingDatasets(
        train=train_dataset, cv=cv_dataset, datastreams=datastreams
    )

    return training_datasets, durations


def make_meta_dataset(audio_dataset, speaker_dataset, duration_dataset):
    """
    :param OggZipDataset audio_dataset:
    :param HDFDataset speaker_dataset:
    :param HDFDataset duration_dataset:
    :return:
    :rtype: MetaDataset
    """
    meta_dataset = MetaDataset(
        data_map={
            "audio_features": ("audio", "data"),
            "phon_labels": ("audio", "classes"),
            "speaker_labels": ("speaker", "data"),
            "phon_durations": ("duration", "data"),
        },
        datasets={
            "audio": audio_dataset.as_returnn_opts(),
            "speaker": speaker_dataset.as_returnn_opts(),
            "duration": duration_dataset.as_returnn_opts(),
        },
        seq_order_control_dataset="audio",
    )
    return meta_dataset


# Custom Datastream definitions
class DurationDatastream(Datastream):
    """
    Helper class for duration Datastreams
    """

    def as_returnn_extern_data_opts(
            self, available_for_inference: Optional[bool] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param available_for_inference:
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(
                available_for_inference=available_for_inference
            ),
            "dim": 1,
            "dtype": "int32",
        }
        d.update(kwargs)
        return d

    def as_nnet_constructor_data(
            self, name: str, available_for_inference: Optional[bool] = None, **kwargs
    ):

        d = self.as_returnn_extern_data_opts(
            available_for_inference=available_for_inference
        )
        time_dim = DimInitArgs(
            name="%s_time" % name,
            dim=None,
        )

        dim = d["dim"]
        feature_dim = DimInitArgs(
            name="%s_feature" % name,
            dim=dim,
            is_feature=True,
        )
        return DataInitArgs(
            name=name,
            available_for_inference=d["available_for_inference"],
            dim_tags=[time_dim, feature_dim],
            sparse_dim=None,
            dtype="int32",
        )
