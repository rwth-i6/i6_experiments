from dataclasses import dataclass
from typing import Dict, Any, Optional

from i6_core.returnn.dataset import SpeakerLabelHDFFromBlissJob
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.setups.returnn_common.serialization import DimInitArgs, DataInitArgs

from i6_experiments.users.rossenbach.common_setups.returnn.datasets import MetaDataset, HDFDataset, OggZipDataset, GenericDataset
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream, FeatureDatastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_bliss_corpus_dict
from i6_experiments.users.rossenbach.tts.duration_extraction import ViterbiAlignmentToDurationsJob


from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import (
    DistributeSpeakerEmbeddings, RandomSpeakerAssignmentJob, SingularizeHDFPerSpeakerJob, DistributeHDFByMappingJob, AverageF0OverDurationJob,
)

from ..data import (
    get_tts_log_mel_datastream,
    get_bliss_and_zip,
    get_vocab_datastream,
    get_lexicon,
    process_corpus_text_with_extended_lexicon
)

from ..default_tools import RETURNN_DATA_ROOT, RETURNN_EXE, RETURNN_RC_ROOT


@dataclass(frozen=True)
class TTSTrainingDatasets:
    """
    Dataclass for TTS Datasets
    """
    train: MetaDataset
    cv: MetaDataset
    datastreams: Dict[str, Datastream]


@dataclass(frozen=True)
class TTSForwardData:
    """
    Dataclass for TTS Datasets
    """

    dataset: GenericDataset
    datastreams: Dict[str, Datastream]


def get_tts_data_from_ctc_align(alignment_hdf, ls_corpus_key="train-clean-100", silence_preprocessed=True, partition_epoch=1):
    """
    Build the datastreams for TTS training
    :param tk.Path alignment_hdf: alignment hdf
    :return:
    """
    bliss_dataset, zip_dataset = get_bliss_and_zip(ls_corpus_key=ls_corpus_key, silence_preprocessed=silence_preprocessed)

    # segments for train-clean-100-tts-train and train-clean-100-tts-dev
    # (1004 segments for dev, 4 segments for each of the 251 speakers)
    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key=ls_corpus_key)

    vocab_datastream = get_vocab_datastream(corpus_key=ls_corpus_key)
    log_mel_datastream = get_tts_log_mel_datastream(center=False)  # CTC setup is with window/frame centering

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
        alignment_hdf, bliss_lexicon=get_lexicon(with_blank=True), returnn_root=RETURNN_DATA_ROOT, time_rqmt=4, mem_rqmt=16,
        blank_token=43,
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
        partition_epoch=partition_epoch,
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


def get_tts_forward_data_legacy(librispeech_subcorpus, speaker_embedding_hdf, segment_file = None, speaker_embedding_size=256):
    vocab_datastream = get_vocab_datastream()

    bliss_corpus = get_bliss_corpus_dict(audio_format="ogg")[librispeech_subcorpus]
    bliss_corpus_tts_format = process_corpus_text_with_extended_lexicon(
        bliss_corpus=bliss_corpus,
        lexicon=get_lexicon(corpus_key="train-other-960")  # use full lexicon
    )
    speaker_bliss_corpus = get_bliss_corpus_dict()["train-clean-100"]

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_corpus_tts_format,
        no_audio=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
    ).out_ogg_zip


    inference_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_options=None,
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=segment_file,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )

    mapping_pkl = RandomSpeakerAssignmentJob(bliss_corpus=bliss_corpus, speaker_bliss_corpus=speaker_bliss_corpus, shuffle=True).out_mapping
    if speaker_embedding_hdf:
        speaker_embedding_hdf = SingularizeHDFPerSpeakerJob(hdf_file=speaker_embedding_hdf, speaker_bliss=speaker_bliss_corpus).out_hdf
        speaker_hdf = DistributeHDFByMappingJob(hdf_file=speaker_embedding_hdf, mapping=mapping_pkl).out_hdf
        speaker_hdf_dataset = HDFDataset(files=[speaker_hdf])
    else:
        speaker_hdf_dataset = None

    inference_dataset = _make_inference_meta_dataset(
        inference_ogg_zip, speaker_hdf_dataset, duration_dataset=None
    )

    datastreams = {
        "phon_labels": vocab_datastream,
    }
    datastreams["speaker_labels"] = FeatureDatastream(
        available_for_inference=True, feature_size=speaker_embedding_size)

    return TTSForwardData(dataset=inference_dataset, datastreams=datastreams)


def get_tts_forward_data_legacy_v2(bliss_corpus, speaker_embedding_hdf, segment_file = None, speaker_embedding_size=256):
    vocab_datastream = get_vocab_datastream()

    bliss_corpus_tts_format = process_corpus_text_with_extended_lexicon(
        bliss_corpus=bliss_corpus,
        lexicon=get_lexicon(corpus_key="train-other-960")  # use full lexicon
    )
    speaker_bliss_corpus = get_bliss_corpus_dict()["train-clean-100"]

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_corpus_tts_format,
        no_audio=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
    ).out_ogg_zip


    inference_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_options=None,
        target_options=vocab_datastream.as_returnn_targets_opts(),
        segment_file=segment_file,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )

    mapping_pkl = RandomSpeakerAssignmentJob(bliss_corpus=bliss_corpus, speaker_bliss_corpus=speaker_bliss_corpus, shuffle=True).out_mapping
    if speaker_embedding_hdf:
        speaker_embedding_hdf = SingularizeHDFPerSpeakerJob(hdf_file=speaker_embedding_hdf, speaker_bliss=speaker_bliss_corpus).out_hdf
        speaker_hdf = DistributeHDFByMappingJob(hdf_file=speaker_embedding_hdf, mapping=mapping_pkl).out_hdf
        speaker_hdf_dataset = HDFDataset(files=[speaker_hdf])
    else:
        speaker_hdf_dataset = None

    inference_dataset = _make_inference_meta_dataset(
        inference_ogg_zip, speaker_hdf_dataset, duration_dataset=None
    )

    datastreams = {
        "phon_labels": vocab_datastream,
    }
    datastreams["speaker_labels"] = FeatureDatastream(
        available_for_inference=True, feature_size=speaker_embedding_size)

    return TTSForwardData(dataset=inference_dataset, datastreams=datastreams)


def _make_inference_meta_dataset(
        audio_dataset, speaker_dataset: Optional[HDFDataset], duration_dataset: Optional[HDFDataset]
):
    """
    :param OggZipDataset audio_dataset:
    :param HDFDataset speaker_dataset:
    :param HDFDataset duration_dataset:
    :return:
    :rtype: MetaDataset
    """
    data_map = {
        "phon_labels": ("audio", "classes"),
    }

    datasets = {
        "audio": audio_dataset.as_returnn_opts(),
    }

    if speaker_dataset is not None:
        data_map["speaker_labels"] = ("speaker", "data")
        datasets["speaker"] = speaker_dataset.as_returnn_opts()

    if duration_dataset is not None:
        data_map["duration_data"] = ("duration", "data")
        datasets["duration"] = duration_dataset.as_returnn_opts()

    meta_dataset = MetaDataset(
        data_map=data_map,
        datasets=datasets,
        seq_order_control_dataset="audio",
    )

    return meta_dataset


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
