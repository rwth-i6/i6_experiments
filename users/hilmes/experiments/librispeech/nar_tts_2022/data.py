from dataclasses import dataclass
from typing import Dict, Any, Optional
from sisyphus import tk
from copy import deepcopy

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.hdf import ReturnnDumpHDFJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.returnn.dataset import get_returnn_length_hdfs
from i6_core.meta import CorpusObject

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
)
from i6_experiments.users.rossenbach.datasets.librispeech import (
    get_librispeech_tts_segments,
)
from i6_experiments.users.hilmes.data.librispeech import (
    get_ls_train_clean_100_tts_silencepreprocessed,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datasets import (
    HDFDataset,
    MetaDataset,
    OggZipDataset,
    GenericDataset,
)
from i6_experiments.users.rossenbach.tts.speaker_embedding import (
    SpeakerLabelHDFFromBliss,
)
from i6_experiments.users.hilmes.data.datastream import (
    Datastream,
    LabelDatastream,
    DurationDatastream,
    AudioFeatureDatastream,
    ReturnnAudioFeatureOptions,
    DBMelFilterbankOptions,
)
from i6_experiments.users.hilmes.tools.tts.extract_alignment import (
    ExtractDurationsFromRASRAlignmentJob,
)
from i6_experiments.users.hilmes.tools.tts.viterbi_to_durations import (
    ViterbiToDurationsJob,
)
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import (
    DistributeSpeakerEmbeddings,
)
from i6_experiments.users.hilmes.data.tts_preprocessing import (
    extend_lexicon,
    process_corpus_text_with_extended_lexicon,
)


def dump_dataset(config, returnn_root, returnn_gpu_exe):
    """
    wrapper around DumpHDFJob
    :param config:
    :param returnn_root:
    :param returnn_gpu_exe:
    :return:
    """
    dataset_dump = ReturnnDumpHDFJob(
        data=config, returnn_root=returnn_root, returnn_python_exe=returnn_gpu_exe
    )
    return dataset_dump.out_hdf


@dataclass(frozen=True)
class AlignmentTrainingDatasets:
    """
    Dataclass for Alignment Datasets
    """

    train: GenericDataset
    cv: GenericDataset
    joint: GenericDataset
    datastreams: Dict[str, Datastream]


@dataclass(frozen=True)
class TTSTrainingDatasets:
    """
    Dataclass for TTS Datasets
    """

    train: MetaDataset
    cv: GenericDataset
    datastreams: Dict[str, Datastream]


@dataclass(frozen=True)
class TTSForwardData:
    """
    Dataclass for TTS Datasets
    """

    dataset: GenericDataset
    datastreams: Dict[str, Datastream]


@dataclass(frozen=True)
class VocoderDataclass:
    """
    Dataclass for TTS Datasets
    """

    zip: tk.Path
    audio_opts: AudioFeatureDatastream
    train_segments: tk.Path
    dev_segments: tk.Path


def _make_meta_dataset(audio_dataset, speaker_dataset, duration_dataset):
    """
    :param OggZipDataset audio_dataset:
    :param HDFDataset speaker_dataset:
    :param HDFDataset duration_dataset:
    :return:
    :rtype: MetaDataset
    """
    meta_dataset = MetaDataset(
        data_map={
            "phonemes": ("audio", "classes"),
            "audio_features": ("audio", "data"),
            "speaker_labels": ("speaker", "data"),
            "duration_data": ("duration", "data"),
        },
        datasets={
            "audio": audio_dataset.as_returnn_opts(),
            "speaker": speaker_dataset.as_returnn_opts(),
            "duration": duration_dataset.as_returnn_opts(),
        },
        seq_order_control_dataset="audio",
    )
    return meta_dataset


def _make_inference_meta_dataset(
    audio_dataset, speaker_dataset, duration_dataset: Optional
):
    """
    :param OggZipDataset audio_dataset:
    :param HDFDataset speaker_dataset:
    :param HDFDataset duration_dataset:
    :return:
    :rtype: MetaDataset
    """
    data_map = {
        "phonemes": ("audio", "classes"),
        "speaker_labels": ("speaker", "data"),
    }
    datasets = {
        "audio": audio_dataset.as_returnn_opts(),
        "speaker": speaker_dataset.as_returnn_opts(),
    }

    if duration_dataset is not None:
        data_map["duration_data"] = ("duration", "data")
        datasets["duration"] = duration_dataset.as_returnn_opts()

    meta_dataset = MetaDataset(
        data_map=data_map,
        datasets=datasets,
        seq_order_control_dataset="audio",
    )

    return meta_dataset


def _make_alignment_meta_dataset(audio_dataset, speaker_dataset):
    """
    :param OggZipDataset audio_dataset:
    :param HDFDataset speaker_dataset:
    :return:
    :rtype: MetaDataset
    """
    meta_dataset = MetaDataset(
        data_map={
            "phonemes": ("audio", "classes"),
            "audio_features": ("audio", "data"),
            "speaker_labels": ("speaker", "data"),
        },
        datasets={
            "audio": audio_dataset.as_returnn_opts(),
            "speaker": speaker_dataset.as_returnn_opts(),
        },
        seq_order_control_dataset="audio",
    )
    return meta_dataset


def get_tts_audio_datastream(
    train_ogg_zip,
    segment_file,
    returnn_cpu_exe,
    returnn_root,
    output_path,
    fmax=7600,
    available_for_inference=False,
    center=True,
):
    """
    Get the TTS audio datastream
    :param train_ogg_zip:
    :param segment_file:
    :param returnn_cpu_exe:
    :param returnn_root:
    :param output_path:
    :param fmax:
    :param available_for_inference:
    :param center:
    :return:
    """
    db_options = DBMelFilterbankOptions(
        fmin=60, fmax=fmax, min_amp=1e-10, center=center
    )
    options = ReturnnAudioFeatureOptions(
        sample_rate=16000,
        num_feature_filters=80,
        features="db_mel_filterbank",
        feature_options=db_options,
        preemphasis=0.97,
        window_len=0.05,
        step_len=0.0125,
        peak_normalization=False,
    )

    audio_datastream = AudioFeatureDatastream(
        available_for_inference=available_for_inference, options=options
    )

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        train_ogg_zip,
        segment_file=segment_file,
        use_scalar_only=True,
        returnn_python_exe=returnn_cpu_exe,
        returnn_root=returnn_root,
        alias_path=output_path,
    )

    return audio_datastream


def get_vocab_datastream(lexicon: tk.Path, alias_path: str) -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)
    :param lexicon:
    :param alias_path:
    :return:
    :rtype: VocabularyDatastream
    """
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(
        lexicon, blacklist={"[SILENCE]"}
    )
    returnn_vocab_job.add_alias(alias_path + "/returnn_vocab_from_lexicon")

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size,
    )

    return vocab_datastream


def get_returnn_durations(corpus, returnn_exe, returnn_root, output_path):
    """
    Get an hdf file containing the returnn durations after feature extraction
    :param corpus:
    :param returnn_exe:
    :param returnn_root:
    :param output_path:
    :return:
    """
    zip_dataset = BlissToOggZipJob(
        bliss_corpus=corpus,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    log_mel_datastream = get_tts_audio_datastream(
        zip_dataset,
        None,
        returnn_exe,
        returnn_root,
        output_path=output_path + "center_false",
        available_for_inference=False,
        center=False,
    )
    ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=None,
        segment_file=None,
        partition_epoch=1,
        seq_ordering="sorted",
    )

    extern_data = {
        "data": log_mel_datastream.as_returnn_extern_data_opts(
            available_for_inference=True
        )
    }
    data_keys = ["data"]
    hdf_dict = get_returnn_length_hdfs(
        dataset_dict=ogg_zip.as_returnn_opts(),
        dataset_keys=data_keys,
        extern_data=extern_data,
        returnn_root=returnn_root,
        returnn_exe=returnn_exe,
        mem_rqmt=8,
    )
    return hdf_dict["data"]


def get_tts_data_from_rasr_alignment(
    output_path, returnn_exe, returnn_root, rasr_alignment, rasr_allophones
):
    """
    Build the datastreams for TTS training from RASR alignment
    :param output_path:
    :param returnn_exe:
    :param returnn_root:
    :param rasr_alignment:
    :param rasr_allophones:
    :return:
    """
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()
    returnn_durations = get_returnn_durations(
        corpus=sil_pp_train_clean_100_co.corpus_file,
        returnn_exe=returnn_exe,
        returnn_root=returnn_root,
        output_path=(output_path + "/get_durations/"),
    )

    converter = ExtractDurationsFromRASRAlignmentJob(
        rasr_alignment=rasr_alignment,
        rasr_allophones=rasr_allophones,
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        target_duration_hdf=returnn_durations,
        silence_token="[SILENCE]",
        start_token="[start]",
        end_token="[end]",
        boundary_token="[space]",
    )
    converter.add_alias(output_path + "/extract_job")

    durations = converter.out_durations_hdf
    tk.register_output(output_path + "durations.hdf", durations)
    new_corpus = converter.out_bliss
    zip_dataset = BlissToOggZipJob(
        bliss_corpus=new_corpus,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    speaker_label_job = SpeakerLabelHDFFromBliss(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file
    )
    train_speakers = speaker_label_job.out_speaker_hdf

    # get datastreams
    train_segments, cv_segments = get_librispeech_tts_segments()
    log_mel_datastream = get_tts_audio_datastream(
        zip_dataset,
        train_segments,
        returnn_exe,
        returnn_root,
        output_path=output_path + "center_false",
        available_for_inference=False,
        center=False,
    )

    lexicon = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)[
        "train-clean-100"
    ]
    librispeech_g2p_lexicon = extend_lexicon(lexicon)

    vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, output_path)

    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )
    duration_datastream = DurationDatastream(available_for_inference=True)

    datastreams = {
        "audio_features": log_mel_datastream,
        "phonemes": vocab_datastream,
        "speaker_labels": speaker_datastream,
        "duration_data": duration_datastream,
    }

    train_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=1,
        seq_ordering="laplace:.1000",
    )

    speaker_hdf_dataset = HDFDataset(files=[train_speakers])
    train_duration_hdf = HDFDataset(files=[durations])
    train_dataset = _make_meta_dataset(
        train_ogg_zip, speaker_hdf_dataset, train_duration_hdf
    )

    cv_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = _make_meta_dataset(cv_ogg_zip, speaker_hdf_dataset, train_duration_hdf)

    audio_datastream_vocoder = get_tts_audio_datastream(
        zip_dataset,
        train_segments,
        returnn_exe,
        returnn_root,
        output_path=output_path + "center_true",
        available_for_inference=False,
        center=True,
    )
    # Temporary debugging
    full_ogg = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="laplace:.1000",
    )
    dump_audio = dump_dataset(
        full_ogg.as_returnn_opts(),
        returnn_root=returnn_root,
        returnn_gpu_exe=returnn_exe,
    )
    tk.register_output(output_path + "/returnn_extraction", dump_audio)

    tts_datasets = TTSTrainingDatasets(
        train=train_dataset, cv=cv_dataset, datastreams=datastreams
    )
    vocoder_data = VocoderDataclass(
        zip=zip_dataset,
        audio_opts=audio_datastream_vocoder,
        train_segments=train_segments,
        dev_segments=cv_segments,
    )
    return tts_datasets, vocoder_data, new_corpus, durations


def get_alignment_data(output_path, returnn_exe, returnn_root):
    """
    Build the data for alignment training
    :param output_path:
    :param returnn_exe:
    :param returnn_root:
    :return:
    """
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()
    librispeech_g2p_lexicon = extend_lexicon(
        get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-100"]
    )

    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=librispeech_g2p_lexicon,
    )

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=sil_pp_train_clean_100_tts,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip
    train_segments, cv_segments = get_librispeech_tts_segments()

    vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, output_path)
    log_mel_datastream = get_tts_audio_datastream(
        zip_dataset,
        train_segments,
        returnn_exe,
        returnn_root,
        output_path=output_path + "/center_true",
        available_for_inference=False,
        center=True,
    )
    speaker_label_job = SpeakerLabelHDFFromBliss(
        bliss_corpus=sil_pp_train_clean_100_tts
    )
    train_speakers = speaker_label_job.out_speaker_hdf
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )
    datastreams = {
        "audio_features": log_mel_datastream,
        "phonemes": vocab_datastream,
        "speaker_labels": speaker_datastream,
    }

    train_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=1,
        seq_ordering="laplace:.1000",
    )
    speaker_hdf_dataset = HDFDataset(files=[train_speakers])
    train_dataset = _make_alignment_meta_dataset(train_ogg_zip, speaker_hdf_dataset)

    cv_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    dev_dataset = _make_alignment_meta_dataset(cv_ogg_zip, speaker_hdf_dataset)

    dump_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted",
    )
    joint_metadataset = _make_alignment_meta_dataset(dump_ogg_zip, speaker_hdf_dataset)

    align_dataset = AlignmentTrainingDatasets(
        train=train_dataset,
        cv=dev_dataset,
        joint=joint_metadataset,
        datastreams=datastreams,
    )

    return align_dataset


def get_tts_data_from_ctc_align(output_path, returnn_exe, returnn_root, alignment):
    """
    Build the datastreams for TTS training
    :param output_path:
    :param returnn_exe:
    :param returnn_root:
    :param alignment:
    :return:
    """
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()

    librispeech_g2p_lexicon = extend_lexicon(
        get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-100"]
    )

    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=librispeech_g2p_lexicon,
    )

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=sil_pp_train_clean_100_tts,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    ).out_ogg_zip

    train_segments, cv_segments = get_librispeech_tts_segments()

    vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, output_path)
    log_mel_datastream = get_tts_audio_datastream(
        zip_dataset,
        train_segments,
        returnn_exe,
        returnn_root,
        output_path=output_path + "/center_true",
        available_for_inference=False,
        center=True,
    )
    speaker_label_job = SpeakerLabelHDFFromBliss(
        bliss_corpus=sil_pp_train_clean_100_tts
    )
    train_speakers = speaker_label_job.out_speaker_hdf
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )

    tk.register_output(output_path + "/alignment.hdf", alignment)
    viterbi_job = ViterbiToDurationsJob(
        alignment, skip_token=43, time_rqmt=4, mem_rqmt=16
    )
    viterbi_job.add_alias(output_path + "/viterbi_to_alignments")
    durations = viterbi_job.out_durations_hdf
    tk.register_output(output_path + "/durations.hdf", durations)
    duration_datastream = DurationDatastream(available_for_inference=True)

    datastreams = {
        "audio_features": log_mel_datastream,
        "phonemes": vocab_datastream,
        "speaker_labels": speaker_datastream,
        "duration_data": duration_datastream,
    }

    train_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=train_segments,
        partition_epoch=1,
        seq_ordering="laplace:.1000",
    )
    speaker_hdf_dataset = HDFDataset(files=[train_speakers])
    duration_hdf_dataset = HDFDataset(files=[durations])
    train_dataset = _make_meta_dataset(
        train_ogg_zip, speaker_hdf_dataset, duration_hdf_dataset
    )

    cv_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = _make_meta_dataset(
        cv_ogg_zip, speaker_hdf_dataset, duration_hdf_dataset
    )

    training_datasets = TTSTrainingDatasets(
        train=train_dataset, cv=cv_dataset, datastreams=datastreams
    )
    vocoder_data = VocoderDataclass(
        zip=zip_dataset,
        audio_opts=log_mel_datastream,
        train_segments=train_segments,
        dev_segments=cv_segments,
    )

    return training_datasets, vocoder_data


def get_inference_dataset(
    corpus_file: tk.Path,
    returnn_root,
    returnn_exe,
    datastreams: Dict[str, Any],
    durations: Optional,
    speaker_embedding_hdf,
    process_corpus: bool = True,
):

    if process_corpus:
        librispeech_g2p_lexicon = extend_lexicon(
            get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)[
                "train-clean-100"
            ]
        )

        inference_corpus = process_corpus_text_with_extended_lexicon(
            bliss_corpus=corpus_file,
            lexicon=librispeech_g2p_lexicon,
        )
    else:
        inference_corpus = corpus_file

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=inference_corpus,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
        no_audio=True,
    ).out_ogg_zip

    inference_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=None,
        target_opts=datastreams["phonemes"].as_returnn_targets_opts(),
        segment_file=None,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )
    speaker_hdf = DistributeSpeakerEmbeddings(
        speaker_embedding_hdf=speaker_embedding_hdf, bliss_corpus=inference_corpus
    ).out
    speaker_hdf_dataset = HDFDataset(files=[speaker_hdf])
    duration_hdf_dataset = None
    if durations is not None:
        duration_hdf_dataset = HDFDataset(files=[durations])
    inference_dataset = _make_inference_meta_dataset(
        inference_ogg_zip, speaker_hdf_dataset, duration_hdf_dataset
    )

    datastreams = deepcopy(datastreams)
    del datastreams["audio_features"]
    if durations is None:
        del datastreams["duration_data"]

    return TTSForwardData(dataset=inference_dataset, datastreams=datastreams)
