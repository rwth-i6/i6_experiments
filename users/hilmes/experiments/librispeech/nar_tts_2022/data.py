from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from sisyphus import tk
from copy import deepcopy

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.hdf import ReturnnDumpHDFJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_experiments.common.setups.returnn.data import get_returnn_length_hdfs
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.corpus.segments import SegmentCorpusJob, ShuffleAndSplitSegmentsJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
)
from i6_experiments.users.rossenbach.datasets.librispeech import (
    get_librispeech_tts_segments,
)
from i6_experiments.users.hilmes.data.librispeech import (
    get_ls_train_clean_100_tts_silencepreprocessed, get_ls_train_clean_360_tts_silencepreprocessed
)
from i6_experiments.users.hilmes.data.datasets import (
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
    SpeakerEmbeddingDatastream,
    ReturnnAudioFeatureOptions,
    DBMelFilterbankOptions,
    F0Options,
    MFCCOptions
)
from i6_experiments.users.hilmes.tools.tts.extract_alignment import (
    ExtractDurationsFromRASRAlignmentJob,
)
from i6_experiments.users.hilmes.tools.tts.viterbi_to_durations import (
    ViterbiToDurationsJob,
)
from i6_experiments.users.hilmes.tools.tts.speaker_embeddings import (
    DistributeSpeakerEmbeddings, RandomSpeakerAssignmentJob, SingularizeHDFPerSpeakerJob, DistributeHDFByMappingJob, AverageF0OverDurationJob, AddSpeakerTagsFromMappingJob
)
from i6_experiments.users.hilmes.data.tts_preprocessing import (
    extend_lexicon,
    process_corpus_text_with_extended_lexicon,
)
from i6_experiments.users.hilmes.tools.data_manipulation import ApplyLogOnHDFJob
from i6_experiments.common.datasets.librispeech import (
  get_corpus_object_dict,
)

def dump_dataset(dataset, returnn_root, returnn_gpu_exe, name):
    """
    wrapper around DumpHDFJob
    :param dataset:
    :param returnn_root:
    :param returnn_gpu_exe:
    :return:
    """
    dataset_dump = ReturnnDumpHDFJob(
        data=dataset, returnn_root=returnn_root, returnn_python_exe=returnn_gpu_exe
    )
    dataset_dump.add_alias(name + "/dump_dataset")
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
    cv: MetaDataset
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
    audio_dataset, speaker_dataset: Optional[HDFDataset], duration_dataset: Optional[HDFDataset], prior_dataset: Optional[HDFDataset] = None,
    pitch_dataset: Optional[HDFDataset] = None, energy_dataset: Optional = None
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

    if prior_dataset is not None:
        data_map["speaker_prior"] = ("prior", "data")
        datasets["prior"] = prior_dataset.as_returnn_opts()

    if pitch_dataset is not None:
        data_map["pitch_data"] = ("pitch", "data")
        datasets["pitch"] = pitch_dataset.as_returnn_opts()

    if energy_dataset is not None:
        data_map["energy_data"] = ("energy", "data")
        datasets["energy"] = energy_dataset.as_returnn_opts()

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
        output_path=output_path + "/center_false",
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
    output_path, returnn_exe, returnn_root, rasr_alignment, rasr_allophones, silence_prep=True, speaker_embeddings: Tuple = None,
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
    if silence_prep:
        sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()
    else:
        corpus_object_dict = get_corpus_object_dict(
            audio_format="ogg", output_prefix="corpora"
        )
        sil_pp_train_clean_100_co = corpus_object_dict["train-clean-100"]
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
    tk.register_output(output_path + "/durations.hdf", durations)
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

    from i6_core.corpus.filter import FilterCorpusBySegmentsJob
    train_job = FilterCorpusBySegmentsJob(bliss_corpus=new_corpus, segment_file=train_segments)
    tk.register_output(output_path + "/train_corpus.xml.gz", train_job.out_corpus)
    dev_job = FilterCorpusBySegmentsJob(bliss_corpus=new_corpus, segment_file=cv_segments)
    tk.register_output(output_path + "/dev_corpus.xml.gz", dev_job.out_corpus)

    log_mel_datastream = get_tts_audio_datastream(
        zip_dataset,
        train_segments,
        returnn_exe,
        returnn_root,
        output_path=output_path + "/center_false",
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
    ) if speaker_embeddings is None else SpeakerEmbeddingDatastream(available_for_inference=True, embedding_size=speaker_embeddings[0])
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

    speaker_hdf_dataset = HDFDataset(files=[train_speakers if speaker_embeddings is None else speaker_embeddings[1]])
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
        output_path=output_path + "/center_true",
        available_for_inference=False,
        center=True,
    )

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


def get_vocoder_data(output_path):
    returnn_exe = tk.Path(
        "/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
        hash_overwrite="GENERIC_RETURNN_LAUNCHER",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="aadac2637ed6ec00925b9debf0dbd3c0ee20d6a6",
    ).out_repository
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()

    librispeech_g2p_lexicon = extend_lexicon(
        get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-100"]
    )

    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=librispeech_g2p_lexicon,
    )
    train_segments, cv_segments = get_librispeech_tts_segments()

    zip_job = BlissToOggZipJob(
        bliss_corpus=sil_pp_train_clean_100_tts,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    zip_job.add_alias(output_path + "/bliss_to_ogg")
    zip_dataset = zip_job.out_ogg_zip

    log_mel_datastream = get_tts_audio_datastream(
        zip_dataset,
        train_segments,
        returnn_exe,
        returnn_root,
        output_path=output_path + "/center_true",
        available_for_inference=False,
        center=True,
    )

    vocoder_data = VocoderDataclass(
        zip=zip_dataset,
        audio_opts=log_mel_datastream,
        train_segments=train_segments,
        dev_segments=cv_segments,
    )
    return vocoder_data


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

    zip_job = BlissToOggZipJob(
        bliss_corpus=sil_pp_train_clean_100_tts,
        no_conversion=True,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    zip_job.add_alias(output_path + "/bliss_to_ogg")
    zip_dataset = zip_job.out_ogg_zip

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

    return training_datasets, sil_pp_train_clean_100_tts, durations


def get_inference_dataset_old(
    corpus_file: tk.Path,
    returnn_root,
    returnn_exe,
    datastreams: Dict[str, Any],
    durations: Optional,
    speaker_embedding_hdf,
    speaker_embedding_size=256,
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
    datastreams["speaker_labels"] = SpeakerEmbeddingDatastream(
        available_for_inference=True, embedding_size=speaker_embedding_size)

    return TTSForwardData(dataset=inference_dataset, datastreams=datastreams)


def get_inference_dataset(
    corpus_file: tk.Path,
    returnn_root,
    returnn_exe,
    datastreams: Dict[str, Any],
    speaker_embedding_hdf: Optional = None,
    durations: Optional = None,
    speaker_prior_hdf: Optional = None,
    speaker_embedding_size=256,
    speaker_prior_size=32,
    pitch_hdf: Optional = None,
    energy_hdf: Optional = None,
    process_corpus: bool = True,
    original_corpus: Optional[tk.Path] = None,
    segments: Optional[tk.Path] = None,
    shuffle_info: bool = True,
    return_mapping: bool = False,
    original_speakers = False,
    alias: str = None,
):
    """
    Builds the inference dataset, gives option for different additional datasets to be passed depending on experiment
    :param corpus_file:
    :param returnn_root:
    :param returnn_exe:
    :param datastreams:
    :param speaker_embedding_hdf:
    :param durations:
    :param speaker_prior_hdf:
    :param speaker_embedding_size:
    :param speaker_prior_size:
    :param pitch_hdf
    :param process_corpus:
    :param shuffle_info: Whether to shuffle the speaker specific info or not. Usually True but for certain cheating false
    :return:
    """
    if energy_hdf is not None:
        assert durations is not None, "For now durations and energy have to be fed together for energy cheat"
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
        segment_file=segments,
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )
    if original_corpus:
        if alias is not None:
            job = RandomSpeakerAssignmentJob(bliss_corpus=corpus_file, speaker_bliss_corpus=original_corpus, shuffle=shuffle_info)
            job.add_alias(alias + "/speaker_assignment")
            mapping_pkl = job.out_mapping
        else:
            mapping_pkl = RandomSpeakerAssignmentJob(bliss_corpus=corpus_file, speaker_bliss_corpus=original_corpus, shuffle=shuffle_info).out_mapping
    else:
        if alias is not None:
            job = RandomSpeakerAssignmentJob(bliss_corpus=corpus_file, shuffle=shuffle_info)
            job.add_alias(alias + "/speaker_assignment")
            mapping_pkl = job.out_mapping
        else:
            mapping_pkl = RandomSpeakerAssignmentJob(bliss_corpus=corpus_file, shuffle=shuffle_info).out_mapping
    if original_speakers and speaker_embedding_hdf:
        speaker_hdf_dataset = HDFDataset(files=[speaker_embedding_hdf])
    elif speaker_embedding_hdf:
        speaker_embedding_job = SingularizeHDFPerSpeakerJob(hdf_file=speaker_embedding_hdf, speaker_bliss=corpus_file if original_corpus is None else original_corpus)
        if alias is not None:
            speaker_embedding_job.add_alias(alias + "/singularize_speaker_emb")
        speaker_embedding_hdf = speaker_embedding_job.out_hdf
        speaker_hdf = DistributeHDFByMappingJob(hdf_file=speaker_embedding_hdf, mapping=mapping_pkl).out_hdf
        speaker_hdf_dataset = HDFDataset(files=[speaker_hdf])
    else:
        speaker_hdf_dataset = None
    if speaker_prior_hdf is not None:
        prior_hdf = DistributeHDFByMappingJob(hdf_file=speaker_prior_hdf, mapping=mapping_pkl).out_hdf
        prior_hdf_dataset = HDFDataset(files=[prior_hdf])
    else:
        prior_hdf_dataset = None

    duration_hdf_dataset = HDFDataset(files=[durations]) if durations is not None else None
    pitch_hdf_dataset = HDFDataset(files=[pitch_hdf]) if pitch_hdf is not None else None
    energy_hdf_dataset = HDFDataset(files=[energy_hdf]) if energy_hdf is not None else None
    inference_dataset = _make_inference_meta_dataset(
        inference_ogg_zip, speaker_hdf_dataset, duration_hdf_dataset, prior_dataset=prior_hdf_dataset,
        pitch_dataset=pitch_hdf_dataset, energy_dataset=energy_hdf_dataset
    )

    datastreams = deepcopy(datastreams)
    del datastreams["audio_features"]
    if durations is None:
        del datastreams["duration_data"]
    datastreams["speaker_labels"] = SpeakerEmbeddingDatastream(
        available_for_inference=True, embedding_size=speaker_embedding_size)
    if speaker_prior_hdf is not None:
        datastreams["speaker_prior"] = SpeakerEmbeddingDatastream(
            available_for_inference=True, embedding_size=speaker_prior_size
        )
    if pitch_hdf is not None:
        datastreams["pitch_data"] = SpeakerEmbeddingDatastream(embedding_size=1, available_for_inference=True)
    if energy_hdf is not None:
        datastreams["energy_data"] = SpeakerEmbeddingDatastream(embedding_size=1, available_for_inference=True)

    if return_mapping:
        return TTSForwardData(dataset=inference_dataset, datastreams=datastreams), mapping_pkl
    else:
        return TTSForwardData(dataset=inference_dataset, datastreams=datastreams)


def get_ls_100_f0_hdf(durations: tk.Path, returnn_root: tk.Path, returnn_exe: tk.Path, prefix: str, center: bool = False, phoneme_level: bool = True):
    """
    Returns the pitch hdf for given duration mapping for the ls 100 corpus
    :param durations:
    :param returnn_root:
    :param returnn_exe:
    :param prefix:
    :param center:
    :param phoneme_level:
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
    options = ReturnnAudioFeatureOptions(
        sample_rate=16000,
        features="f0",
        feature_options=F0Options(fmin=80, fmax=400),
        window_len=0.05,
        step_len=0.0125,
        num_feature_filters=1
    )
    audio_datastream = AudioFeatureDatastream(available_for_inference=True, options=options)
    vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, prefix)
    full_ogg = OggZipDataset(
        path=zip_dataset,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted_reverse",
    )
    dataset_dump = ReturnnDumpHDFJob(
        data=full_ogg.as_returnn_opts(), returnn_root=returnn_root, returnn_python_exe=returnn_exe, time=24)
    job = AverageF0OverDurationJob(dataset_dump.out_hdf, duration_hdf=durations, center=True, phoneme_level=phoneme_level)
    job.add_alias(prefix + "/average")
    avrg = job.out_hdf
    tk.register_output(prefix + "/f0_durations", avrg)
    return avrg


def extend_meta_datasets_with_f0(datasets: TTSTrainingDatasets, f0_dataset: tk.Path):

    train_meta = deepcopy(datasets.train)
    train_meta.datasets["pitch"] = HDFDataset(files=[f0_dataset]).as_returnn_opts()
    train_meta.data_map["pitch_data"] = ("pitch", "data")

    cv_meta = deepcopy(datasets.cv)
    cv_meta.datasets["pitch"] = HDFDataset(files=[f0_dataset]).as_returnn_opts()
    cv_meta.data_map["pitch_data"] = ("pitch", "data")

    datastreams = deepcopy(datasets.datastreams)
    datastreams["pitch_data"] = SpeakerEmbeddingDatastream(embedding_size=1, available_for_inference=False)

    return TTSTrainingDatasets(train=train_meta, cv=cv_meta, datastreams=datastreams)


def extend_meta_datasets_with_energy(datasets: TTSTrainingDatasets, energy_dataset: tk.Path):

    train_meta = deepcopy(datasets.train)
    train_meta.datasets["energy"] = HDFDataset(files=[energy_dataset]).as_returnn_opts()
    train_meta.data_map["energy_data"] = ("energy", "data")

    cv_meta = deepcopy(datasets.cv)
    cv_meta.datasets["energy"] = HDFDataset(files=[energy_dataset]).as_returnn_opts()
    cv_meta.data_map["energy_data"] = ("energy", "data")

    datastreams = deepcopy(datasets.datastreams)
    datastreams["energy_data"] = SpeakerEmbeddingDatastream(embedding_size=1, available_for_inference=False)

    return TTSTrainingDatasets(train=train_meta, cv=cv_meta, datastreams=datastreams)


def get_ls_100_energy_hdf(returnn_root: tk.Path, returnn_exe: tk.Path, prefix: str, center=True, log_norm=False):
  """
  Returns the pitch hdf for given duration mapping for the ls 100 corpus
  :param durations:
  :param returnn_root:
  :param returnn_exe:
  :param prefix:
  :param center:
  :param phoneme_level:
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
  options = ReturnnAudioFeatureOptions(
    sample_rate=16000,
    features="mfcc",
    feature_options=MFCCOptions(center=center),
    window_len=0.05,
    step_len=0.0125,
    num_feature_filters=1,
  )
  audio_datastream = AudioFeatureDatastream(available_for_inference=True, options=options)
  vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, prefix)
  full_ogg = OggZipDataset(
    path=zip_dataset,
    audio_opts=audio_datastream.as_returnn_audio_opts(),
    target_opts=vocab_datastream.as_returnn_targets_opts(),
    partition_epoch=1,
    seq_ordering="sorted_reverse",
  )
  dataset_dump = ReturnnDumpHDFJob(
    data=full_ogg.as_returnn_opts(), returnn_root=returnn_root, returnn_python_exe=returnn_exe, time=24)
  tk.register_output(prefix + "/energy", dataset_dump.out_hdf)
  hdf = dataset_dump.out_hdf
  if log_norm:
    hdf = ApplyLogOnHDFJob(hdf_file=hdf, normalize=True).out_hdf
  return hdf


def get_ls360_100h_data():

    sil_pp_train_clean_360_co = get_ls_train_clean_360_tts_silencepreprocessed("datasets/ls360")
    librispeech_g2p_lexicon = extend_lexicon(
        get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-360"]
    )

    sil_pp_train_clean_360_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_360_co.corpus_file,
        lexicon=librispeech_g2p_lexicon,
    )
    ls_360_segments = SegmentCorpusJob(sil_pp_train_clean_360_tts, 1).out_single_segment_files[1]
    ls_360_100h_segments = ShuffleAndSplitSegmentsJob(ls_360_segments, split={'train': 0.27, 'dev': 0.73}).out_segments["train"]

    return sil_pp_train_clean_360_tts, ls_360_100h_segments


def get_ls_100_features(vocoder, returnn_root: tk.Path, returnn_exe: tk.Path, prefix: str, center=True, silence_prep=True,
                        add_speaker_tags=False):
    from i6_private.users.hilmes.tools.tts import VerifyCorpus, MultiJobCleanup
    from i6_core.corpus import (
        CorpusReplaceOrthFromReferenceCorpus,
        MergeCorporaJob,
        SegmentCorpusJob,
    )
    if silence_prep:
        sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()
    else:
        sil_pp_train_clean_100_co = get_corpus_object_dict(
            audio_format="ogg", output_prefix="corpora"
        )["train-clean-100"]
    librispeech_g2p_lexicon = extend_lexicon(
        get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-100"]
    )
    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=librispeech_g2p_lexicon,
    )

    forward_segments = SegmentCorpusJob(sil_pp_train_clean_100_tts, 10)
    verifications = []
    output_corpora = []
    reconstruction_norm = True
    vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, prefix)

    for i in range(10):
        name = prefix + f"/part_{i}"
        zip_dataset = BlissToOggZipJob(
            segments=forward_segments.out_single_segment_files[i + 1],
            bliss_corpus=sil_pp_train_clean_100_tts,
            no_conversion=True,
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root,
        ).out_ogg_zip

        db_options = DBMelFilterbankOptions(
            fmin=60, fmax=7600, min_amp=1e-10, center=center
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
            available_for_inference=True, options=options
        )

        audio_datastream.add_global_statistics_to_audio_feature_datastream(
            zip_dataset,
            segment_file=None,
            use_scalar_only=True,
            returnn_python_exe=returnn_exe,
            returnn_root=returnn_root,
            alias_path=name,
        )

        full_ogg = OggZipDataset(
            path=zip_dataset,
            audio_opts=audio_datastream.as_returnn_audio_opts(),
            target_opts=vocab_datastream.as_returnn_targets_opts(),
            partition_epoch=1,
            seq_ordering="sorted_reverse",
        )

        dataset_dump = ReturnnDumpHDFJob(
            data=full_ogg.as_returnn_opts(), returnn_root=returnn_root, returnn_python_exe=returnn_exe, time=24)
        tk.register_output(name + "/audio_features", dataset_dump.out_hdf)
        hdf = dataset_dump.out_hdf

        forward_vocoded, vocoder_forward_job = vocoder.vocode(
            hdf, iterations=30, cleanup=True, name=name, recon_norm=reconstruction_norm
        )
        tk.register_output(name + "/synthesized_corpus.xml.gz", forward_vocoded)
        output_corpora.append(forward_vocoded)
        verification = VerifyCorpus(forward_vocoded).out
        verifications.append(verification)

        cleanup = MultiJobCleanup(
            [dataset_dump, vocoder_forward_job], verification, output_only=True
        )
        tk.register_output(
            name + "/cleanup/cleanup.log", cleanup.out
        )

    from i6_core.corpus.transform import MergeStrategy

    merge_job = MergeCorporaJob(
        output_corpora, "train-clean-100", merge_strategy=MergeStrategy.FLAT
    )
    for verfication in verifications:
        merge_job.add_input(verfication)

    cv_synth_corpus = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=merge_job.out_merged_corpus,
        reference_bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
    ).out_corpus

    tk.register_output(prefix + "/synthesized_corpus.xml.gz", cv_synth_corpus)
    if add_speaker_tags:
        job = RandomSpeakerAssignmentJob(bliss_corpus=sil_pp_train_clean_100_co.corpus_file, shuffle=False)
        job.add_alias(prefix + "/get_speaker_tags")
        speaker_mapping = job.out_mapping
        tag_corpus_job = AddSpeakerTagsFromMappingJob(corpus=cv_synth_corpus, mapping=speaker_mapping)
        tag_corpus_job.add_alias(prefix + "/add_tags")
        tk.register_output(prefix + "/add_tags/corpus.xml.gz", tag_corpus_job.out_corpus)
        cv_synth_corpus = tag_corpus_job.out_corpus
    return cv_synth_corpus


def cov_prepare_ls_100_alignment(
  alignment: tk.Path,
  allophones: tk.Path,
  name_prefix: str,
  returnn_root: tk.Path,
  returnn_exe: tk.Path,
  only_dev: bool = True):

  sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()
  librispeech_g2p_lexicon = extend_lexicon(
    get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-100"]
  )
  train_segments, cv_segments = get_librispeech_tts_segments()

  returnn_durations = get_returnn_durations(
    corpus=sil_pp_train_clean_100_co.corpus_file,
    returnn_exe=returnn_exe,
    returnn_root=returnn_root,
    output_path=(name_prefix + "/get_durations/"),
  )

  converter = ExtractDurationsFromRASRAlignmentJob(
    rasr_alignment=alignment["tts_align_sat"],
    rasr_allophones=allophones,
    bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
    target_duration_hdf=returnn_durations,
    silence_token="[SILENCE]",
    start_token="[start]",
    end_token="[end]",
    boundary_token="[space]",
  )
  converter.add_alias(name_prefix + "/extract_job")
  durations = converter.out_durations_hdf
  corpus = converter.out_bliss
  tk.register_output(name_prefix + "/durations.hdf", durations)

  if only_dev:
    corpus = FilterCorpusBySegmentsJob(
      bliss_corpus=corpus,
      segment_file=cv_segments).out_corpus
  tk.register_output(name_prefix + "/corpus.xml.gz", corpus)

  zip_dataset = BlissToOggZipJob(
    segments=None,
    bliss_corpus=corpus,
    no_conversion=True,
    returnn_python_exe=returnn_exe,
    returnn_root=returnn_root,
  ).out_ogg_zip
  audio_datastream = get_tts_audio_datastream(
    train_ogg_zip=zip_dataset,
    segment_file=cv_segments,
    returnn_cpu_exe=returnn_exe,
    returnn_root=returnn_root,
    output_path=name_prefix,
    center=False
  )

  vocab_datastream = get_vocab_datastream(librispeech_g2p_lexicon, name_prefix)

  full_ogg = OggZipDataset(
    path=zip_dataset,
    audio_opts=audio_datastream.as_returnn_audio_opts(),
    target_opts=vocab_datastream.as_returnn_targets_opts(),
    partition_epoch=1,
    seq_ordering="sorted_reverse",
  )
  dataset_dump = ReturnnDumpHDFJob(
      data=full_ogg.as_returnn_opts(), returnn_root=returnn_root, returnn_python_exe=returnn_exe, time=24)
  tk.register_output(name_prefix + "/audio_features", dataset_dump.out_hdf)

