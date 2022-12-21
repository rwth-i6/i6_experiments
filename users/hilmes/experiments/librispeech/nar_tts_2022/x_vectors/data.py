from sisyphus import tk
from typing import Dict
from dataclasses import dataclass

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
)
from i6_experiments.users.hilmes.data.tts_preprocessing import (
    extend_lexicon,
    process_corpus_text_with_extended_lexicon,
)
from i6_experiments.users.hilmes.data.librispeech import (
    get_ls_train_clean_100_tts_silencepreprocessed,
    get_librispeech_tts_segments,
)
from i6_experiments.users.hilmes.data.datastream import (
    Datastream,
    LabelDatastream,
    AudioFeatureDatastream,
    ReturnnAudioFeatureOptions,
    DBMelFilterbankOptions,
)
from i6_experiments.users.hilmes.data.datasets import (
    HDFDataset,
    MetaDataset,
    OggZipDataset,
)
from i6_experiments.users.rossenbach.tts.speaker_embedding import (
    SpeakerLabelHDFFromBliss,
)


@dataclass(frozen=True)
class TTSTrainingDatasets:
    """
    Dataclass for TTS Datasets
    """

    train: MetaDataset
    cv: MetaDataset
    datastreams: Dict[str, Datastream]


def _make_meta_dataset(audio_dataset, speaker_dataset):
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


def get_vocab_datastream(lexicon: tk.Path, alias_path: str) -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)
    :param lexicon:
    :param alias_path:
    :return:
    :rtype: VocabularyDatastream
    """
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon, blacklist={"[SILENCE]"})
    returnn_vocab_job.add_alias(alias_path + "/returnn_vocab_from_lexicon")

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size,
    )

    return vocab_datastream


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
    # TODO check values
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
    db_options = DBMelFilterbankOptions(fmin=60, fmax=fmax, min_amp=1e-10, center=center)
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

    audio_datastream = AudioFeatureDatastream(available_for_inference=available_for_inference, options=options)

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        train_ogg_zip,
        segment_file=segment_file,
        use_scalar_only=True,
        returnn_python_exe=returnn_cpu_exe,
        returnn_root=returnn_root,
        alias_path=output_path,
    )

    return audio_datastream


def get_xvector_data(returnn_exe: tk.Path, returnn_root: tk.Path, output_path: str, silence_prep: bool = True):

    if silence_prep:
        sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()
    else:
        # sil_pp_train_clean_100_co = get_corpus_object_dict(
        #  audio_format="ogg", output_prefix="corpora"
        # )["train-clean-100"]
        assert False, "not implemented yet"

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

    # Vocab Datastream probably not needed but for completeness
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
    speaker_label_job = SpeakerLabelHDFFromBliss(bliss_corpus=sil_pp_train_clean_100_tts)
    train_speakers = speaker_label_job.out_speaker_hdf
    speaker_datastream = LabelDatastream(
        available_for_inference=True,
        vocab_size=speaker_label_job.out_num_speakers,
        vocab=speaker_label_job.out_speaker_dict,
    )

    datastreams = {
        "audio_features": log_mel_datastream,
        "speaker_labels": speaker_datastream,
        "phonemes": vocab_datastream,
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
    train_dataset = _make_meta_dataset(train_ogg_zip, speaker_hdf_dataset)

    cv_ogg_zip = OggZipDataset(
        path=zip_dataset,
        audio_opts=log_mel_datastream.as_returnn_audio_opts(),
        target_opts=vocab_datastream.as_returnn_targets_opts(),
        segment_file=cv_segments,
        partition_epoch=1,
        seq_ordering="sorted",
    )
    cv_dataset = _make_meta_dataset(cv_ogg_zip, speaker_hdf_dataset)

    training_datasets = TTSTrainingDatasets(train=train_dataset, cv=cv_dataset, datastreams=datastreams)

    return training_datasets
