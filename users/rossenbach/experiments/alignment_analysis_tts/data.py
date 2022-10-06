from dataclasses import asdict
from functools import lru_cache
import os
from sisyphus import tk
from typing import List

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioFeatureDatastream, DBMelFilterbankOptions, ReturnnAudioFeatureOptions, FeatureType
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.rossenbach.common_setups.returnn import datasets

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_ls_train_clean_100_tts_silencepreprocessed
from i6_experiments.users.rossenbach.setups.tts.preprocessing import (
    process_corpus_text_with_extended_lexicon,
    extend_lexicon_with_tts_lemmas,
    extend_lexicon_with_blank
)

from .default_tools import RETURNN_EXE, RETURNN_DATA_ROOT

DATA_PREFIX = "experiments/alignment_analysis_tts/data/"


@lru_cache
def get_librispeech_lexicon() -> tk.Path:
    """

    :return:
    """
    return extend_lexicon_with_tts_lemmas(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-100"])


@lru_cache
def get_ls100_silence_preprocessed_bliss() -> tk.Path:
    """
    Get the modified ls100 corpus for the TTS task
    :return: Bliss xml file
    """
    # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()

    # get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker


    # convert the corpus transcriptions into phoneme and marker representation
    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=get_librispeech_lexicon())

    return sil_pp_train_clean_100_tts


@lru_cache
def get_ls100_silence_preprocess_ogg_zip() -> tk.Path:
    """
    :return: Returnn OggZip .zip file
    """

    sil_pp_train_clean_100_tts = get_ls100_silence_preprocessed_bliss()

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=sil_pp_train_clean_100_tts,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_DATA_ROOT,
    ).out_ogg_zip

    return zip_dataset


def make_meta_dataset(audio_dataset, speaker_dataset):
    """
    Shared function to create a metadatset with joined audio and speaker information

    :param datasets.OggZipDataset audio_dataset:
    :param datasets.HDFDataset speaker_dataset:
    :return:
    :rtype: MetaDataset
    """
    meta_dataset = datasets.MetaDataset(
        data_map={'audio_features': ('audio', 'data'),
                  'phon_labels': ('audio', 'classes'),
                  'speaker_labels': ('speaker', 'data'),
                  },
        datasets={
            'audio': audio_dataset.as_returnn_opts(),
            'speaker': speaker_dataset.as_returnn_opts()
        },
        seq_order_control_dataset="audio",
    )
    return meta_dataset


def get_tts_log_mel_datastream(
        center: bool = False,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    Supports both centered and non-centered windowing, as we need non-centered windowing for RASR-compatible
    feature extraction, but centered windowing to support linear-features for the vocoder mel-to-linear training.

    :param center: use center for CTC and Attention alignment, but not for GMM for RASR compatibility
    """
    # default: mfcc-40-dim
    feature_options_center = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features=FeatureType.DB_MEL_FILTERBANK,
        peak_normalization=False,
        preemphasis=0.97,
        sample_rate=16000,
        feature_options=DBMelFilterbankOptions(
            fmin=60,
            fmax=7600,
            min_amp=1e-10,
            center=True,
        )
    )
    audio_datastream = AudioFeatureDatastream(
        available_for_inference=False, options=feature_options_center
    )

    ls100_ogg_zip = get_ls100_silence_preprocess_ogg_zip()
    train_segments, _ = get_librispeech_tts_segments()

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        [ls100_ogg_zip],
        segment_file=train_segments,
        use_scalar_only=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_DATA_ROOT,
        alias_path=DATA_PREFIX + "ls100/",
    )
    if center == False:
        # take the normalization from center=True so that everything is compatible
        params = asdict(feature_options_center)
        params.pop("feature_options")
        feature_options_no_center = ReturnnAudioFeatureOptions(
            **params,
            feature_options=DBMelFilterbankOptions(
                fmin=60,
                fmax=7600,
                min_amp=1e-10,
                center=False,
            )
        )
        audio_datastream_no_center = AudioFeatureDatastream(
            available_for_inference=False, options=feature_options_no_center
        )
        audio_datastream_no_center.additional_options[
            "norm_mean"
        ] = audio_datastream.additional_options["norm_mean"]
        audio_datastream_no_center.additional_options[
            "norm_std_dev"
        ] = audio_datastream.additional_options["norm_std_dev"]

        return audio_datastream_no_center

    return audio_datastream


def get_lexicon(with_blank: bool = False) -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_librispeech_lexicon()
    lexicon = extend_lexicon_with_tts_lemmas(lexicon)
    if with_blank:
        lexicon = extend_lexicon_with_blank(lexicon)
    return lexicon



def get_vocab_datastream(with_blank: bool = False) -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

    :param with_blank: datastream for CTC training
    """
    lexicon = get_lexicon(with_blank)
    blacklist =  {"[SILENCE]"}
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon, blacklist=blacklist)
    name = "returnn_vocab_from_lexicon_with_blank" if with_blank else "returnn_vocab_from_lexicon"
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, name))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream





