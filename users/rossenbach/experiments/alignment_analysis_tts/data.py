from functools import lru_cache
import os
from sisyphus import tk
from typing import List

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import AudioFeatureDatastream, DBMelFilterbankOptions, ReturnnAudioFeatureOptions, FeatureType
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_ls_train_clean_100_tts_silencepreprocessed
from i6_experiments.users.rossenbach.setups.tts.preprocessing import process_corpus_text_with_extended_lexicon, extend_lexicon

from .default_tools import RETURNN_EXE, RETURNN_DATA_ROOT

DATA_PREFIX = "experiments/alignment_analysis_tts/data/"


@lru_cache
def get_librispeech_lexicon() -> tk.Path:
    """

    :return:
    """
    return extend_lexicon(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=True)["train-clean-100"])


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


def get_tts_log_mel_datastream(
        center: bool = False,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param statistics_ogg_zip: ogg zip file(s) of the training corpus for statistics
    :param returnn_python_exe:
    :param returnn_root:
    :param alias_path:
    """
    # default: mfcc-40-dim
    feature_options = ReturnnAudioFeatureOptions(
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
            center=center,
        )
    )
    audio_datastream = AudioFeatureDatastream(
        available_for_inference=False, options=feature_options
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
    return audio_datastream


def get_vocab_datastream() -> LabelDatastream:
    """
    Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

    :param alias_path:
    :return:
    :rtype: VocabularyDatastream
    """
    lexicon = get_librispeech_lexicon()
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, "returnn_vocab_from_lexicon"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream





