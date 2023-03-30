from dataclasses import asdict
from functools import lru_cache
import os
from sisyphus import tk
from typing import List

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict, get_bliss_corpus_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.rossenbach.common_setups.returnn import datasets

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_tts_segments, get_ls_train_clean_100_tts_silencepreprocessed
from i6_experiments.users.rossenbach.setups.tts.preprocessing import (
    process_corpus_text_with_extended_lexicon,
    extend_lexicon_with_tts_lemmas,
    extend_lexicon_with_blank
)

from .default_tools import RETURNN_EXE, RETURNN_ROOT

DATA_PREFIX = "experiments/alignment_analysis_tts/data/"


@lru_cache
def get_librispeech_lexicon(corpus_key="train-clean-100") -> tk.Path:
    """
    get the TTS-extended g2p bliss lexicon with [start], [end] and [space] marker
    :return:
    """
    return extend_lexicon_with_tts_lemmas(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)[corpus_key])


@lru_cache
def get_ls100_silence_preprocessed_bliss() -> tk.Path:
    """
    Get the modified ls100 corpus for the TTS task
    :return: Bliss xml file
    """
    # this is the FFmpeg silence preprocessed version of LibriSpeech train-clean-100
    sil_pp_train_clean_100_co = get_ls_train_clean_100_tts_silencepreprocessed()

    # convert the corpus transcriptions into phoneme and marker representation
    sil_pp_train_clean_100_tts = process_corpus_text_with_extended_lexicon(
        bliss_corpus=sil_pp_train_clean_100_co.corpus_file,
        lexicon=get_librispeech_lexicon())

    return sil_pp_train_clean_100_tts


@lru_cache
def get_ls360_zip_for_synthesis_only() -> tk.Path:
    """
    TTS label processed librispeech 360 without audio

    :return:
    """
    ls460_lexicon = extend_lexicon_with_tts_lemmas(get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False)["train-clean-460"])
    corpus = get_bliss_corpus_dict()["train-clean-360"]  # original corpus as .flac
    tts_corpus = process_corpus_text_with_extended_lexicon(
        bliss_corpus=corpus,
        lexicon=ls460_lexicon
    )
    zip_dataset = BlissToOggZipJob(
        bliss_corpus=tts_corpus,
        no_audio=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    ).out_ogg_zip

    return zip_dataset


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
        returnn_root=RETURNN_ROOT,
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
        data_map={'audio_samples': ('audio', 'data'),
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


from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import ReturnnAudioRawOptions, AudioRawDatastream

@lru_cache()
def get_audio_raw_datastream():
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=False, preemphasis=0.97)
    )
    return audio_datastream


def get_lexicon(with_blank: bool = False, corpus_key="train-clean-100") -> tk.Path:
    """
    Get the TTS/CTC lexicon

    :param with_blank: add blank (e.g. for CTC training or extraction)
    :return: path to bliss lexicon file
    """
    lexicon = get_librispeech_lexicon(corpus_key=corpus_key)
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





