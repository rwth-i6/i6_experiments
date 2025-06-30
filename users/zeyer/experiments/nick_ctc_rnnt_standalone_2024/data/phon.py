"""
Dataset helpers for the EOW-augmented phoneme training
"""
from sisyphus import tk

import os
from typing import List, Optional

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_corpus_dict,
    get_bliss_lexicon,
)
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from .common import get_zip, build_training_datasets, TrainingDatasets, DatasetSettings


def get_eow_lexicon(g2p_librispeech_key: Optional[str], with_g2p: bool) -> tk.Path:
    """
    get the g2p bliss lexicon with EOW tokens added

    :param g2p_librispeech_key: which librispeech to use as baseline data
    :param with_g2p:
    :return: phoneme based bliss lexicon
    """
    if with_g2p:
        assert g2p_librispeech_key is not None
        lex = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=False)[g2p_librispeech_key]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=False)

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_bliss(librispeech_key: str, g2p_librispeech_key: str, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param librispeech_key: which bliss dataset to "get"
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="ogg")[librispeech_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            # cv may include words from g2p
            bliss_lexicon=get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True),
            all_unknown=False,
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(librispeech_key: str, g2p_librispeech_key: str, remove_unk_seqs=False):
    """
    :param librispeech_key: which bliss dataset to "get"
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(
        librispeech_key=librispeech_key, g2p_librispeech_key=g2p_librispeech_key, remove_unk_seqs=remove_unk_seqs
    )
    zip_dataset = get_zip(f"{g2p_librispeech_key}_{librispeech_key}_filtered_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream(prefix: str, g2p_librispeech_key: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream

    :param prefix:
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    """
    lexicon = get_eow_lexicon(g2p_librispeech_key=g2p_librispeech_key, with_g2p=True)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{g2p_librispeech_key}", "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_text_lexicon() -> tk.Path:
    """
    :return: Text lexicon for the Flashlight decoder
    """
    bliss_lex = get_eow_lexicon(g2p_librispeech_key=None, with_g2p=False)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def synthetic_librispeech_bliss_to_ogg_zip(
        prefix: str,
        bliss: str,
        lexicon_librispeech_key,
    ) -> tk.Path:
    """

    :param bliss: path to the bliss to generate ogg zip
    :param lexicon_librispeech_key: lexicon coverage needed for the synthetic data (train-clean-100, 460 etc...)
    :return:
    """
    lexicon = get_eow_lexicon(g2p_librispeech_key=lexicon_librispeech_key, with_g2p=True)
    converted_bliss = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus
    ogg_zip = get_zip(alias_name=prefix + "/syn_ogg_zip", bliss_dataset=converted_bliss)
    return ogg_zip


def build_eow_phon_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: DatasetSettings,
    lexicon_librispeech_key: Optional[str] = None,
    extra_train_ogg_zips: Optional[List[tk.Path]] = None,
    data_repetition_factors: Optional[List[int]] = None,
) -> TrainingDatasets:
    """
    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    :param lexicon_librispeech_key: if we are using extra synthetic data, we might want a lexicon with the OOV coverage of that data as well
    :param extra_train_ogg_zips: add additional ogg zips for training, e.g. created by `synthetic_librispeech_bliss_to_ogg_zip`
    :param data_repetition_factors: list if integers, first entry is for the original librispeech data
    """
    label_datastream = get_eow_vocab_datastream(
        prefix=prefix, g2p_librispeech_key=lexicon_librispeech_key or librispeech_key
    )

    _, train_ogg = get_eow_bliss_and_zip(
        librispeech_key=librispeech_key, g2p_librispeech_key=librispeech_key, remove_unk_seqs=False
    )
    _, dev_clean_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-clean", g2p_librispeech_key=librispeech_key, remove_unk_seqs=True
    )
    _, dev_other_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-other", g2p_librispeech_key=librispeech_key, remove_unk_seqs=True
    )
    if extra_train_ogg_zips is None:
        ogg_zips = train_ogg
    else:
        assert data_repetition_factors, "please provide repetition factors if you provide extra ogg zips"
        assert len(extra_train_ogg_zips) + 1 == len(data_repetition_factors)
        ogg_zips = [train_ogg] * data_repetition_factors[0]
        for ogg_zip, repetition in zip(extra_train_ogg_zips, data_repetition_factors[1:]):
            ogg_zips += [ogg_zip] * repetition

    return build_training_datasets(
        train_ogg=ogg_zips,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
    )
