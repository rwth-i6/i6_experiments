"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Dict, List, Optional, Tuple

from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob

from i6_experiments.common.datasets.librispeech import (
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_corpus_dict,
    get_ogg_zip_dict,
    get_bliss_lexicon,
)

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream


from .common import get_zip, DATA_PREFIX, build_training_datasets, TrainingDatasets, TrainingDatasetSettings


def get_eow_lexicon(librispeech_key: str, with_g2p=True) -> tk.Path:

    """
    get the g2p bliss lexicon with EOW tokens added
    :return:
    """
    if with_g2p:
        lex = get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=False, add_silence=False, output_prefix="librispeech_g2p_datasets"
        )[librispeech_key]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=False, output_prefix="librispeech_datasets")

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_bliss(librispeech_key: str, train_librispeech_key: str, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param corpus_key: train, dev, test
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="ogg")[librispeech_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            bliss_lexicon=get_eow_lexicon(
                librispeech_key=train_librispeech_key, with_g2p=True
            ),  # cv may include words from g2p
            all_unknown=False,
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(librispeech_key=train_librispeech_key, with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(librispeech_key: str, train_librispeech_key: str, remove_unk_seqs=False):
    """
    :param corpus_key: e.g. "train", "dev", or "test,
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(
        librispeech_key=librispeech_key, train_librispeech_key=train_librispeech_key, remove_unk_seqs=remove_unk_seqs
    )
    zip_dataset = get_zip(f"{librispeech_key}_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream(librispeech_key: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream for Tedlium-2

    :param with_blank: datastream for CTC training
    """
    lexicon = get_eow_lexicon(librispeech_key=librispeech_key)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, f"{librispeech_key}", "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_text_lexicon(librispeech_key: str) -> tk.Path:
    """

    :return:
    """
    bliss_lex = get_eow_lexicon(librispeech_key=librispeech_key, with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon

    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def build_eow_phon_training_datasets(
    librispeech_key: str,
    settings: TrainingDatasetSettings,
) -> TrainingDatasets:
    """
    :param settings: configuration object for the dataset pipeline
    """
    label_datastream = get_eow_vocab_datastream(librispeech_key=librispeech_key)

    _, train_ogg = get_eow_bliss_and_zip(
        librispeech_key=librispeech_key, train_librispeech_key=librispeech_key, remove_unk_seqs=False
    )
    _, dev_clean_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-clean", train_librispeech_key=librispeech_key, remove_unk_seqs=True
    )
    _, dev_other_ogg = get_eow_bliss_and_zip(
        librispeech_key="dev-other", train_librispeech_key=librispeech_key, remove_unk_seqs=True
    )

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
    )
