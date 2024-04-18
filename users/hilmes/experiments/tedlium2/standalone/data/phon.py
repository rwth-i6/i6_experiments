"""
Dataset helpers for the EOW-augmented phoneme training
"""
from sisyphus import tk

import os
from typing import Optional

from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory

from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.lexicon import get_bliss_lexicon, get_g2p_augmented_bliss_lexicon
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from .common import get_zip, build_training_datasets, TrainingDatasets, DatasetSettings


def get_eow_lexicon(with_g2p: bool) -> tk.Path:
    """
    get the g2p bliss lexicon with EOW tokens added
    :param with_g2p:
    :return: phoneme based bliss lexicon
    """
    if with_g2p:
        lex = get_g2p_augmented_bliss_lexicon(output_prefix="tedliumv2_g2p_datasets", add_silence=False)
    else:
        lex = get_bliss_lexicon(output_prefix="tedliumv2_eow_datasets", add_silence=False)

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_bliss(corpus_key: str, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param corpus_key: train, dev, test
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="ogg")[corpus_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            # cv may include words from g2p
            bliss_lexicon=get_eow_lexicon(with_g2p=True), # TODO
            all_unknown=False,
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(corpus_key: str, remove_unk_seqs=False):
    """
    :param corpus_key: which bliss dataset to "get"
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(corpus_key=corpus_key, remove_unk_seqs=remove_unk_seqs)
    zip_dataset = get_zip(f"{corpus_key}_filtered_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream(prefix: str) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream

    :param prefix:
    """
    lexicon = get_eow_lexicon(with_g2p=True)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_text_lexicon() -> tk.Path:
    """
    :return: Text lexicon for the Flashlight decoder
    """
    bliss_lex = get_eow_lexicon(with_g2p=False)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_eow_phon_training_datasets(
    prefix: str,
    settings: DatasetSettings,
) -> TrainingDatasets:
    """
    :param prefix:
    :param settings: configuration object for the dataset pipeline
    """
    label_datastream = get_eow_vocab_datastream(prefix=prefix)

    _, train_ogg = get_eow_bliss_and_zip(corpus_key="train", remove_unk_seqs=False)
    _, dev_ogg = get_eow_bliss_and_zip(corpus_key="dev", remove_unk_seqs=True)

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_ogg=dev_ogg,
        settings=settings,
        label_datastream=label_datastream,
    )
