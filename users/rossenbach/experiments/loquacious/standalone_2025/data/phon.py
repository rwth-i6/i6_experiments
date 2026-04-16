
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

from i6_experiments.common.datasets.loquacious.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.loquacious.lexicon import (
    get_g2p_augmented_bliss_lexicon_dict,
    get_bliss_lexicon,
)
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.datasets.loquacious import get_dev_segments

from .common import get_zip, build_training_datasets, TrainingDatasets, DatasetSettings
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT




def get_eow_lexicon(g2p_loquacious_key: Optional[str], with_g2p: bool, variant: int) -> tk.Path:
    """
    get the g2p bliss lexicon with EOW tokens added

    :param g2p_loquacious_key: which loquacious to use as baseline data
    :param with_g2p:
    :return: phoneme based bliss lexicon
    """
    if with_g2p:
        assert g2p_loquacious_key is not None
        lex = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=False, variant=variant)[g2p_loquacious_key]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=False, variant=variant)

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_bliss(loquacious_key: str, g2p_loquacious_key: str, remove_unk_seqs=False, variant: int = 1) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param loquacious_key: which bliss dataset to "get"
    :param g2p_loquacious_key: baseline loquacious dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict()[loquacious_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            # cv may include words from g2p
            bliss_lexicon=get_eow_lexicon(g2p_loquacious_key=g2p_loquacious_key, with_g2p=True, variant=variant),
            all_unknown=False,
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(g2p_loquacious_key=g2p_loquacious_key, with_g2p=True, variant=variant)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_bliss_and_zip(loquacious_key: str, g2p_loquacious_key: str, remove_unk_seqs=False, variant: int = 1):
    """
    :param loquacious_key: which bliss dataset to "get"
    :param g2p_loquacious_key: baseline loquacious dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(
        loquacious_key=loquacious_key, g2p_loquacious_key=g2p_loquacious_key, remove_unk_seqs=remove_unk_seqs, variant=variant
    )
    zip_dataset = get_zip(f"{g2p_loquacious_key}_{loquacious_key}_filtered_eow", bliss_dataset=bliss_dataset)

    return bliss_dataset, zip_dataset


def get_eow_vocab_datastream(prefix: str, g2p_loquacious_key: str, variant: int) -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream

    :param prefix:
    :param g2p_loquacious_key: baseline loquacious dataset that is used for g2p
    """
    lexicon = get_eow_lexicon(g2p_loquacious_key=g2p_loquacious_key, with_g2p=True, variant=variant)
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(os.path.join(prefix, f"{g2p_loquacious_key}", "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True, vocab=returnn_vocab_job.out_vocab, vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream

def synthetic_bliss_to_ogg_zip(
        prefix: str,
        bliss: str,
        lexicon_loq_key,
        variant: int,
        custom_lexicon=None,
        return_bliss=False,
    ) -> tk.Path:
    """

    :param bliss: path to the bliss to generate ogg zip
    :param lexicon_loq_key: lexicon coverage needed for the synthetic data (train-clean-100, 460 etc...)
    :return:
    """
    if custom_lexicon:
        lexicon = custom_lexicon
        assert lexicon_librispeech_key is None
    else:
        lexicon = get_eow_lexicon(g2p_loquacious_key=lexicon_loq_key, with_g2p=True, variant=variant)
    converted_bliss = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus
    ogg_zip = get_zip(alias_name=prefix + "/syn_ogg_zip", bliss_dataset=converted_bliss)
    return (ogg_zip, converted_bliss) if return_bliss else ogg_zip


def get_text_lexicon(variant: int) -> tk.Path:
    """
    :return: Text lexicon for the Flashlight decoder
    """
    bliss_lex = get_eow_lexicon(g2p_loquacious_key=None, with_g2p=False, variant=variant)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_eow_phon_training_datasets(
    prefix: str,
    loquacious_key: str,
    settings: DatasetSettings,
    lexicon_loquacious_key: Optional[str] = None,
    extra_train_ogg_zips: Optional[List[tk.Path]] = None,
    data_repetition_factors: Optional[List[int]] = None,
    variant: int = 1,
    explicit_devtrain=False,
    extra_train_bliss: Optional[List[tk.Path]] = None,
) -> TrainingDatasets:
    """
    :param prefix:
    :param loquacious_key: which sub corpus to use
    :param settings: configuration object for the dataset pipeline
    :param lexicon_loquacious_key: if we are using extra synthetic data, we might want a lexicon with the OOV coverage of that data as well
    :param extra_train_ogg_zips: add additional ogg zips for training, e.g. created by `synthetic_librispeech_bliss_to_ogg_zip`
    :param data_repetition_factors: list if integers, first entry is for the original librispeech data
    :param variant: lexicon variant for Loquacious, please remove this and use Huggingface whenever taking over this setup
    :param explicit_devtrain: make an explicit devtrain bliss, requires extra_train_bliss to be set
    """
    if explicit_devtrain and extra_train_ogg_zips is not None:
        assert extra_train_bliss is not None
        assert len(extra_train_bliss) == len(extra_train_ogg_zips)

    label_datastream = get_eow_vocab_datastream(
        prefix=prefix, g2p_loquacious_key=lexicon_loquacious_key or loquacious_key, variant=variant
    )

    train_bliss, train_ogg = get_eow_bliss_and_zip(
        loquacious_key=loquacious_key, g2p_loquacious_key=loquacious_key, remove_unk_seqs=False, variant=variant
    )
    _, dev_ogg = get_eow_bliss_and_zip(
        loquacious_key="dev.all", g2p_loquacious_key=loquacious_key, remove_unk_seqs=True, variant=variant
    )

    if extra_train_ogg_zips is None:
        ogg_zips = train_ogg
    else:
        assert data_repetition_factors, "please provide repetition factors if you provide extra ogg zips"
        assert len(extra_train_ogg_zips) + 1 == len(data_repetition_factors)
        ogg_zips = [train_ogg] * data_repetition_factors[0]
        for ogg_zip, repetition in zip(extra_train_ogg_zips, data_repetition_factors[1:]):
            ogg_zips += [ogg_zip] * repetition

    dev_segments = get_dev_segments()

    explicit_devtrain_ogg = None
    if explicit_devtrain:
        if extra_train_bliss:
            from i6_core.corpus.transform import MergeCorporaJob
            merge_corpora = [train_bliss] + extra_train_bliss
            devtrain_ref_bliss = MergeCorporaJob(merge_corpora, "devtrain").out_merged_corpus
        else:
            devtrain_ref_bliss = train_bliss

        from i6_core.corpus.filter import FilterCorpusBySegmentsJob
        from i6_core.corpus.segments import SegmentCorpusJob
        from i6_core.returnn.oggzip import BlissToOggZipJob
        # still 1 based nightmare
        all_segments = SegmentCorpusJob(devtrain_ref_bliss, 1).out_single_segment_files[1]
        from i6_experiments.users.rossenbach.segments.helper import shuffle_and_head
        devtrain_segments = shuffle_and_head(all_segments, num_lines=3000)
        devtrain_corpus = FilterCorpusBySegmentsJob(
            devtrain_ref_bliss,
            devtrain_segments,
            compressed=True,
            delete_empty_recordings=True
        ).out_corpus
        zip_dataset_job = BlissToOggZipJob(
            bliss_corpus=devtrain_corpus,
            no_conversion=True,
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        zip_dataset_job.rqmt = {"cpu": 16, "mem": 16, "time": 24}
        explicit_devtrain_ogg = zip_dataset_job.out_ogg_zip

    return build_training_datasets(
        train_ogg=ogg_zips,
        dev_ogg=dev_ogg,
        dev_segments=dev_segments,
        settings=settings,
        label_datastream=label_datastream,
        explicit_devtrain_ogg=explicit_devtrain_ogg,
    )
