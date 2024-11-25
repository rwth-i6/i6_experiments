"""
Dataset helpers for the EOW-augmented phoneme training
"""
from sisyphus import tk

import os
from typing import Optional, Tuple

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


def get_lexicon(
    g2p_librispeech_key: Optional[str], with_g2p: bool, add_eow_phonemes: bool, add_silence: bool
) -> tk.Path:
    """
    get the g2p bliss lexicon (optionally with EOW tokens added)

    :param g2p_librispeech_key: which librispeech to use as baseline data
    :param with_g2p:
    :return: phoneme based bliss lexicon
    """
    if with_g2p:
        assert g2p_librispeech_key is not None
        lex = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=False, add_silence=add_silence)[
            g2p_librispeech_key
        ]
    else:
        lex = get_bliss_lexicon(use_stress_marker=False, add_silence=add_silence)

    if add_eow_phonemes:
        return AddEowPhonemesToLexiconJob(lex).out_lexicon
    else:
        return lex


def get_bliss(
    librispeech_key: str,
    g2p_librispeech_key: str,
    add_eow_phonemes: bool,
    add_silence: bool,
    remove_unk_seqs: bool = False,
    apply_lexicon: bool = True,
) -> tk.Path:
    """
    get an optionally EOW modified corpus with optional unknown removed for cross validation

    :param librispeech_key: which bliss dataset to "get"
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :param apply_lexicon: convert corpus word sequences to phoneme sequences
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="ogg")[librispeech_key]

    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            # cv may include words from g2p
            bliss_lexicon=get_lexicon(
                g2p_librispeech_key=g2p_librispeech_key,
                with_g2p=True,
                add_eow_phonemes=add_eow_phonemes,
                add_silence=add_silence,
            ),
            all_unknown=False,
        ).out_corpus

    if apply_lexicon:
        # default train lexicon
        lexicon = get_lexicon(
            g2p_librispeech_key=g2p_librispeech_key,
            with_g2p=True,
            add_eow_phonemes=add_eow_phonemes,
            add_silence=add_silence,
        )
        bliss = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return bliss


def get_bliss_and_zip(
    prefix: str,
    librispeech_key: str,
    g2p_librispeech_key: str,
    add_eow_phonemes: bool,
    add_silence: bool,
    remove_unk_seqs: bool = False,
    apply_lexicon: bool = True,
):
    """
    :param librispeech_key: which bliss dataset to "get"
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :param apply_lexicon: convert corpus word sequences to phoneme sequences
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_bliss(
        librispeech_key=librispeech_key,
        g2p_librispeech_key=g2p_librispeech_key,
        add_eow_phonemes=add_eow_phonemes,
        add_silence=add_silence,
        remove_unk_seqs=remove_unk_seqs,
        apply_lexicon=apply_lexicon,
    )
    zip_dataset = get_zip(
        prefix,
        f"{g2p_librispeech_key}_{librispeech_key}{'_filtered' if remove_unk_seqs else ''}{'_eow' if add_eow_phonemes else ''}{'_sil' if add_silence else ''}{'_lexapplied' if apply_lexicon else ''}",
        bliss_dataset=bliss_dataset,
    )

    return bliss_dataset, zip_dataset


def get_vocab_datastream(
    prefix: str, g2p_librispeech_key: str, add_eow_phonemes: bool, add_silence: bool
) -> LabelDatastream:
    """
    Phoneme with(out) EOW LabelDatastream

    :param prefix:
    :param g2p_librispeech_key: baseline librispeech dataset that is used for g2p
    """
    lexicon = get_lexicon(
        g2p_librispeech_key=g2p_librispeech_key,
        with_g2p=True,
        add_eow_phonemes=add_eow_phonemes,
        add_silence=add_silence,
    )
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon)
    returnn_vocab_job.add_alias(
        os.path.join(
            prefix,
            f"{g2p_librispeech_key}",
            f"{'eow_' if add_eow_phonemes else ''}{'sil_' if add_silence else ''}returnn_vocab_job",
        )
    )

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size,
        unk_label="[UNKNOWN]",
    )

    return vocab_datastream


def get_text_lexicon() -> tk.Path:
    """
    :return: Text lexicon for the Flashlight decoder
    """
    bliss_lex = get_lexicon(g2p_librispeech_key=None, with_g2p=False, add_eow_phonemes=True, add_silence=False)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_phon_training_datasets(
    prefix: str,
    librispeech_key: str,
    settings: DatasetSettings,
    add_eow_phonemes: bool,
    add_silence: bool,
    lexicon_librispeech_key: Optional[str] = None,
    use_tags: bool = False,
    apply_lexicon: bool = True,
    set_target_opts: bool = True,
) -> Tuple[TrainingDatasets, dict[str, tk.Path]]:
    """
    :param prefix:
    :param librispeech_key: which librispeech corpus to use
    :param settings: configuration object for the dataset pipeline
    :param lexicon_librispeech_key: if we are using extra synthetic data, we might want a lexicon with the OOV coverage of that data as well
    :param use_tags: Use sequence tag instead of labels in training
    :param apply_lexicon: convert corpus word sequences to phoneme sequences
    """
    label_datastream = get_vocab_datastream(
        prefix=prefix,
        g2p_librispeech_key=lexicon_librispeech_key or librispeech_key,
        add_eow_phonemes=add_eow_phonemes,
        add_silence=add_silence,
    )

    train_bliss, train_ogg = get_bliss_and_zip(
        prefix=prefix,
        librispeech_key=librispeech_key,
        g2p_librispeech_key=librispeech_key,
        add_eow_phonemes=add_eow_phonemes,
        add_silence=add_silence,
        remove_unk_seqs=False,
        apply_lexicon=apply_lexicon,
    )
    dev_clean_bliss, dev_clean_ogg = get_bliss_and_zip(
        prefix=prefix,
        librispeech_key="dev-clean",
        g2p_librispeech_key=librispeech_key,
        add_eow_phonemes=add_eow_phonemes,
        add_silence=add_silence,
        remove_unk_seqs=True,
        apply_lexicon=apply_lexicon,
    )
    dev_other_bliss, dev_other_ogg = get_bliss_and_zip(
        prefix=prefix,
        librispeech_key="dev-other",
        g2p_librispeech_key=librispeech_key,
        add_eow_phonemes=add_eow_phonemes,
        add_silence=add_silence,
        remove_unk_seqs=True,
        apply_lexicon=apply_lexicon,
    )

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
        use_tags=use_tags,
        set_target_opts=set_target_opts,
    ), {librispeech_key: train_bliss, "dev-clean": dev_clean_bliss, "dev-other": dev_other_bliss}
