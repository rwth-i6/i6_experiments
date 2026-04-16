"""
Dataset helpers for the BPE-based training
"""
from sisyphus import tk
from typing import List, Optional

from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob

from i6_experiments.common.datasets.loquacious.corpus import get_ogg_zip_dict
from i6_experiments.common.datasets.loquacious.vocab import get_subword_nmt_bpe
from i6_experiments.common.datasets.loquacious.lexicon import get_bliss_lexicon
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.hilmes.data.loquacious import get_dev_segments

from .common import DatasetSettings, TrainingDatasets, build_training_datasets
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE, SUBWORD_NMT_REPO


def get_bpe_datastream(loquacious_key: str, bpe_size: int, is_recog: bool, use_postfix: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    :param use_postfix: True for RNN-T or Attention, False for CTC
    """
    bpe_settings = get_subword_nmt_bpe(corpus_key=loquacious_key, bpe_size=bpe_size, unk_label="<unk>")

    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog,
        seq_postfix=0 if use_postfix else None,
    )
    return bpe_targets


def get_bpe_lexicon(loquacious_key: str, bpe_size: int, variant=1) -> tk.Path:
    """
    Create BPE lexicon without unknown and silence

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon bliss xml file
    """
    bpe_settings = get_subword_nmt_bpe(corpus_key=loquacious_key, bpe_size=bpe_size, unk_label="<unk>")
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False, variant=variant),
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon

    return bpe_lexicon


def get_text_lexicon(prefix: str, loquacious_key: str, bpe_size: int, variant=1) -> tk.Path:
    """
    Get a bpe lexicon in line-based text format to be used for torchaudio/Flashlight decoding

    :param prefix:
    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon text file
    """
    bliss_lex = get_bpe_lexicon(loquacious_key=loquacious_key, bpe_size=bpe_size, variant=1)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_custom_bpe_lexicon(bliss_lexicon, bpe_codes, bpe_vocab):
    """

    :param bliss_lexicon:
    :param bpe_codes:
    :param bpe_vocab:
    :return:
    """
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=bliss_lexicon,
        bpe_codes=bpe_codes,
        bpe_vocab=bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bpe_lexicon,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon



def build_bpe_training_datasets(
    prefix: str,
    loquacious_key: str,
    bpe_size: int,
    settings: DatasetSettings,
    use_postfix: bool,
    extra_train_ogg_zips: Optional[List[tk.Path]] = None,
    data_repetition_factors: Optional[List[int]] = None,
) -> TrainingDatasets:
    """

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :param settings: configuration object for the dataset pipeline
    :param use_postfix: True for RNN-T or Attention, False for CTC
    :param extra_train_ogg_zips: add additional ogg zips for training, e.g. created by `synthetic_librispeech_bliss_to_ogg_zip`
    :param data_repetition_factors: list if integers, first entry is for the original librispeech data
    """
    label_datastream = get_bpe_datastream(
        loquacious_key=loquacious_key, bpe_size=bpe_size, is_recog=True, use_postfix=use_postfix
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[loquacious_key]

    if extra_train_ogg_zips is None:
        ogg_zips = train_ogg
    else:
        assert data_repetition_factors, "please provide repetition factors if you provide extra ogg zips"
        assert len(extra_train_ogg_zips) + 1 == len(data_repetition_factors)
        ogg_zips = [train_ogg] * data_repetition_factors[0]
        for ogg_zip, repetition in zip(extra_train_ogg_zips, data_repetition_factors[1:]):
            ogg_zips += [ogg_zip] * repetition

    dev_ogg = ogg_zip_dict["dev.all"]
    dev_segments = get_dev_segments()

    return build_training_datasets(
        train_ogg=ogg_zips,
        dev_ogg=dev_ogg,
        dev_segments=dev_segments,
        settings=settings,
        label_datastream=label_datastream,
    )


def get_bpe_bliss_lexicon(bpe_size: int, add_blank: bool, loquacious_key: str,) -> tk.Path:
    bpe_settings = get_subword_nmt_bpe(bpe_size=bpe_size, corpus_key=loquacious_key, unk_label="<unk>")
    lexicon = get_bliss_lexicon(add_unknown_phoneme_and_mapping=True, add_silence=True)
    bpe_lexicon_file = CreateBPELexiconJob(
        base_lexicon_path=lexicon,
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        unk_label="<unk>",
        vocab_blacklist=["</s>"],
        subword_nmt_repo=SUBWORD_NMT_REPO,
        keep_special_lemmas=False,
    ).out_lexicon

    from i6_core.lib.lexicon import Lemma, Lexicon
    from i6_core.lexicon.modification import MergeLexiconJob, WriteLexiconJob

    lexicon_ext = Lexicon()
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-BEGIN]"], phon=["<s>"], synt=["<s>"], special="sentence-begin"))
    lexicon_ext.add_lemma(Lemma(orth=["[SENTENCE-END]"], phon=["<s>"], synt=["</s>"], special="sentence-end"))
    lexicon_ext.add_lemma(Lemma(orth=["[UNKNOWN]"], phon=["<unk>"], synt=["<UNK>"], special="unknown"))

    if add_blank:
        lexicon_ext.add_lemma(Lemma(orth=["[BLANK]"], phon=["<blank>"], special="blank"))
        lexicon_ext.add_phoneme("<blank>", variation="none")

    lexicon_ext_file = WriteLexiconJob(lexicon_ext).out_bliss_lexicon

    return MergeLexiconJob([bpe_lexicon_file, lexicon_ext_file]).out_bliss_lexicon

