"""
Dataset helpers for the BPE-based training
"""
from sisyphus import tk

from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream

from .common import DatasetSettings, TrainingDatasets, build_training_datasets
from ..train_util import TrainingDatasetSettings
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE, SUBWORD_NMT_REPO


def get_bpe_datastream(librispeech_key: str, bpe_size: int, is_recog: bool, use_postfix: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    :param use_postfix: True for RNN-T or Attention, False for CTC
    """
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="<unk>")

    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog,
        seq_postfix=0 if use_postfix else None,
    )
    return bpe_targets


def get_bpe_lexicon(librispeech_key: str, bpe_size: int) -> tk.Path:
    """
    Create BPE lexicon without unknown and silence

    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon bliss xml file
    """
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="<unk>")
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False),
        bpe_codes=bpe_settings.bpe_codes,
        bpe_vocab=bpe_settings.bpe_vocab,
        subword_nmt_repo=SUBWORD_NMT_REPO,
        unk_label="<unk>",
    ).out_lexicon

    return bpe_lexicon


def get_text_lexicon(prefix: str, librispeech_key: str, bpe_size: int) -> tk.Path:
    """
    Get a bpe lexicon in line-based text format to be used for torchaudio/Flashlight decoding

    :param prefix:
    :param librispeech_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :return: path to a lexicon text file
    """
    bliss_lex = get_bpe_lexicon(librispeech_key=librispeech_key, bpe_size=bpe_size)
    word_lexicon = BlissLexiconToG2PLexiconJob(
        bliss_lex,
        include_pronunciation_variants=True,
        include_orthography_variants=True,
    ).out_g2p_lexicon
    return word_lexicon


def build_bpe_training_datasets(
        bpe_size: int,
        settings: TrainingDatasetSettings,
) -> TrainingDatasets:
    """
    :param settings: configuration object for the dataset pipeline
    """
    label_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    ogg_zip_dict = get_ogg_zip_dict(
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT
    )
    train_ogg = ogg_zip_dict["train"]
    dev_ogg = ogg_zip_dict["dev"]

    return build_training_datasets(
        settings=settings,
        train_ogg=train_ogg,
        dev_ogg=dev_ogg,
        label_datastream=label_datastream
    )
