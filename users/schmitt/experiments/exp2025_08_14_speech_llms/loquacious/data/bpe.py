"""
Dataset helpers for the BPE-based training
"""
from sisyphus import tk

from i6_core.g2p.convert import BlissLexiconToG2PLexiconJob
from i6_core.lexicon.bpe import CreateBPELexiconJob

from i6_experiments.common.datasets.loquacious.corpus import get_ogg_zip_dict
from i6_experiments.common.datasets.loquacious.vocab import get_subword_nmt_bpe
from i6_experiments.common.setups.returnn.datastreams.vocabulary import BpeDatastream

from .common import DatasetSettings, TrainingDatasets, build_training_datasets
from .cv_segments import get_dev_segments
from ...default_tools import RETURNN_ROOT, RETURNN_EXE, SUBWORD_NMT_REPO


def get_bpe_datastream(loquacious_key: str, bpe_size: int, is_recog: bool, use_postfix: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param loquacious_key: which loquacious corpus to use for bpe training
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


def build_bpe_training_datasets(
    prefix: str,
    loquacious_key: str,
    bpe_size: int,
    settings: DatasetSettings,
    use_postfix: bool,
) -> TrainingDatasets:
    """

    :param loquacious_key: which librispeech corpus to use for bpe training
    :param bpe_size: number of BPE splits
    :param settings: configuration object for the dataset pipeline
    :param use_postfix: True for RNN-T or Attention, False for CTC
    """
    # we set is_recog=True so that it uses the UNK label for training
    # otherwise, we get a key error for some CV seqs because the BPE split is different than in the train data
    label_datastream = get_bpe_datastream(
        loquacious_key=loquacious_key, bpe_size=bpe_size, is_recog=True, use_postfix=use_postfix
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[loquacious_key]
    dev_ogg = ogg_zip_dict["dev.all"]
    dev_segments = get_dev_segments()

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_ogg=dev_ogg,
        dev_segments=dev_segments,
        settings=settings,
        label_datastream=label_datastream,
    )
