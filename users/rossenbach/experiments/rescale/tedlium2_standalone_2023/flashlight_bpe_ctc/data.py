"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from i6_core.returnn import CodeWrapper

from i6_experiments.common.datasets.tedlium2.corpus import get_ogg_zip_dict
from i6_experiments.common.datasets.tedlium2.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.datasets.tedlium2.lexicon import get_bliss_lexicon
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.rossenbach.lexicon.bpe_lexicon import CreateBPELexiconJob

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset

from ..data import build_training_datasets, TrainingDatasetSettings, TrainingDatasets

from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE


from ..data import DATA_PREFIX

def get_lexicon(bpe_size: int) -> tk.Path:
    subword_nmt_repo = get_returnn_subword_nmt(
        commit_hash = "5015a45e28a958f800ef1c50e7880c0c9ef414cf",
        output_prefix=DATA_PREFIX
    )
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"

    bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=get_bliss_lexicon(add_unknown_phoneme_and_mapping=False, add_silence=False),
        bpe_codes=bpe_datastream.codes,
        bpe_vocab=bpe_datastream.vocab,
        subword_nmt_repo=subword_nmt_repo,
        unk_label="<unk>"
    ).out_lexicon

    return bpe_lexicon


def get_text_lexicon(bpe_size: int) -> tk.Path:
    """

    :return:
    """
    bliss_lex = get_lexicon(bpe_size=bpe_size)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def get_bpe_datastream(bpe_size: int, is_recog: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param librispeech_key:
    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    :param use_v2: subword_nmt had a bug where it would not find python, use corrected version which changes hash
    """
    bpe_settings = get_subword_nmt_bpe_v2(bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


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