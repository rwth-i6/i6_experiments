"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk
from functools import lru_cache
from typing import Dict, List, Optional, Tuple


from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.vocab import get_subword_nmt_bpe_v2
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream
from i6_experiments.users.rossenbach.lexicon.bpe_lexicon import CreateBPELexiconJob

from .common import TrainingDatasetSettings, TrainingDatasets, build_training_datasets, DATA_PREFIX
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_EXE


@lru_cache()
def get_bpe_datastream(librispeech_key: str, bpe_size: int, is_recog: bool) -> BpeDatastream:
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param librispeech_key:
    :param bpe_size: size for the bpe labels
    :param is_recog: removes the UNK label when not in training
    """
    bpe_settings = get_subword_nmt_bpe_v2(corpus_key=librispeech_key, bpe_size=bpe_size, unk_label="<unk>")

    # TODO: Try without sequence postfix (seq_postfix=None)
    # otherwise every sequence gets a <s> at the end
    bpe_targets = BpeDatastream(available_for_inference=False, bpe_settings=bpe_settings, use_unk_label=is_recog)
    return bpe_targets


def get_lexicon(librispeech_key: str, bpe_size: int) -> tk.Path:
    subword_nmt_repo = get_returnn_subword_nmt(
        commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf", output_prefix=DATA_PREFIX
    )
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"

    bpe_datastream = get_bpe_datastream(librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=False)
    bpe_lexicon = CreateBPELexiconJob(
        base_lexicon_path=get_bliss_lexicon(
            add_unknown_phoneme_and_mapping=False, add_silence=False, output_prefix="librispeech_datasets"
        ),
        bpe_codes=bpe_datastream.codes,
        bpe_vocab=bpe_datastream.vocab,
        subword_nmt_repo=subword_nmt_repo,
        unk_label="<unk>",
    ).out_lexicon

    return bpe_lexicon


def get_text_lexicon(librispeech_key: str, bpe_size: int) -> tk.Path:
    """

    :return:
    """
    bliss_lex = get_lexicon(librispeech_key=librispeech_key, bpe_size=bpe_size)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon

    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def build_bpe_training_datasets(
    librispeech_key: str,
    bpe_size: int,
    settings: TrainingDatasetSettings,
) -> TrainingDatasets:
    """
    :param settings: configuration object for the dataset pipeline
    """
    label_datastream = get_bpe_datastream(librispeech_key=librispeech_key, bpe_size=bpe_size, is_recog=False)

    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=MINI_RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict["dev-clean"]
    dev_other_ogg = ogg_zip_dict["dev-other"]

    return build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
    )
