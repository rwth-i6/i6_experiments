"""
Dataset helpers for the SPM-based training
"""
from typing import Optional, Dict

from sisyphus import tk

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt
from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from i6_experiments.users.zeyer.datasets.librispeech import _get_spm_vocab  # TODO: warning! external import!
from i6_experiments.users.zeyer.datasets.utils.spm import SentencePieceModel
from .dataset_commons import build_lm_training_datasets
from ...default_tools import RETURNN_ROOT, RETURNN_EXE


def get_spm_datastream(vocab_size: int) -> SentencePieceDatastream:
    """
    Returns the datastream for the spm labels.
    Computes the SPM vocabulary.

    :param vocab_size: the size of the vocabulary
    """
    spm_model: SentencePieceModel = _get_spm_vocab(dim=vocab_size)  # Where TrainSentencePieceJob is called
    return SentencePieceDatastream(
        available_for_inference=False,
        spm_model=spm_model.model_file,
        vocab_size=vocab_size,
    )# TODO: maybe pass down the spm model for more info?

def get_subword_repo():
    """
    This is a for now very ugly helper to get the same subword_nmt repo
    as the get_subword_nmt_bpe_v2 is using
    :return:
    """
    subword_nmt_repo = get_returnn_subword_nmt(
        commit_hash="5015a45e28a958f800ef1c50e7880c0c9ef414cf", output_prefix=""
    )
    # overwrite hash for future bugfixes, it is unlikely the logic will ever be changed
    subword_nmt_repo.hash_overwrite = "I6_SUBWORD_NMT_V2"
    return subword_nmt_repo

def build_spm_lm_training_datasets(
        prefix: str,
        librispeech_key: str,
        vocab_size: int,
        return_settings: ReturnnDatasetSettings,
        returnn_root: tk.Path = RETURNN_ROOT,
        alpha: Optional[float] = None,
) -> TrainingDatasets:
    """
    Builds the training datasets for the SPM-based training - For LM.
    """
    label_datastream = get_spm_datastream(vocab_size=vocab_size)

    ogg_zip_dict: Dict[str, tk.Path] = get_ogg_zip_dict(prefix, returnn_root=returnn_root,
                                                        returnn_python_exe=RETURNN_EXE)
    training_datasets = build_lm_training_datasets(
        train_ogg=ogg_zip_dict[librispeech_key],
        dev_clean_ogg=ogg_zip_dict["dev-clean"],
        dev_other_ogg=ogg_zip_dict["dev-other"],
        label_datastream=label_datastream,
        returnn_settings=return_settings,
        alpha=alpha,
    )

    return training_datasets
