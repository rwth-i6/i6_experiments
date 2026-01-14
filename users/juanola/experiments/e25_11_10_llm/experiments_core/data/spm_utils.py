"""
Dataset helpers for the SPM-based training
"""
from typing import Optional, Dict

from sisyphus import tk

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .dataset_commons import build_lm_training_datasets
from .librispeech_utils import get_librispeech_spm_datastream, get_librispeech_train_corpus_text, \
    get_librispeech_normalized_lm_data, get_librispeech_lm_combined_txt
from ...default_tools import RETURNN_ROOT, RETURNN_EXE

"""
NEW LM CODE - mostly from Alberts librispeech file
"""


def build_spm_lm_training_datasets(
        prefix: str,
        librispeech_key: str,
        vocab_size: int,
        return_settings: ReturnnDatasetSettings,
        returnn_root: tk.Path = RETURNN_ROOT,
        alpha: Optional[float] = None,
        use_train_corpus_text: bool = True,
        use_normalized_lm_data: bool = False,
) -> TrainingDatasets:
    """
    Builds the training datasets for the SPM-based training - For LM.
    """
    label_datastream = get_librispeech_spm_datastream(
        vocab_size, use_train_corpus_text, use_normalized_lm_data
    )

    if use_train_corpus_text and not use_normalized_lm_data:
        training_text = get_librispeech_train_corpus_text()
    elif not use_train_corpus_text and use_normalized_lm_data:
        training_text = get_librispeech_normalized_lm_data()
    elif use_train_corpus_text and use_normalized_lm_data:
        training_text = get_librispeech_lm_combined_txt()
    else:
        raise ValueError("At least one corpus is needed for SPM training.")

    training_datasets = build_lm_training_datasets(
        text_file=training_text,
        label_datastream=label_datastream,
        returnn_settings=return_settings,
        alpha=alpha,
    )

    return training_datasets
