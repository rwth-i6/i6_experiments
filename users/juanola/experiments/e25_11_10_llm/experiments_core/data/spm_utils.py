"""
Dataset helpers for the SPM-based training
"""
from typing import Optional, Tuple

from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from .dataset_commons import build_lm_training_datasets
from .librispeech_lm_utils import get_librispeech_spm_datastream, get_librispeech_train_corpus_text, \
    get_librispeech_normalized_lm_data, get_librispeech_lm_combined_txt, get_cv_lm_text
from ...configurations.data.dataset_config import DatasetConfig
from ...default_tools import RETURNN_ROOT

"""
NEW LM CODE - mostly from Alberts librispeech file
"""


def build_spm_lm_training_datasets(
        prefix: str,
        librispeech_key: str,
        vocab_size: int,
        return_settings: ReturnnDatasetSettings,
        dataset_config: DatasetConfig,
        returnn_root: tk.Path = RETURNN_ROOT,
) -> Tuple[TrainingDatasets, SentencePieceDatastream]:
    """
    Builds the training datasets for the SPM-based training - For LM.
    """
    label_datastream = get_librispeech_spm_datastream(
        vocab_size, dataset_config.use_train_corpus_text, dataset_config.use_normalized_lm_data
    )

    # TRAIN DATA
    if dataset_config.use_train_corpus_text and not dataset_config.use_normalized_lm_data:
        training_text = get_librispeech_train_corpus_text()
    elif not dataset_config.use_train_corpus_text and dataset_config.use_normalized_lm_data:
        training_text = get_librispeech_normalized_lm_data()
    elif dataset_config.use_train_corpus_text and dataset_config.use_normalized_lm_data:
        training_text = get_librispeech_lm_combined_txt()
    else:
        raise ValueError("At least one corpus is needed for SPM training.")

    # VALIDATION DATA
    cv_text_file = get_cv_lm_text(n_lines=dataset_config.cv_n_lines)

    training_datasets = build_lm_training_datasets(
        train_text_file=training_text,
        cv_text_file=cv_text_file,
        label_datastream=label_datastream,
        returnn_settings=return_settings,
        alpha=dataset_config.sampling_alpha,
        dev_train_lines=dataset_config.devtrain_n_lines
    )

    return training_datasets, label_datastream
