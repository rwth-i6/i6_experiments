"""
Dataset helpers for the SPM-based training
"""
from typing import Tuple


from sisyphus import tk
from i6_core.corpus import CorpusToTxtJob
from i6_core.text.info import CountLinesJob
from i6_core.text.label.sentencepiece.apply import ApplySentencepieceToTextJob
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from sisyphus import Path
from .dataset_commons import build_lm_training_datasets
from .librispeech_lm_utils import get_librispeech_spm_datastream, get_librispeech_train_corpus_text, \
    get_librispeech_normalized_lm_data, get_librispeech_lm_combined_txt, get_cv_lm_text
from ...configurations.data.dataset_config import DatasetConfig

"""
NEW LM CODE - mostly from Alberts librispeech file
"""


def build_spm_lm_training_datasets(
        vocab_size: int,
        return_settings: ReturnnDatasetSettings,
        dataset_config: DatasetConfig,
) -> Tuple[TrainingDatasets, SentencePieceDatastream]:
    """
    Builds the training datasets for the SPM-based training - For LM.
    """

    if dataset_config.spm_model_path is None:
        label_datastream = get_librispeech_spm_datastream(
            vocab_size, dataset_config.use_train_corpus_text, dataset_config.use_normalized_lm_data
        )
    else:
        label_datastream = SentencePieceDatastream(
            available_for_inference=False,
            spm_model=Path(dataset_config.spm_model_path),
            vocab_size=vocab_size,
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


    # DATASET STATS
    datasets = {
        "train_lm_data": get_librispeech_normalized_lm_data(),
        "train_other_960h_trans": get_librispeech_train_corpus_text(),
        "train_comb": get_librispeech_lm_combined_txt(),

        "dev-clean": CorpusToTxtJob(get_bliss_corpus_dict()["dev-clean"], gzip=True).out_txt,
        "dev-other": CorpusToTxtJob(get_bliss_corpus_dict()["dev-other"], gzip=True).out_txt,
    }

    for dataset_key, dataset_text_file in datasets.items():
        tokenized_text = ApplySentencepieceToTextJob(
            text_file= dataset_text_file,
            sentencepiece_model= label_datastream.spm_model,
            enable_unk=False,
        ).out_sentencepiece_text

        # todo: stats


        tk.register_output(f"datasets/LibriSpeech/statistics/lm_data/tokenized_datasets/{dataset_key}_tokenized.txt.gz", tokenized_text)
    #CountLinesJob

    return training_datasets, label_datastream
