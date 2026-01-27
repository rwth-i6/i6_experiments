"""
Dataset helpers for the SPM-based training
"""
from typing import Optional, Dict

from sisyphus import tk

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from i6_experiments.users.zeyer.datasets.librispeech import _get_spm_vocab  # TODO: warning! external import!
from .dataset_commons import build_training_datasets
from ...default_tools import RETURNN_ROOT, RETURNN_EXE


def get_spm_datastream(vocab_size: int) -> SentencePieceDatastream:
    """
    Returns the datastream for the spm labels

    :param vocab_size: the size of the vocabulary
    """
    spm_model = _get_spm_vocab(dim=vocab_size)  # Where TrainSentencePieceJob is called

    spm_targets = SentencePieceDatastream(
        available_for_inference=False,
        spm_model=spm_model.model_file,
        vocab_size=vocab_size,
    )
    return spm_targets


def build_spm_training_datasets(
        prefix: str,
        librispeech_key: str,
        vocab_size: int,
        return_settings: ReturnnDatasetSettings,
        returnn_root: tk.Path = RETURNN_ROOT,
        alpha: Optional[float] = None,
        datasets_num_workers=int,
) -> TrainingDatasets:
    """
    Builds the training datasets for the SPM-based training.
    """
    label_datastream = get_spm_datastream(vocab_size=vocab_size)

    ogg_zip_dict: Dict[str, tk.Path] = get_ogg_zip_dict(prefix, returnn_root=returnn_root,
                                                        returnn_python_exe=RETURNN_EXE)
    training_datasets = build_training_datasets(
        train_ogg=ogg_zip_dict[librispeech_key],
        dev_clean_ogg=ogg_zip_dict["dev-clean"],
        dev_other_ogg=ogg_zip_dict["dev-other"],
        returnn_settings=return_settings,
        label_datastream=label_datastream,
        datasets_num_workers=datasets_num_workers,
    )

    # SentencePieceDatastream only covers limited options and always adds EOS, which we don't want
    for dataset in [training_datasets.train, training_datasets.cv, training_datasets.devtrain]:
        dataset.dataset.target_options.pop("add_eos", None)
        if dataset == training_datasets.train:
            dataset.dataset.target_options.update(
                {
                    "alpha": alpha,
                    "enable_sampling": True,
                }
                if alpha is not None
                else {},
            )
    return training_datasets
