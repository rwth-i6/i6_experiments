"""
Dataset helpers for the SPM-based training
"""
from typing import Optional

from sisyphus import tk

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict
from i6_experiments.common.setups.returnn.datastreams.vocabulary import SentencePieceDatastream
from i6_experiments.users.zeyer.datasets.librispeech import _get_spm_vocab
from .common import DatasetSettings, TrainingDatasets, build_training_datasets
from ..default_tools import RETURNN_ROOT, RETURNN_EXE


def get_spm_datastream(vocab_size: int) -> SentencePieceDatastream:
    """
    Returns the datastream for the spm labels

    :param vocab_size: the size of the vocabulary
    """
    spm_model = _get_spm_vocab(dim=vocab_size)

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
    settings: DatasetSettings,
    returnn_root: tk.Path = RETURNN_ROOT,
    alpha: Optional[float] = None,
) -> TrainingDatasets:
    """
    Builds the training datasets for the SPM-based training.
    """
    label_datastream = get_spm_datastream(
        vocab_size=vocab_size,
    )

    ogg_zip_dict = get_ogg_zip_dict(prefix, returnn_root=returnn_root, returnn_python_exe=RETURNN_EXE)
    train_ogg = ogg_zip_dict[librispeech_key]
    dev_clean_ogg = ogg_zip_dict["dev-clean"]
    dev_other_ogg = ogg_zip_dict["dev-other"]

    training_datasets = build_training_datasets(
        train_ogg=train_ogg,
        dev_clean_ogg=dev_clean_ogg,
        dev_other_ogg=dev_other_ogg,
        settings=settings,
        label_datastream=label_datastream,
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
