"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
import copy
from typing import Tuple

from sisyphus import tk

from i6_core.text import TakeNRandomLinesJob
from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.data.lm_dataset import LmDataset
from i6_experiments.users.juanola.data.training_datasets import TrainingDatasets
from i6_experiments.users.juanola.sisyphus_jobs.text.ShuffleJob import ShuffleJob
from i6_experiments.users.juanola.sisyphus_jobs.text.TwoWaySplitJob import TowWaySplitJob
from ...default_tools import RETURNN_ROOT, RETURNN_EXE


def get_random_subset(train_text_file: tk.Path, n_lines: int = 3000) -> tk.Path:
    """
    because handling textual data we can get n random lines and this should be enough
    :param train_text_file:
    :param n_lines:
    :return:
    """
    n_random_lines_job = TakeNRandomLinesJob(text_file=train_text_file, num_lines=n_lines)
    return n_random_lines_job.out


def split_train_test(text_file: tk.Path, cv_lines: int = 3000) -> tuple[tk.Path, tk.Path]:
    shuffle_job = ShuffleJob(text_file=text_file)
    split_job = TowWaySplitJob(text_file=shuffle_job.out, a_file_n_lines=cv_lines)
    cv_split = split_job.out_a
    train_split = split_job.out_b
    return train_split, cv_split


def build_lm_training_datasets(
        train_text_file: tk.Path,
        cv_text_file: tk.Path,
        label_datastream: LabelDatastream,
        returnn_settings: ReturnnDatasetSettings,
        alpha: float,
        dev_train_lines: int
) -> TrainingDatasets:
    """
    generic dataset construction helper to be used by the phon/bpe specific variants

    :param train_file: path to the train zip, potentially containing altered transcriptions
    :param dev_clean_file: path to the ls dev-clean zip, potentially containing altered transcriptions
    :param dev_other_file: path to the ls dev-other zip, potentially containing altered transcriptions
    :param label_datastream: label datastream (e.g. phoneme or bpe related)
    :param returnn_settings: settings object for the RETURNN data pipeline
    """
    # Vocab settings   # TODO: not sure what is and what isn't needed
    vocab_settings = label_datastream.as_returnn_targets_opts()
    vocab_settings.pop("add_eos", None)  # SentencePieceDatastream only covers limited options and always adds EOS, which we don't want
    training_vocab_settings = copy.deepcopy(vocab_settings)
    training_vocab_settings.update({"alpha": alpha, "enable_sampling": True} if alpha is not None else {})

    # Data splits
    train_dataset = LmDataset(
        corpus_file=train_text_file,
        vocab_settings=training_vocab_settings,
        seq_ordering=returnn_settings.train_seq_ordering,
        partition_epoch=returnn_settings.train_partition_epoch,
    )

    cv_dataset = LmDataset(
        corpus_file=cv_text_file,
        vocab_settings=vocab_settings,
        seq_ordering="sorted_reverse",  # TODO: needed ??
    )

    devtrain_text_file = get_random_subset(train_text_file, n_lines=dev_train_lines)
    devtrain_dataset = LmDataset(
        corpus_file=devtrain_text_file,
        vocab_settings=vocab_settings,
        seq_ordering="sorted_reverse",  # TODO: needed ??
    )

    datastreams = {"labels": label_datastream}

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
    )


def build_lm_test_dataset(
        dataset_key: str,
        settings: ReturnnDatasetSettings,
) -> Tuple[Dataset, tk.Path]:
    """
    Create ASR test set that only contains the audio stream

    :param dataset_key: e.g. dev-other, which test set to create
    :param settings: settings object for the RETURNN data pipeline
    :return: tuple of the test dataset and a path to the corresponding bliss corpus file
    """
    # TODO: do we need recognition?!
    # TODO: not addapted!
    ogg_zip_dict = get_ogg_zip_dict("corpora", returnn_root=RETURNN_ROOT, returnn_python_exe=RETURNN_EXE)
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        seq_ordering="sorted_reverse"
    )

    return test_zip_dataset, bliss_dict[dataset_key]
