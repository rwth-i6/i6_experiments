"""
The new version of data.py for the 2023 Slurm and Rescale/NeuroSys setups
"""
from sisyphus import tk

from dataclasses import dataclass
from functools import lru_cache
import os
from typing import Any, Dict, List, Optional, Tuple

from i6_core.returnn import CodeWrapper, BlissToOggZipJob
from i6_core.returnn.vocabulary import ReturnnVocabFromPhonemeInventory
from i6_core.corpus.transform import ApplyLexiconToCorpusJob
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob

from i6_experiments.common.datasets.tedlium2.lexicon import get_g2p_augmented_bliss_lexicon, get_bliss_lexicon
from i6_experiments.common.datasets.tedlium2.corpus import get_bliss_corpus_dict

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import ReturnnAudioRawOptions, AudioRawDatastream
from i6_experiments.common.setups.returnn.datasets import Dataset, OggZipDataset, MetaDataset

from .default_tools import MINI_RETURNN_ROOT, RETURNN_EXE

DATA_PREFIX = "rescale/tedlium2_standalone_2023/data/"

# -------------- Dataclasses for configuration and data passing -------------------

# here: (<from-epoch> , <to-epoch>, <max-mean-length>)
EpochWiseFilter = Tuple[int, int, int]

@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]
    prior: Optional[Dataset]


@dataclass()
class TrainingDatasetSettings:
    # features settings
    custom_processing_function: Optional[str]

    # training settings
    partition_epoch: int
    epoch_wise_filters: List[EpochWiseFilter]
    seq_ordering: str

# --------------------------- Helper functions  -----------------------------------

def get_eow_lexicon(with_g2p=True) -> tk.Path:
    """
    Standard bliss lexicon modified with EOW
    :return:
    """
    if with_g2p:
        lex = get_g2p_augmented_bliss_lexicon()
    else:
        lex = get_bliss_lexicon()

    return AddEowPhonemesToLexiconJob(lex).out_lexicon


def get_eow_text_lexicon() -> tk.Path:
    """

    :return:
    """
    bliss_lex = get_eow_lexicon(with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(bliss_lex).out_lexicon
    return word_lexicon


def get_eow_bliss(corpus_key, remove_unk_seqs=False) -> tk.Path:
    """
    get an EOW modified corpus with optional unknown removed for cross validation

    :param corpus_key: train, dev, test
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="wav")[corpus_key]
    if remove_unk_seqs:
        from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob
        bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
            bliss_corpus=bliss,
            bliss_lexicon=get_eow_lexicon(),  # assume no g2p when removing unknown for test sets
            all_unknown=False
        ).out_corpus

    # default train lexicon
    lexicon = get_eow_lexicon(with_g2p=True)
    converted_bliss_corpus = ApplyLexiconToCorpusJob(bliss, lexicon, word_separation_orth=None).out_corpus

    return converted_bliss_corpus


def get_eow_vocab_datastream() -> LabelDatastream:
    """
    Phoneme with EOW LabelDatastream for Tedlium-2

    :param with_blank: datastream for CTC training
    """
    lexicon = get_eow_lexicon()
    blacklist = {"[SILENCE]"}
    returnn_vocab_job = ReturnnVocabFromPhonemeInventory(lexicon, blacklist=blacklist)
    returnn_vocab_job.add_alias(os.path.join(DATA_PREFIX, "eow_returnn_vocab_job"))

    vocab_datastream = LabelDatastream(
        available_for_inference=True,
        vocab=returnn_vocab_job.out_vocab,
        vocab_size=returnn_vocab_job.out_vocab_size
    )

    return vocab_datastream


def get_eow_bliss_and_zip(corpus_key, remove_unk_seqs=False):
    """
    :param corpus_key: e.g. "train", "dev", or "test,
    :param remove_unk_seqs: remove all sequences with unknowns, used for dev-clean and dev-other
        in case of using them for cross validation
    :return: tuple of bliss and zip
    """

    bliss_dataset = get_eow_bliss(corpus_key=corpus_key, remove_unk_seqs=remove_unk_seqs)

    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss_dataset,
        no_conversion=False,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    ).out_ogg_zip

    return bliss_dataset, zip_dataset


def get_test_bliss_and_zip(corpus_key):
    """
    for now just return the original ogg zip

    :param corpus_key: e.g. "train", "dev", "test"
    :return:
    """
    bliss = get_bliss_corpus_dict(audio_format="wav")[corpus_key]
    zip_dataset = BlissToOggZipJob(
        bliss_corpus=bliss,
        no_conversion=False,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    ).out_ogg_zip
    return bliss, zip_dataset


@lru_cache()
def get_audio_raw_datastream():
    audio_datastream = AudioRawDatastream(
        available_for_inference=True,
        options=ReturnnAudioRawOptions(peak_normalization=False, preemphasis=0.97)
    )
    return audio_datastream

# --------------------------- Dataset functions  -----------------------------------

def build_phon_training_datasets(
        settings: TrainingDatasetSettings,
) -> TrainingDatasets:
    """
    :param settings:
    """

    train_bliss, train_ogg = get_eow_bliss_and_zip("train")
    dev_bliss, dev_ogg = get_eow_bliss_and_zip("dev", remove_unk_seqs=True)

    train_phon_datastream = get_eow_vocab_datastream()
    audio_datastream = get_audio_raw_datastream()

    datastreams = {
        'raw_audio': audio_datastream,
        'labels': train_phon_datastream,
    }

    data_map = {"raw_audio": ("zip_dataset", "data"),
                "labels": ("zip_dataset", "classes")}

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if settings.custom_processing_function:
        training_audio_opts["pre_process"] = CodeWrapper(settings.custom_processing_function)

    additional_opts = {}
    if settings.epoch_wise_filters:
        additional_opts['epoch_wise_filter'] = {}
        for fr, to, max_mean_len in settings.epoch_wise_filters:
            additional_opts['epoch_wise_filter'][(fr, to)] = {"max_mean_len": max_mean_len}

    def make_meta(dataset: OggZipDataset):
        return MetaDataset(
            data_map=data_map,
            datasets={"zip_dataset": dataset},
            seq_order_control_dataset="zip_dataset"
        )

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_phon_datastream.as_returnn_targets_opts(),
        partition_epoch=settings.partition_epoch,
        seq_ordering=settings.seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = make_meta(train_zip_dataset)

    cv_zip_dataset = OggZipDataset(
        files=dev_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_phon_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = make_meta(cv_zip_dataset)

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_phon_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=3000,
    )
    devtrain_dataset = make_meta(devtrain_zip_dataset)
    
    prior_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_phon_datastream.as_returnn_targets_opts(),
        partition_epoch=1,
        seq_ordering="sorted_reverse",
        additional_options=additional_opts,
    )
    prior_dataset = make_meta(prior_zip_dataset)

    return TrainingDatasets(
        train=train_dataset,
        cv=cv_dataset,
        devtrain=devtrain_dataset,
        datastreams=datastreams,
        prior=prior_dataset,
    )


@lru_cache()
def build_test_dataset(dataset_key: str):
    """
    :param dataset_key: test dataset to generate ("eval" or "test")
    """

    _, test_ogg = get_test_bliss_and_zip(dataset_key)
    bliss_dict = get_bliss_corpus_dict()  # unmodified bliss

    audio_datastream = get_audio_raw_datastream()

    data_map = {"raw_audio": ("zip_dataset", "data")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": test_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, bliss_dict[dataset_key]
