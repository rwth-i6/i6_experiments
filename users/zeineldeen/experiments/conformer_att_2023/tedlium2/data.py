from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any

from sisyphus import *

from i6_core.returnn import CodeWrapper
from i6_experiments.common.datasets.tedlium2.corpus import get_ogg_zip_dict, get_bliss_corpus_dict
from i6_experiments.common.datasets.tedlium2.vocab import get_subword_nmt_bpe
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import (
    AudioRawDatastream,
    ReturnnAudioRawOptions,
)
from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import BpeDatastream

from returnn_common.datasets import Dataset, OggZipDataset, MetaDataset


@lru_cache()
def get_bpe_datastream(bpe_size, is_recog):
    """
    Returns the datastream for the bpe labels

    Uses the legacy BPE setup that is compatible with old LM models

    :param int bpe_size: size for the bpe labels
    :param bool is_recog: removes the UNK label when not in training
    :return:
    """
    # build dataset
    bpe_settings = get_subword_nmt_bpe(bpe_size=bpe_size, unk_label="<unk>")
    bpe_targets = BpeDatastream(available_for_inference=False, bpe_settings=bpe_settings, use_unk_label=is_recog)
    return bpe_targets


@lru_cache()
def get_audio_raw_datastream(preemphasis=None):
    audio_datastream = AudioRawDatastream(
        available_for_inference=True, options=ReturnnAudioRawOptions(peak_normalization=True, preemphasis=preemphasis)
    )
    return audio_datastream


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    extern_data: Dict[str, Dict[str, Any]]


def build_training_datasets(
    bpe_size=1000,
    partition_epoch=4,
    epoch_wise_filter=None,
    seq_ordering="laplace:.1000",
    link_speed_perturbation=False,
    use_raw_features=False,
    preemphasis=None,  # TODO: by default is None, but we want to experiment
    devtrain_subset=None,
):
    """

    :param returnn_python_exe:
    :param returnn_root:
    :param output_path:
    :param int bpe_size:
    :param int bpe_partition_epoch:
    :param str seq_ordering:
    :return:
    """
    ogg_zip_dict = get_ogg_zip_dict("corpora", bliss_to_ogg_job_rqmt={"mem": 4, "time": 8, "cpu": 1})
    train_ogg = ogg_zip_dict["train"]
    dev_ogg = ogg_zip_dict["dev"]

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)
    else:
        raise NotImplementedError

    extern_data = {
        "audio_features": audio_datastream.as_returnn_extern_data_opts(),
        "bpe_labels": train_bpe_datastream.as_returnn_extern_data_opts(),
    }

    data_map = {"audio_features": ("zip_dataset", "data"), "bpe_labels": ("zip_dataset", "classes")}

    training_audio_opts = audio_datastream.as_returnn_audio_opts()
    if link_speed_perturbation:
        training_audio_opts["pre_process"] = CodeWrapper("speed_pert")

    additional_opts = {}
    if epoch_wise_filter:
        additional_opts["epoch_wise_filter"] = {}
        for fr, to, max_mean_len in epoch_wise_filter:
            additional_opts["epoch_wise_filter"][(fr, to)] = {"max_mean_len": max_mean_len}

    train_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=training_audio_opts,
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=partition_epoch,
        seq_ordering=seq_ordering,
        additional_options=additional_opts,
    )
    train_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": train_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    cv_zip_dataset = OggZipDataset(
        files=dev_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
    )
    cv_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": cv_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = OggZipDataset(
        files=train_ogg,
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
        random_subset=devtrain_subset if devtrain_subset is not None else 3000,
    )
    devtrain_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": devtrain_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return TrainingDatasets(train=train_dataset, cv=cv_dataset, devtrain=devtrain_dataset, extern_data=extern_data)


@lru_cache()
def build_test_dataset(dataset_key, bpe_size=10000, use_raw_features=False, preemphasis=None, merge_contractions=False):
    ogg_zip_dict = get_ogg_zip_dict("corpora")
    bliss_dict = get_bliss_corpus_dict()
    test_ogg = ogg_zip_dict[dataset_key]
    from i6_core.corpus.convert import CorpusToTextDictJob

    test_reference_dict_file = CorpusToTextDictJob(bliss_dict[dataset_key]).out_dictionary
    if merge_contractions:
        test_reference_dict_file = MergeContractionsJob(test_reference_dict_file).out_dict

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=True)

    if use_raw_features:
        audio_datastream = get_audio_raw_datastream(preemphasis)
    else:
        raise NotImplementedError

    data_map = {"audio_features": ("zip_dataset", "data"), "bpe_labels": ("zip_dataset", "classes")}

    test_zip_dataset = OggZipDataset(
        files=[test_ogg],
        audio_options=audio_datastream.as_returnn_audio_opts(),
        target_options=train_bpe_datastream.as_returnn_targets_opts(),
        seq_ordering="sorted_reverse",
    )
    test_dataset = MetaDataset(
        data_map=data_map, datasets={"zip_dataset": test_zip_dataset}, seq_order_control_dataset="zip_dataset"
    )

    return test_dataset, test_reference_dict_file, bliss_dict[dataset_key]


def _merge_contractions(s):
    import re

    res = re.sub(r"(\b\w+)\s'([a-zA-Z]+)\b", r"\1'\2", s)
    res = res.lower()
    return res


class MergeContractionsJob(Job):
    """
    Merge contractions (ending with 's)
    example: he 's => he's

    Takes as input a dict with format {<seq-tag>: <text>}
    """

    def __init__(self, dict_file: tk.Path):
        self.dict_file = dict_file
        self.out_dict = self.output_path("out_dict.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        if self.dict_file.get_path().endswith(".gz"):
            import gzip

            open_func = gzip.open
        else:
            open_func = open

        d = eval(open_func(self.dict_file.get_path()).read())

        out_f = open(self.out_dict.get_path(), "wt")
        out_f.write("{\n")
        for k, v in d.items():
            out_f.write(f"{k!r}: {_merge_contractions(v)!r},\n")
        out_f.write("}\n")


class CorpusMergeContractionsJob(Job):

    def __init__(self, corpus_file: tk.Path):
        self.corpus_file = corpus_file

        self.out_corpus_file = self.output_path("out_corpus.xml")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from i6_core.lib.corpus import Corpus

        c = Corpus()
        c.load(self.corpus_file.get_path())

        for seg in c.segments():
            seg.text = _merge_contractions(seg.text)

        c.dump(self.out_corpus_file.get_path())
