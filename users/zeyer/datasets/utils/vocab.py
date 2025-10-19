"""
Vocab utils
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Tuple
import functools
from sisyphus import Job, Task, tk
from i6_core import util

from i6_experiments.users.zeyer.datasets.task import Task as DatasetsTask


def get_vocab_w_blank_file_from_task(task: DatasetsTask, *, blank_label: str, blank_idx: int) -> tk.Path:
    vocab_file = ExtractVocabLabelsJob(get_vocab_opts_from_task(task)).out_vocab
    vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
        vocab=vocab_file, new_label=blank_label, new_label_idx=blank_idx
    ).out_vocab
    return vocab_w_blank_file


def get_vocab_opts_from_task(task: DatasetsTask) -> Dict[str, Any]:
    dataset = task.dev_dataset
    extern_data_dict = dataset.get_extern_data()
    target_dict = extern_data_dict[dataset.get_default_target()]
    return target_dict["vocab"]


def get_vocab_opts_from_dataset_dict(dataset_dict: Dict[str, Any], data_key: str = "classes") -> Dict[str, Any]:
    cls_name = dataset_dict["class"]
    if cls_name == "OggZipDataset":
        assert data_key == "classes"
        return dataset_dict["targets"]
    elif cls_name == "LmDataset":
        return dataset_dict["orth_vocab"]
    else:
        raise NotImplementedError(f"Cannot get vocab opts from dataset dict of class {cls_name!r} in {dataset_dict}")


def update_vocab_opts_in_dataset_dict(
    dataset_dict: Dict[str, Any], new_vocab_opts: Dict[str, Any], data_key: str = "classes"
) -> Dict[str, Any]:
    dataset_dict = dataset_dict.copy()
    cls_name = dataset_dict["class"]
    if cls_name == "OggZipDataset":
        assert data_key == "classes"
        dataset_dict["targets"] = new_vocab_opts
        return dataset_dict
    elif cls_name == "LmDataset":
        dataset_dict["orth_vocab"] = new_vocab_opts
        return dataset_dict
    elif cls_name == "HuggingFaceDataset":
        dataset_dict["data_format"] = dataset_dict["data_format"].copy()
        dataset_dict["data_format"][data_key] = dataset_dict["data_format"][data_key].copy()
        assert dataset_dict["data_format"][data_key]["vocab"]
        dataset_dict["data_format"][data_key]["vocab"] = new_vocab_opts
        return dataset_dict
    elif cls_name == "DistributeFilesDataset":
        f = dataset_dict["get_sub_epoch_dataset"]
        assert isinstance(f, functools.partial)
        new_kwargs = f.keywords.copy()
        k, sub_ds_dict = _find_relevant_ds_dict_in_arbitrary_kwargs(new_kwargs)
        new_kwargs[k] = update_vocab_opts_in_dataset_dict(sub_ds_dict, new_vocab_opts, data_key=data_key)
        dataset_dict["get_sub_epoch_dataset"] = functools.partial(f.func, *f.args, **new_kwargs)
        return dataset_dict
    else:
        raise NotImplementedError(
            f"Cannot update vocab opts in dataset dict of class {cls_name!r}"
            f" in {dataset_dict} for data key {data_key!r}"
        )


def _find_relevant_ds_dict_in_arbitrary_kwargs(kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    prev_k = None
    ds_dict = None
    for k, v in kwargs.items():
        if isinstance(v, dict) and "class" in v:
            assert ds_dict is None, (
                f"Found multiple dataset dicts in kwargs {kwargs}, prev key {prev_k}, current key {k}"
            )
            prev_k = k
            ds_dict = v
    if ds_dict:
        return prev_k, ds_dict
    raise ValueError(f"Cannot find relevant dataset dict in kwargs {kwargs}")


class ExtractVocabLabelsJob(Job):
    """
    Takes any RETURNN vocabulary, and extracts all labels from it.
    """

    def __init__(self, vocab_opts: Dict[str, Any], *, returnn_root: Optional[tk.Path] = None):
        """
        :param vocab_opts:
        :param returnn_root: path, optional, the RETURNN root dir.
        """
        self.vocab_opts = vocab_opts
        self.returnn_root = returnn_root

        self.out_vocab = self.output_path("vocab.txt.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = self.vocab_opts
        vocab = util.instanciate_delayed(vocab)
        print("RETURNN vocab opts:", vocab)
        vocab = Vocabulary.create_vocab(**vocab)
        print("Vocab:", vocab)
        print("num labels:", vocab.num_labels)
        assert vocab.num_labels == len(vocab.labels)

        with util.uopen(self.out_vocab.get_path(), "wt") as f:
            for label in vocab.labels:
                f.write(label + "\n")


class ExtractVocabSpecialLabelsJob(Job):
    """
    Takes any RETURNN vocabulary, and extracts special labels from it,
    stores it in a dict.
    """

    def __init__(self, vocab_opts: Dict[str, Any], *, returnn_root: Optional[tk.Path] = None):
        """
        :param vocab_opts:
        :param returnn_root: path, optional, the RETURNN root dir.
        """
        self.vocab_opts = vocab_opts
        self.returnn_root = returnn_root

        self.out_vocab_special_labels_dict = self.output_path("vocab_special_labels.py")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = self.vocab_opts
        vocab = util.instanciate_delayed(vocab)
        print("RETURNN vocab opts:", vocab)
        vocab = Vocabulary.create_vocab(**vocab)
        print("Vocab:", vocab)
        print("num labels:", vocab.num_labels)
        assert vocab.num_labels == len(vocab.labels)
        labels = vocab.labels

        def _repr_label_id(label_id: int) -> str:
            label = labels[label_id]
            # Currently assume the label is unique.
            # We could make this configurable later,
            # or maybe also return the label id in this case,
            # but for now, just make sure it is unique.
            assert labels.count(label) == 1
            return label

        d = {"unknown_label": vocab.unknown_label}
        if vocab.bos_label_id is not None:
            d["bos_label"] = _repr_label_id(vocab.bos_label_id)
        if vocab.eos_label_id is not None:
            d["eos_label"] = _repr_label_id(vocab.eos_label_id)
        if vocab.pad_label_id is not None:
            d["pad_label"] = _repr_label_id(vocab.pad_label_id)
        if vocab.control_symbol_ids:
            d["control_symbols"] = {k: _repr_label_id(v) for k, v in vocab.control_symbol_ids.items()}
        if vocab.user_defined_symbol_ids:
            d["user_defined_symbols"] = {k: _repr_label_id(v) for k, v in vocab.user_defined_symbol_ids.items()}

        with util.uopen(self.out_vocab_special_labels_dict.get_path(), "wt") as f:
            f.write("{\n")
            for k, v in d.items():
                f.write(f"  {k!r}: {v!r},\n")
            f.write("}\n")


class ExtendVocabLabelsByNewLabelJob(Job):
    """
    Takes vocab (line-based (maybe gzipped) list of labels),
    and extends it by a new label,
    at a given index.
    """

    __sis_version__ = 2

    def __init__(self, *, vocab: tk.Path, new_label: str, new_label_idx: int):
        """
        :param vocab:
        :param new_label:
        :param new_label_idx: can also be negative, then it is counted from the end.
        """
        self.vocab = vocab
        self.new_label = new_label
        self.new_label_idx = new_label_idx

        self.out_vocab = self.output_path("vocab.txt.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        labels = util.uopen(self.vocab.get_path(), "rt").read().splitlines()
        new_label_idx = self.new_label_idx
        if new_label_idx < 0:
            new_label_idx = len(labels) + 1 + new_label_idx
        labels.insert(new_label_idx, self.new_label)

        with util.uopen(self.out_vocab.get_path(), "wt") as f:
            for label in labels:
                f.write(label + "\n")


class ExtractLineBasedLexiconJob(Job):
    """
    Takes any RETURNN vocabulary, and a list of words, and maps each word to its labels.

    The format is so that TorchAudio/Flashlight can use it.
    """

    def __init__(self, *, vocab_opts: Dict[str, Any], word_list: tk.Path, returnn_root: Optional[tk.Path] = None):
        """
        :param vocab_opts:
        :param word_list:
        :param returnn_root: path, optional, the RETURNN root dir.
        """
        self.vocab_opts = vocab_opts
        self.word_list = word_list
        self.returnn_root = returnn_root

        self.out_lexicon = self.output_path("lexicon.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(1, returnn_root.get_path())

        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = self.vocab_opts
        vocab = util.instanciate_delayed(vocab)
        print("RETURNN vocab opts:", vocab)
        vocab = Vocabulary.create_vocab(**vocab)
        print("Vocab:", vocab)
        print("num labels:", vocab.num_labels)
        assert vocab.num_labels == len(vocab.labels)

        word_list = util.uopen(self.word_list.get_path(), "rt").read().splitlines()

        with util.uopen(self.out_lexicon.get_path(), "wt") as f:
            for word in word_list:
                assert " " not in word
                labels = vocab.get_seq(word)
                assert all(" " not in vocab.labels[label_id] for label_id in labels)
                label_str = " ".join(vocab.labels[label_id] for label_id in labels)
                f.write(f"{word} {label_str}\n")
