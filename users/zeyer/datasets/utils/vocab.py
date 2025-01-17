"""
Vocab utils
"""


from __future__ import annotations
from typing import Optional, Any, Dict
from sisyphus import Job, Task, tk
from i6_core import util


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

        d = {"unknown_label": vocab.unknown_label}
        if vocab.bos_label_id is not None:
            d["bos_label"] = labels[vocab.bos_label_id]
        if vocab.eos_label_id is not None:
            d["eos_label"] = labels[vocab.eos_label_id]
        if vocab.pad_label_id is not None:
            d["pad_label"] = labels[vocab.pad_label_id]
        if vocab.control_symbol_ids:
            d["control_symbols"] = {k: labels[v] for k, v in vocab.control_symbol_ids.items()}
        if vocab.user_defined_symbol_ids:
            d["user_defined_symbols"] = {k: labels[v] for k, v in vocab.user_defined_symbol_ids.items()}

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
