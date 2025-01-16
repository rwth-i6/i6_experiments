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


class ExtendVocabLabelsByNewLabelJob(Job):
    """
    Takes vocab (line-based (maybe gzipped) list of labels),
    and extends it by a new label,
    at a given index.
    """

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
        labels = util.uopen(self.vocab.get_path(), "rt").readlines()
        new_label_idx = self.new_label_idx
        if new_label_idx < 0:
            new_label_idx = len(labels) + 1 + new_label_idx
        labels.insert(new_label_idx, self.new_label)

        with util.uopen(self.out_vocab.get_path(), "wt") as f:
            for label in labels:
                f.write(label + "\n")
