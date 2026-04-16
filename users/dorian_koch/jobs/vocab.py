import collections
import random
from typing import Any, Dict, List, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen
import re
from returnn.datasets.util.vocabulary import Vocabulary
import functools


class VocabToJsonVocab(Job):
    def __init__(self, vocab_opts: Dict[str, Any]):
        
        self.vocab_opts = vocab_opts
        
        self.out_json_vocab = self.output_path("vocab.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 1
        return super().hash(d)

    def run(self):
        import sys
        import os
        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(None)  # self.returnn_root
        sys.path.insert(1, returnn_root.get_path())

        from returnn.datasets.util.vocabulary import Vocabulary

        vocab = self.vocab_opts
        vocab = util.instanciate_delayed(vocab)
        print("RETURNN vocab opts:", vocab)
        vocab = Vocabulary.create_vocab(**vocab)
        print("Vocab:", vocab)
        print("num labels:", vocab.num_labels)
        assert vocab.num_labels == len(vocab.labels)

        with open(self.out_json_vocab, "w") as f:
            # label -> index
            vocab_dict = {label: i for i, label in enumerate(vocab.labels)}
            import json
            json.dump(vocab_dict, f, indent=4, sort_keys=True)
            f.write("\n")
