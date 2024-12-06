from typing import Any, Callable, Dict, Optional
from sisyphus import Job, Task, tk
import logging
from i6_core.util import uopen

from recipe.returnn.returnn.datasets.basic import Vocabulary

class DatasetToTextDictJob(Job):
    """
    Takes a dataset and converts the tag specified in data_key into a text dictionary.
    """

    def __init__(
        self,
        *,
        returnn_dataset: Dict[str, Any],
        returnn_root: Optional[tk.Path] = None,
        data_key: str,
        take_vocab_from_key: Optional[str] = None,
        vocab_to_words: Optional[Callable] = None,
        gzip: bool = False,
        rqmt: Optional[Dict[str, Any]] = None,
    ):
        self.returnn_dataset = returnn_dataset
        self.returnn_root = returnn_root
        self.data_key = data_key
        self.vocab_to_words = vocab_to_words
        self.take_vocab_from_key = take_vocab_from_key
        self.rqmt = rqmt

        self.out_dictionary = self.output_path("text_dictionary.py" + (".gz" if gzip else ""))

    def tasks(self):
        if self.rqmt:
            yield Task("run", rqmt=self.rqmt)
        else:
            yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import sys

        if self.returnn_root is not None:
            sys.path.insert(0, self.returnn_root.get_path())
        from returnn.config import set_global_config, Config
        from returnn.log import log

        config = Config()
        set_global_config(config)

        if not config.has("log_verbosity"):
            config.typed_dict["log_verbosity"] = 5
        log.init_by_config(config)
        from returnn.datasets import init_dataset
        import tree

        dataset_dict = self.returnn_dataset
        dataset_dict = tree.map_structure(lambda x: x.get_path() if isinstance(x, tk.Path) else x, dataset_dict)
        logging.debug("RETURNN dataset dict:", dataset_dict)
        assert isinstance(dataset_dict, dict)
        dataset = init_dataset(dataset_dict)
        dataset.init_seq_order(epoch=1)

        assert self.data_key in dataset.get_data_keys()
        
        if self.take_vocab_from_key:
            vocab = Vocabulary.create_vocab_from_labels(dataset.labels[self.take_vocab_from_key])
        else:
            assert dataset.can_serialize_data(self.data_key)
            vocab = Vocabulary.create_vocab_from_labels(dataset.labels[self.data_key])

        with uopen(self.out_dictionary, "wt") as out:
            out.write("{\n")
            seq_idx = 0
            while dataset.is_less_than_num_seqs(seq_idx):
                dataset.load_seqs(seq_idx, seq_idx + 1)
                if seq_idx % 10000 == 0:
                    logging.info(f"seq_idx {seq_idx}")
                key = dataset.get_tag(seq_idx)
                orth = vocab.get_seq_labels(dataset.get_data(seq_idx, self.data_key))
                if self.vocab_to_words:
                    orth = self.vocab_to_words(orth)
                out.write("%r: %r,\n" % (key, orth))
                seq_idx += 1
            out.write("}\n")