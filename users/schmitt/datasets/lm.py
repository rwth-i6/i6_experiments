from typing import List, Optional, Dict, Any, Union, Callable

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import Dataset


class LmDataset(Dataset):
    def __init__(
            self,
            *,
            corpus_file: tk.Path,
            orth_vocab: Dict,
            partition_epoch: int,
            seq_ordering: str,
    ):
        super().__init__(
            additional_options=None
        )
        self.corpus_file = corpus_file
        self.orth_vocab = orth_vocab
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """

        d = {
            "class": "LmDataset",
            "corpus_file": self.corpus_file,
            "orth_vocab": self.orth_vocab,
            "partition_epoch": self.partition_epoch,
            "seq_ordering": self.seq_ordering,
            "use_cache_manager": True,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
