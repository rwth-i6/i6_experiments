from typing import List, Optional, Dict, Any, Union, Callable

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import Dataset


class LmDataset(Dataset):
    def __init__(
        self,
        *,
        corpus_file: tk.Path,
        partition_epoch: int = 1,
        seq_ordering: Optional[str] = None,
        orth_vocab: Optional[Dict] = None,
        seq_list_file: Optional[Union[str, Path]] = None,
    ):
        super().__init__(additional_options=None)
        self.corpus_file = corpus_file
        self.orth_vocab = orth_vocab
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.seq_list_file = seq_list_file

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """

        d = {
            "class": "LmDataset",
            "corpus_file": self.corpus_file,
            "orth_vocab": self.orth_vocab,
            "partition_epoch": self.partition_epoch,
            "use_cache_manager": True,
        }

        if self.seq_ordering is not None:
            d["seq_ordering"] = self.seq_ordering

        if self.seq_list_file is not None:
            d["seq_list_file"] = self.seq_list_file

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
