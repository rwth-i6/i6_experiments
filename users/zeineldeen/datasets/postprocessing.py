from typing import List, Optional, Dict, Any, Union, Callable

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import ControlDataset, Dataset


class PostprocessingDataset(Dataset):
    def __init__(
        self,
        *,
        dataset: Dataset,
        map_seq_stream: Optional[Dict],
        map_seq: Optional[Callable] = None,
        buf_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ):
        super().__init__(additional_options=None)
        self.dataset = dataset
        self.map_seq_stream = map_seq_stream
        self.map_seq = map_seq

        assert (
            self.map_seq_stream is not None or self.map_seq is not None
        ), "Either map_seq_stream or map_seq must be specified"
        assert not (
            self.map_seq_stream is not None and self.map_seq is not None
        ), "Only one of map_seq_stream and map_seq can be specified"

        self.buf_size = buf_size
        self.num_workers = num_workers

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """

        d = {
            "class": "PostprocessingDataset",
            "dataset": self.dataset.as_returnn_opts(),
            "map_seq_stream": self.map_seq_stream,
        }
        if self.map_seq is not None:
            d["map_seq"] = self.map_seq
        if self.buf_size is not None:
            d["buf_size"] = self.buf_size
        if self.num_workers is not None:
            d["num_workers"] = self.num_workers

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
