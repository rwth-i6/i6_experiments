from typing import List, Optional, Dict, Any, Union, Callable

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import ControlDataset, Dataset


class PostprocessingDataset(Dataset):
    def __init__(
        self,
        *,
        dataset: Dataset,
        map_seq_stream: Optional[Dict] = None,
        buf_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        map_outputs: Optional[Dict] = None,
        map_seq=None,
    ):
        super().__init__(additional_options=None)

        assert not (map_seq and map_seq_stream), "map_seq and map_seq_stream cannot be used at the same time"
        assert not (not map_seq and not map_seq_stream), "one of map_seq and map_seq_stream must be provided"

        self.dataset = dataset
        self.map_seq_stream = map_seq_stream
        self.buf_size = buf_size
        self.num_workers = num_workers
        self.map_outputs = map_outputs
        self.map_seq = map_seq

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """

        d = {
            "class": "PostprocessingDataset",
            "dataset": self.dataset.as_returnn_opts(),
        }
        if self.map_seq_stream is not None:
            d["map_seq_stream"] = self.map_seq_stream
        if self.buf_size is not None:
            d["buf_size"] = self.buf_size
        if self.num_workers is not None:
            d["num_workers"] = self.num_workers
        if self.map_outputs is not None:
            d["map_outputs"] = self.map_outputs
        if self.map_seq is not None:
            d["map_seq"] = self.map_seq

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
