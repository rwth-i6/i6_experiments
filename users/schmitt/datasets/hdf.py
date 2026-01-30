from typing import List, Optional, Dict, Any, Union

from sisyphus import Path, tk

from i6_experiments.common.setups.returnn.datasets.base import ControlDataset


def get_dataset_dict(
        hdf_files: List[Union[Path, str]],
        segment_file: Optional[Path],
        partition_epoch: Optional[int] = None,
        seq_ordering: Optional[str] = None,
):
    d = {
        "class": "HDFDataset",
        "files": hdf_files,
        "seq_list_filter_file": segment_file,
        "use_cache_manager": True,
    }

    if partition_epoch:
        d["partition_epoch"] = partition_epoch

    if seq_ordering is not None:
        d["seq_ordering"] = seq_ordering

    return d


def get_subepoch_dataset(
        files: List[str],
        multi_proc: bool = True,
        postprocessing_opts: Optional[Dict[str, Any]] = None,
):
    d = {
        "class": "HDFDataset",
        "files": files,
        "seq_ordering": "laplace:.1000",
        "use_cache_manager": True,
    }

    if multi_proc:
        d = {
            "class": "MultiProcDataset",
            "dataset": d,
            "num_workers": 2,
            "buffer_size": 10,
        }

    if postprocessing_opts:
        d = {
            "class": "PostprocessingDataset",
            "dataset": d,
            **postprocessing_opts,
        }

    return d


class HdfDataset(ControlDataset):
    def __init__(
            self,
            *,
            files: List[tk.Path],
            segment_file: Optional[tk.Path] = None,
            # super parameters
            partition_epoch: Optional[int] = None,
            seq_ordering: Optional[str] = None,
    ):
        super().__init__(
            partition_epoch=partition_epoch,
            seq_ordering=seq_ordering,
        )
        self.files = files
        self.segment_file = segment_file

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        See `Dataset` definition
        """
        d = get_dataset_dict(
            hdf_files=self.files,
            segment_file=self.segment_file,
        )

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s" % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
