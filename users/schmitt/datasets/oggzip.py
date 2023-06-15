from i6_core.returnn.config import CodeWrapper

from typing import List, Optional, Dict
from sisyphus import Path


def get_dataset_dict(
        oggzip_path_list: List[Path],
        bpe_file: Path,
        vocab_file: Path,
        segment_file: Optional[Path],
        fixed_random_subset: Optional[int],
        partition_epoch: int,
        pre_process: Optional[CodeWrapper],
        seq_ordering: str,
        epoch_wise_filter: Optional[Dict],
):
  return {
    "class": "MetaDataset",
    "data_map": {
        "data": ("zip_dataset", "data"),
        "targets": ("zip_dataset", "classes"),
    },
    "datasets": {
        "zip_dataset": {
            "class": "OggZipDataset",
            "path": oggzip_path_list,
            "use_cache_manager": True,
            "audio": {
                "features": "raw",
                "peak_normalization": True,
                "preemphasis": None,
                "pre_process": pre_process
            },
            "targets": {
                "class": "BytePairEncoding",
                "bpe_file": bpe_file,
                "vocab_file": vocab_file,
                "unknown_label": None,
                "seq_postfix": [0],
            },
            "segment_file": segment_file,
            "partition_epoch": partition_epoch,
            "fixed_random_subset": fixed_random_subset,
            "seq_ordering": seq_ordering,
            "epoch_wise_filter": epoch_wise_filter
        }
    },
    "seq_order_control_dataset": "zip_dataset",
  }
