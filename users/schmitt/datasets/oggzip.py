from i6_core.returnn.config import CodeWrapper

from typing import List, Optional, Dict
from sisyphus import Path


def get_dataset_dict(
        oggzip_path_list: List[Path],
        bpe_file: Optional[Path],
        vocab_file: Optional[Path],
        segment_file: Optional[Path],
        fixed_random_subset: Optional[int],
        partition_epoch: int,
        pre_process: Optional[CodeWrapper],
        seq_ordering: str,
        epoch_wise_filter: Optional[Dict],
        hdf_targets: Optional[Path] = None,
        seq_postfix: Optional[int] = 0,
        use_targets: bool = True,
        peak_normalization: bool = True,
):
  assert not use_targets or (bpe_file is not None and vocab_file is not None)
  dataset_dict = {
    "class": "MetaDataset",
    "data_map": {
      "data": ("zip_dataset", "data")
    },
    "datasets": {
      "zip_dataset": {
        "class": "OggZipDataset",
        "path": oggzip_path_list,
        "use_cache_manager": True,
        "audio": {
          "features": "raw",
          "peak_normalization": peak_normalization,
          "preemphasis": None,
          "pre_process": pre_process
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

  if use_targets:
    dataset_dict["datasets"]["zip_dataset"]["targets"] = {
      "class": "BytePairEncoding",
      "bpe_file": bpe_file,
      "vocab_file": vocab_file,
      "unknown_label": None,
      "seq_postfix": [seq_postfix] if seq_postfix is not None else None,
    }
    dataset_dict["data_map"]["targets"] = ("zip_dataset", "classes")
  else:
    dataset_dict["datasets"]["zip_dataset"]["targets"] = None

  if hdf_targets is not None:
    dataset_dict["datasets"]["align"] = {
      "class": "HDFDataset",
      "files": [
        hdf_targets
      ],
      "partition_epoch": partition_epoch,
      "seq_list_filter_file": segment_file,
      "use_cache_manager": True,
    }

    dataset_dict["data_map"]["targets"] = ("align", "data")

  return dataset_dict
