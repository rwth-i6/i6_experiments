from typing import List, Optional

from sisyphus import Path


def get_dataset_dict(
        hdf_files: List[Path],
        partition_epoch: int,
        segment_file: Optional[Path],
):
  return {
    "class": "HDFDataset",
    "files": hdf_files,
    "partition_epoch": partition_epoch,
    "seq_list_filter_file": segment_file,
    "use_cache_manager": True,
  }
