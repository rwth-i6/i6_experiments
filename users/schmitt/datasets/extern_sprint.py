from sisyphus.delayed_ops import DelayedFormat
from sisyphus import Path

from typing import Optional


def get_dataset_dict(
        rasr_config_path: Path,
        rasr_nn_trainer_exe: Path,
        target_hdf: Optional[Path],
        segment_path: Optional[Path],
        partition_epoch: Optional[int],
        seq_order_seq_lens_file: Optional[Path],
        seq_ordering: Optional[str]
  ):
  assert target_hdf is None or (partition_epoch is not None and seq_ordering is not None)

  sprint_dataset = {
    "class": "ExternSprintDataset",
    "input_stddev": 3.0,
    "sprintConfigStr": DelayedFormat("--config={}", rasr_config_path),
    "sprintTrainerExecPath": rasr_nn_trainer_exe,
    "suppress_load_seqs_print": True
  }

  if target_hdf is not None:
    meta_dataset = {
      "class": "MetaDataset",
      "data_map": {"targets": ("targets", "data"), "data": ("sprint", "data")},
      "datasets": {
        "targets": {
          "class": "HDFDataset",
          "partition_epoch": partition_epoch,
          "files": [target_hdf],
          "seq_list_filter_file": segment_path,
          "seq_order_seq_lens_file": seq_order_seq_lens_file,
          "seq_ordering": seq_ordering,
          "use_cache_manager": True,
        },
        "sprint": sprint_dataset,
      },
      "seq_order_control_dataset": "targets",
    }
    return meta_dataset
  else:
    return sprint_dataset
