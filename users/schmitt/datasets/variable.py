
def get_interpolation_alignment_dataset(*, epoch: int, **_):
  import returnn.frontend as rf
  from returnn.config import get_global_config
  config = get_global_config()
  train_dataset_dict = config.typed_value("train")
  partition_epoch = train_dataset_dict["datasets"]["zip_dataset"]["partition_epoch"]

  if 1 <= epoch <= partition_epoch or not rf.get_run_ctx().train_flag:
    return {
      "class": "AnythingDataset",
      "data_keys": {
        "data": {
          "dim": 1,
          "shape": (None,),
          "dtype": "int32",
        },
      }
    }
  else:
    n_finished_full_epochs = (epoch - 1) // partition_epoch
    hdf_files = [f"interpolation-alignment_full-epoch-{n_finished_full_epochs}.hdf"]
    return {
      "class": "HDFDataset",
      "files": hdf_files,
      "partition_epoch": 1,
      "use_cache_manager": True,
    }


def get_realignment_dataset(*, epoch: int, **_):
  import returnn.frontend as rf
  from returnn.config import get_global_config
  import os
  config = get_global_config()
  partition_epoch = config.int("train_partition_epoch", 20)
  train_dataset = config.typed_value("train")
  train_hdf_dataset = train_dataset["datasets"]["align"]

  if 1 <= epoch <= partition_epoch or not rf.get_run_ctx().train_flag:
    return train_hdf_dataset
  else:
    n_finished_full_epochs = (epoch - 1) // partition_epoch

    if n_finished_full_epochs > 1:
      os.remove(f"realignment_full-epoch-{n_finished_full_epochs - 1}.hdf")

    hdf_files = [f"realignment_full-epoch-{n_finished_full_epochs}.hdf"]
    return {
      "class": "HDFDataset",
      "files": hdf_files,
      "partition_epoch": partition_epoch,
      "use_cache_manager": True,
    }


def get_interpolation_alignment_scores_dataset(*, epoch: int, **_):
  import returnn.frontend as rf
  from returnn.config import get_global_config
  config = get_global_config()
  train_dataset_dict = config.typed_value("train")
  partition_epoch = train_dataset_dict["datasets"]["zip_dataset"]["partition_epoch"]

  if 1 <= epoch <= partition_epoch or not rf.get_run_ctx().train_flag:
    return {
      "class": "AnythingDataset",
      "data_keys": {
        "data": {
          "dim": 1,
          "shape": [None ,],
          "dtype": "float32",
        },
      }
    }
  else:
    n_finished_full_epochs = (epoch - 1) // partition_epoch
    hdf_files = [f"interpolation-alignment-scores_full-epoch-{n_finished_full_epochs}.hdf"]
    return {
      "class": "HDFDataset",
      "files": hdf_files,
      "partition_epoch": 1,
      "use_cache_manager": True,
    }