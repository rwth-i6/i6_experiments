#!/usr/bin/env python3

"""
Creates a HDF file, which can be read by :class:`HDFDataset`.
The input is any other dataset (:class:`Dataset`).
"""

from __future__ import print_function

import json
import sys
import argparse
import numpy as np
import tensorflow as tf


def hdf_dataset_init(dim):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_align", dim=dim, ndim=1)


def hdf_dump_from_dataset(dataset, hdf_dataset):
  """
  :param Dataset dataset: could be any dataset implemented as child of Dataset
  :param hdf_dataset_mod.HDFDatasetWriter hdf_dataset:
  :param parser_args: argparse object from main()
  """
  seq_idx = 0
  end_idx = float("inf")
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_idx:
    dataset.load_seqs(seq_idx, seq_idx + 1)

    data = dataset.get_data(seq_idx, "classes")

    seq_len = dataset.get_seq_length(seq_idx)["classes"]
    tag = dataset.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(data, axis=0), dtype="int32")

    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1 #- (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)
    seq_idx += 1


def hdf_close(hdf_dataset):
  """
  :param HDFDataset.HDFDatasetWriter hdf_dataset: to close
  """
  hdf_dataset.close()


def init(rasr_config_path, rasr_exe):
  """
  :param str config_filename: global config for CRNN
  :param list[str] cmd_line_opts: options for init_config method
  :param str dataset_config_str: dataset via init_dataset_via_str()
  """
  from returnn.log import log
  from returnn.__main__ import init_better_exchook, init_thread_join_hack, init_faulthandler
  init_better_exchook()
  init_thread_join_hack()
  log.initialize(verbosity=[5])
  print("Returnn hdf_dump starting up.", file=log.v3)
  init_faulthandler()

  epoch_split = 1

  sprint_args = [
    "--config=%s" % rasr_config_path,
    "--*.LOGFILE=nn-trainer.train.log", "--*.TASK=1", "--*.corpus.segment-order-shuffle=true",
  ]

  dataset_dict = {
    'class': 'ExternSprintDataset',
    'sprintConfigStr': sprint_args,
    'sprintTrainerExecPath': rasr_exe,
    "partition_epoch": epoch_split}
  dataset = rnn.datasets.init_dataset(dataset_dict)
  print("Source dataset:", dataset.len_info(), file=log.v3)

  return dataset


def main(argv):
  """
  Main entry.
  """
  parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset into external HDF dataset")
  parser.add_argument('rasr_config', type=str,
                      help="Config file for RETURNN, or directly the dataset init string")
  parser.add_argument('--rasr_exe', type=str)
  parser.add_argument('--returnn_root', help="Returnn root to use for imports")

  args = parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  dataset = init(
    rasr_config_path=args.rasr_config, rasr_exe=args.rasr_exe)
  hdf_dataset = hdf_dataset_init(dim=dataset.get_data_dim("classes"))
  hdf_dump_from_dataset(dataset, hdf_dataset)
  hdf_close(hdf_dataset)

  rnn_main.finalize()


if __name__ == '__main__':
  main(sys.argv)
