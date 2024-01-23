#!/usr/bin/env python

"""
For debugging, go through some dataset, forward it through the net, and output the layer activations on stdout.
"""

from __future__ import print_function

import sys

# import returnn.__main__ as rnn
# from returnn.log import log
import argparse
# from returnn.util.basic import pretty_print
from subprocess import check_output
import tensorflow as tf
import numpy as np


def hdf_dataset_init(dim):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_alignment", dim=dim, ndim=1)


def dump(hdf_dataset_in, hdf_dataset_out, blank_idx):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  seq_idx = 0
  keep_segments = []

  while hdf_dataset_in.is_less_than_num_seqs(seq_idx) and seq_idx <= float("inf"):
    hdf_dataset_in.load_seqs(seq_idx, seq_idx + 1)
    data = hdf_dataset_in.get_data(seq_idx, "data")

    if seq_idx % 1000 == 0:
      print("\n\n" + str(seq_idx) + "\n\n")

    # skip sequence if it only consists of blank labels
    if np.all(data == blank_idx):
      seq_idx += 1
      continue

    keep_segments.append(hdf_dataset_in.get_tag(seq_idx))

    seq_len = hdf_dataset_in.get_seq_length(seq_idx)["data"]
    tag = hdf_dataset_in.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(data, axis=0), dtype="int32")

    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset_out.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)

    seq_idx += 1

  with open("out_segment_file", "w+") as f:
    for tag in keep_segments:
      f.write(tag + "\n")


def init(align_path):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  from returnn.log import log
  from returnn.__main__ import init_better_exchook, init_thread_join_hack, init_faulthandler
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()
  init_better_exchook()
  init_thread_join_hack()
  log.initialize(verbosity=[5])
  print("Returnn hdf_dump starting up.", file=log.v3)
  init_faulthandler()

  hdf_align_opts = {
    "class": "HDFDataset", "files": [align_path], "use_cache_manager": True
  }

  hdf_dataset = rnn.init_dataset(hdf_align_opts)

  return hdf_dataset


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Remove alignments which only consist of blank labels.')
  arg_parser.add_argument('align_path')
  arg_parser.add_argument('--blank_idx', type=int)
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  hdf_dataset_in = init(align_path=args.align_path)
  hdf_dataset_out = hdf_dataset_init(dim=hdf_dataset_in.get_data_dim("data"))
  try:
    dump(hdf_dataset_in, hdf_dataset_out, blank_idx=args.blank_idx)
    hdf_dataset_out.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
