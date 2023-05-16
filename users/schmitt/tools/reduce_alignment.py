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


def dump(hdf_dataset_in, hdf_dataset_out, sil_idx, blank_idx, reduction_factor):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  seq_idx = 0

  print("SIL IDX: ", sil_idx)
  print("BLANK IDX: ", blank_idx)

  skipped_seqs = []

  while hdf_dataset_in.is_less_than_num_seqs(seq_idx) and seq_idx <= float("inf"):
    if seq_idx % 1000 == 0:
      complete_frac =hdf_dataset_in.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))

    hdf_dataset_in.load_seqs(seq_idx, seq_idx + 1)
    data = hdf_dataset_in.get_data(seq_idx, "data")
    data_no_blank = data[data != blank_idx]
    # we always summarize reduction_factor frames into a single frame +
    # the remaining frames into an addition frame (ceil)
    red_data_len = int(np.ceil(len(data) / reduction_factor))

    # in this case, the sequence contains too many non-blanks to be reduced
    if red_data_len < len(data_no_blank):
      print("Skip Seq: ", data)
      skipped_seqs.append(hdf_dataset_in.get_tag(seq_idx))
      seq_idx += 1
      continue

    # reduce alignment
    curr_start = 0
    size = reduction_factor
    red_data = []
    queue = []
    # go through the data in slices of size reduction_factor
    for i in range(red_data_len):
      # get next slice
      data_slice = data[curr_start:curr_start+size]
      data_slice = np.array(data_slice)
      # slice without blank frames
      slice_no_blank = data_slice[data_slice != blank_idx]
      # if there are only blanks, check whether there are non-blanks in the queue
      # if yes, reduce to the first item of the queue and remove this item from the queue
      # else, reduce to blank
      if len(slice_no_blank) == 0:
        if len(queue) > 0:
          red_data.append(queue[0])
          queue.pop(0)
        else:
          red_data.append(blank_idx)
      # if there is a single non blank frame, check whether there are non-blanks in the queue
      # if yes, reduce to the first item of the queue, remove this item from the queue and add the current
      #   non-blank to the back of the queue
      # if no, reduce to the current non-blank
      elif len(slice_no_blank) == 1:
        if len(queue) > 0:
          red_data.append(queue[0])
          queue.pop(0)
          queue.append(slice_no_blank[0])
        else:
          red_data.append(slice_no_blank[0])
      # if there are more than one non-blanks, check whether there are non-blanks in the queue
      # if yes, reduce to the first item of the queue, remove this items from the queue and add all current non-blanks
      #   to the back of the queue
      # if no, reduce to the first of the current non-blanks and append the remaining current non-blanks to the back
      #   of the queue
      else:
        if len(queue) > 0:
          red_data.append(queue[0])
          queue.pop(0)
          queue += list(slice_no_blank)
        else:
          red_data.append(slice_no_blank[0])
          queue += list(slice_no_blank)[1:]

      # update current start
      curr_start += size
      # in this case, set size such that it is equal to the remaining number of frames
      if curr_start + size > len(data):
        size = len(data) - curr_start

    # add the remaining non-blanks from the queue to the reduced data
    red_data = list(red_data) + queue
    # in this case, we need to remove as many blanks from the reduced alignment as there are remaining
    # non-blanks in the queue
    if len(queue) > 0:
      blank_pos = np.where(np.array(red_data) == blank_idx)[0][-len(queue):]
      for i, idx in enumerate(blank_pos):
        # idx - i because the indices need to be updated after an element is removed
        red_data.pop(idx - i)

    red_data = np.array(red_data)
    red_data_no_blank = red_data[red_data != blank_idx]

    # assert that we still have the same non-blanks in the same order as in the non-reduced alignment
    assert list(data_no_blank) == list(red_data_no_blank)
    # assert that the reduced alignment has the correct length
    assert len(red_data) == int(np.ceil(len(data) / reduction_factor))

    # dump the reduced alignment into an hdf
    seq_len = len(red_data)
    tag = hdf_dataset_in.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(red_data, axis=0), dtype="int32")

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

  with open("skipped_seqs", "w+") as f:
    f.write(str(skipped_seqs))


def init(align_path, segment_file):
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
    "class": "HDFDataset", "files": [align_path], "use_cache_manager": True,
    "seq_list_filter_file": segment_file}

  hdf_dataset = rnn.init_dataset(hdf_align_opts)

  return hdf_dataset


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('align_path')
  arg_parser.add_argument('--segment_file')
  arg_parser.add_argument('--sil_idx', type=int)
  arg_parser.add_argument('--blank_idx', type=int)
  arg_parser.add_argument('--reduction_factor', type=int)
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  hdf_dataset_in = init(align_path=args.align_path, segment_file=args.segment_file)
  hdf_dataset_out = hdf_dataset_init(dim=hdf_dataset_in.get_data_dim("data"))
  try:
    dump(hdf_dataset_in, hdf_dataset_out, sil_idx=args.sil_idx, blank_idx=args.blank_idx, reduction_factor=args.reduction_factor)
    hdf_dataset_out.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
