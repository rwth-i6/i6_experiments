#!/usr/bin/env python3

"""
Creates a HDF file, which can be read by :class:`HDFDataset`.
The input is any other dataset (:class:`Dataset`).
"""

from __future__ import print_function

import json
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import math


def hdf_dump_from_dataset(dataset, parser_args):
  """
  :param Dataset dataset: could be any dataset implemented as child of Dataset
  :param hdf_dataset_mod.HDFDatasetWriter hdf_dataset:
  :param parser_args: argparse object from main()
  """
  seq_idx = 0
  end_idx = float("inf")
  sil_idx = parser_args.silence_idx
  time_reds = parser_args.time_reds
  dataset.init_seq_order(parser_args.epoch)
  target_dim = dataset.get_data_dim("classes")
  blank_idx = target_dim + 1  # idx target_dim is reserved for SOS token
  segs_to_skip = {}
  file_path = parser_args.out_file
  for red in time_reds:
    segs_to_skip[red] = []

  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_idx:
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "classes")

    # replace every frame label with blank except the last frame of each segment, which is the segment label
    # if sil_use_blanks is False, all frames of silence segments are labeled as silence, without blank labels
    blank_mask = [True if i == j else False for i, j in zip(data[:-1], data[1:])] + [False]
    data[blank_mask] = blank_idx

    data_no_blanks = data[data != blank_idx]
    # data_no_sil_no_blanks = data_no_blanks[data_no_blanks != sil_idx]

    for red in time_reds:
      if math.ceil(len(data) / red) < len(data_no_blanks):
        segs_to_skip[red].append(dataset.get_tag(seq_idx))

    seq_idx += 1

  with open(file_path, "w+") as f:
    json.dump(segs_to_skip, f)


def init(rasr_config_path, segment_file):
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

  estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}
  epoch_split = 1

  sprint_args = [
    "--config=%s" % rasr_config_path,
    "--*.LOGFILE=nn-trainer.train.log",
    "--*.TASK=1", "--*.corpus.segment-order-shuffle=true",
    "--*.reduce-alignment-factor=%d" % 1, "--*.corpus.segment-order-shuffle=true",
    "--*.segment-order-sort-by-time-length=true",
    "--*.corpus.segments.file=%s" % segment_file
  ]

  dataset_dict = {
    'class': 'ExternSprintDataset',
    "reduce_target_factor": 1,
    'sprintConfigStr': sprint_args,
    'sprintTrainerExecPath': '/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding',
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
  parser.add_argument('--out_file', help="Output directory")
  parser.add_argument('--epoch', type=int, default=1, help="Optional start epoch for initialization")
  parser.add_argument(
    '--time_reds', help="Time-downsampling factors to check alignments against", action="append", type=int)
  parser.add_argument('--returnn_root', help="Returnn root to use for imports")
  parser.add_argument('--segment_file')
  parser.add_argument('--silence-idx', help="Needs to be provided if silence-use-blanks is true", type=int, default=0)

  args = parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  dataset = init(rasr_config_path=args.rasr_config, segment_file=args.segment_file)
  hdf_dump_from_dataset(dataset, args)

  rnn_main.finalize()


if __name__ == '__main__':
  main(sys.argv)
