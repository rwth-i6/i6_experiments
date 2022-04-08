#!/usr/bin/env python

"""
For debugging, go through some dataset, forward it through the net, and output the layer activations on stdout.
"""

from __future__ import print_function

import json
import sys

# import returnn.__main__ as rnn
# from returnn.log import log
import argparse
# from returnn.util.basic import pretty_print
from subprocess import check_output
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ast


def dump(meta_dataset, vocab1, vocab2, blank_idx1, blank_idx2, name1, name2):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  seq_idx = 0

  while meta_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= 0:
    if seq_idx % 1000 == 0:
      complete_frac = meta_dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))

    meta_dataset.load_seqs(seq_idx, seq_idx + 1)
    align1 = meta_dataset.get_data(seq_idx, "align1")
    align2 = meta_dataset.get_data(seq_idx, "data")

    plot_aligns(align1, align2, blank_idx1, blank_idx2, vocab1, vocab2, name1, name2, seq_idx)

    seq_idx += 1


def plot_aligns(align1, align2, blank_idx1, blank_idx2, vocab1, vocab2, name1, name2, seq_idx):
  assert len(align1) == len(align2)

  matrix = np.concatenate(
    [np.array([[0 if i == blank_idx1 else 1 for i in align1]]),
     np.array([[0 if i == blank_idx2 else 1 for i in align2]])],
    axis=0
  )

  align1_ticks = np.where(align1 != blank_idx1)[0]
  align1_labels = align1[align1 != blank_idx1]
  align1_labels = [vocab1[i] for i in align1_labels]

  align2_ticks = np.where(align2 != blank_idx2)[0]
  align2_labels = align2[align2 != blank_idx2]
  align2_labels = [vocab2[i] for i in align2_labels]

  plt.figure(figsize=(10, 5), constrained_layout=True)
  # plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
  ax = plt.gca()
  ax.matshow(matrix, aspect="auto", cmap=plt.cm.get_cmap("Blues"))
  # # create second x axis for hmm alignment labels and plot same matrix
  hmm_ax = ax.twiny()

  ax.set_xticks(list(align1_ticks))
  # ax.xaxis.set_major_formatter(ticker.NullFormatter())
  # ax.set_xticks(bpe_xticks_minor, minor=True)
  ax.set_xticklabels(list(align1_labels), rotation="vertical")
  # ax.tick_params(axis="x", which="minor", length=0, labelsize=17)
  # ax.tick_params(axis="x", which="major", length=10)
  ax.set_xlabel(name1)
  # ax.set_ylabel("Output RNA BPE Labels", fontsize=18)
  ax.xaxis.tick_top()
  ax.xaxis.set_label_position('top')
  # ax.spines['top'].set_position(('outward', 50))
  # # for tick in ax.xaxis.get_minor_ticks():
  # #   tick.label1.set_horizontalalignment("center")

  # # set x ticks and labels and positions for hmm axis
  hmm_ax.set_xticks(align2_ticks)
  # hmm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
  # hmm_ax.set_xticks(hmm_xticks_minor, minor=True)
  hmm_ax.set_xticklabels(align2_labels, rotation="vertical")
  # hmm_ax.tick_params(axis="x", which="minor", length=0)
  # hmm_ax.tick_params(axis="x", which="major", length=0)
  hmm_ax.xaxis.set_ticks_position('bottom')
  hmm_ax.xaxis.set_label_position('bottom')
  hmm_ax.set_xlabel(name2)


  plt.savefig("plot%s.png" % seq_idx, bbox_inches="tight")
  plt.close()


def init(align1_path, align2_path, segment_path, vocab1_path, vocab2_path):
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

  hdf_align1_opts = {
    "class": "HDFDataset", "files": [align1_path], "use_cache_manager": True,
    "seq_list_filter_file": segment_path}
  hdf_align2_opts = {
    "class": "HDFDataset", "files": [align2_path], "use_cache_manager": True, "seq_list_filter_file": segment_path}
  meta_dict = {
    'class': 'MetaDataset',
    'data_map': {'align1': ('align1', 'data'), 'data': ('align2', 'data')},
    'datasets': {
      'align1': hdf_align1_opts, "align2": hdf_align2_opts}, 'seq_order_control_dataset': 'align2'}

  hdf_dataset = rnn.init_dataset(meta_dict)

  with open(vocab1_path, "r") as f:
    vocab1 = ast.literal_eval(f.read())
    vocab1 = {int(v): k for k, v in vocab1.items()}

  with open(vocab2_path, "r") as f:
    vocab2 = ast.literal_eval(f.read())
    vocab2 = {int(v): k for k, v in vocab2.items()}

  return hdf_dataset, vocab1, vocab2


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('align1_path')
  arg_parser.add_argument('align2_path')
  arg_parser.add_argument('--segment_path')
  arg_parser.add_argument('--blank_idx_align1', type=int)
  arg_parser.add_argument('--blank_idx_align2', type=int)
  arg_parser.add_argument('--vocab_align1')
  arg_parser.add_argument('--vocab_align2')
  arg_parser.add_argument('--align1_name')
  arg_parser.add_argument('--align2_name')
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  meta_dataset, vocab1, vocab2 = init(
    align1_path=args.align1_path, align2_path=args.align2_path, segment_path=args.segment_path,
    vocab1_path=args.vocab_align1, vocab2_path=args.vocab_align2)
  try:
    dump(
      meta_dataset, vocab1=vocab1, vocab2=vocab2, blank_idx1=args.blank_idx_align1, blank_idx2=args.blank_idx_align2,
      name1=args.align1_name, name2=args.align2_name,)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
