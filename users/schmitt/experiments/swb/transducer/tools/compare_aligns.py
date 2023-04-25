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
from matplotlib import ticker
import ast


def dump(meta_dataset, vocab1, vocab2, blank_idx1, blank_idx2, name1, name2, segment_path):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  seq_idx = 0

  while meta_dataset.is_less_than_num_seqs(seq_idx):
    if seq_idx % 1000 == 0:
      complete_frac = meta_dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))

    seq_tag = meta_dataset.get_tag(seq_idx)
    meta_dataset.load_seqs(seq_idx, seq_idx + 1)
    align1 = meta_dataset.get_data(seq_idx, "align1")
    align2 = meta_dataset.get_data(seq_idx, "data")

    plot_aligns(align1, align2, blank_idx1, blank_idx2, vocab1, vocab2, name1, name2, seq_tag)

    seq_idx += 1


def upscale_alignment(align, blank_idx, last_label_rep, time_red):
  align_upscale = []
  for i, label in enumerate(align):
    if last_label_rep == 0 or i != len(align) - 1:
      if label == blank_idx:
        align_upscale += [blank_idx] * time_red
      else:
        align_upscale += [blank_idx] * (time_red - 1) + [label]
    else:
      if label == blank_idx:
        align_upscale += [blank_idx] * last_label_rep
      else:
        align_upscale += [blank_idx] * (last_label_rep - 1) + [label]

  return np.array(align_upscale)

def plot_aligns(align1, align2, blank_idx1, blank_idx2, vocab1, vocab2, name1, name2, seq_tag):
  import subprocess
  assert len(align1) == len(align2)

  time_red = 6

  # get hmm alignment in list format
  hmm_aligns = [subprocess.check_output(
    ["/u/schmitt/experiments/transducer/config/sprint-executables/archiver", "--mode", "show", "--type", "align",
     "--allophone-file", "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/allophones",
     "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/tuske__2016_01_28__align.combined.train", tag]) for
    tag in [seq_tag]]
  hmm_aligns = [hmm_align.splitlines() for hmm_align in hmm_aligns]
  hmm_aligns = [[row.decode("utf-8").strip() for row in hmm_align] for hmm_align in hmm_aligns]
  hmm_aligns = [[row for row in hmm_align if row.startswith("time")] for hmm_align in hmm_aligns]
  hmm_align_states = [row.split("\t")[-1] for hmm_align in hmm_aligns for row in hmm_align]
  hmm_align = [row.split("\t")[5].split("{")[0] for hmm_align in hmm_aligns for row in hmm_align]
  hmm_align = [phon if phon != "[SILENCE]" else "[S]" for phon in hmm_align]

  hmm_align_borders = [i + 1 for i, x in enumerate(hmm_align) if
                       i < len(hmm_align) - 1 and (
                                 hmm_align[i] != hmm_align[i + 1] or hmm_align_states[i] > hmm_align_states[i + 1])]
  if hmm_align_borders[-1] != len(hmm_align):
    hmm_align_borders += [len(hmm_align)]
  hmm_align_major = [hmm_align[i - 1] for i in range(1, len(hmm_align) + 1) if i in hmm_align_borders]
  if hmm_align_borders[0] != 0:
    hmm_align_borders = [0] + hmm_align_borders
  hmm_align_borders_center = [(i + j) / 2 for i, j in zip(hmm_align_borders[:-1], hmm_align_borders[1:])]

  # upscale alignments to match hmm alignment
  last_label_rep = len(hmm_align) % time_red
  align1 = upscale_alignment(align1, blank_idx1, last_label_rep, time_red)
  align2 = upscale_alignment(align2, blank_idx2, last_label_rep, time_red)

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
  align2_labels = [vocab2[i] if i in vocab2 else "EOS" for i in align2_labels]

  plt.figure(figsize=(10, 5))
  hmm_ax = plt.gca()
  hmm_ax.matshow(matrix, aspect="auto", cmap=plt.cm.get_cmap("Blues"))
  # # create second x axis for hmm alignment labels and plot same matrix
  align2_ax = hmm_ax.twiny()
  align2_ax.set_xlim(hmm_ax.get_xlim())
  align1_ax = hmm_ax.twiny()
  align1_ax.set_xlim(hmm_ax.get_xlim())

  xticks = list(hmm_align_borders)[1:]
  xticks_minor = hmm_align_borders_center
  hmm_ax.set_xticks([tick - .5 for tick in xticks])
  hmm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
  hmm_ax.set_xticks([tick - .5 for tick in xticks_minor], minor=True)
  hmm_ax.set_xticklabels(hmm_align_major, minor=True, rotation=90)
  hmm_ax.tick_params(axis="x", which="minor", length=0, labelsize=10)
  hmm_ax.tick_params(axis="x", which="major", length=10, width=.5)
  hmm_ax.xaxis.tick_top()
  hmm_ax.xaxis.set_label_position('top')
  for idx in xticks:
    hmm_ax.axvline(x=idx - .5, ymin=0, ymax=1, color="k", linestyle="--", linewidth=.5)

  # # set x ticks and labels and positions for hmm axis
  align2_ax.set_xticks(align2_ticks)
  align2_ax.set_xticklabels(align2_labels, rotation="vertical")
  align2_ax.xaxis.set_ticks_position('bottom')
  align2_ax.xaxis.set_label_position('bottom')
  align2_ax.set_xlabel(name2)

  align1_ax.spines['top'].set_position(('outward', 36))
  align1_ax.set_xticks(align1_ticks)
  align1_ax.set_xticklabels(align1_labels, rotation="vertical")
  align1_ax.xaxis.set_ticks_position('top')
  align1_ax.xaxis.set_label_position('top')
  align1_ax.set_xlabel(name1)

  plt.tight_layout()
  # plt.subplots_adjust(bottom=.5)
  plt.savefig("plot_%s.png" % seq_tag.replace("/", "_"), #bbox_inches="tight"
              )
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
      name1=args.align1_name, name2=args.align2_name, segment_path=args.segment_path)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
