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


def dump(ref_dataset, search_dataset, blank_idx):
  output_dict = {}
  label_log_prob_layer = rnn.engine.network.get_layer("label_model/label_log_prob")
  blank_log_prob_layer = rnn.engine.network.get_layer("output/blank_log_prob")
  emit_log_prob_layer = rnn.engine.network.get_layer("output/emit_log_prob")
  output_dict["%s-out" % "label_log_prob"] = label_log_prob_layer.output.get_placeholder_as_batch_major()
  output_dict["%s-out" % "blank_log_prob"] = blank_log_prob_layer.output.get_placeholder_as_batch_major()
  output_dict["%s-out" % "emit_log_prob"] = emit_log_prob_layer.output.get_placeholder_as_batch_major()

  seq_idx = 0
  num_seqs = 0
  num_search_errors = 0
  only_blanks = 0

  with open("search_error_log", "w+") as f:
    f.write("Search error log\n")
    f.write("-----------------------------\n\n")

  while ref_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= float("inf"):
    assert ref_dataset.get_tag(seq_idx) == search_dataset.get_tag(seq_idx)
    num_seqs += 1
    ref_out = rnn.engine.run_single(dataset=ref_dataset, seq_idx=seq_idx, output_dict=output_dict)
    ref_dataset.load_seqs(seq_idx, seq_idx + 1)
    ref_align = ref_dataset.get_data(seq_idx, "alignment")

    # get log probs for labels, blank and emit
    ref_label_log_prob = ref_out["label_log_prob-out"][0]  # [S, V]
    ref_blank_log_prob = ref_out["blank_log_prob-out"][0]  # [T, 1]
    ref_emit_log_prob = ref_out["emit_log_prob-out"][0]  # [T, 1]
    # choose the log probs for the labels in the alignment
    ref_labels = ref_align[ref_align != blank_idx]
    ref_label_log_probs_seq = np.take_along_axis(ref_label_log_prob, ref_labels[:, None], axis=-1)
    ref_emit_log_probs_seq = ref_emit_log_prob[ref_align != blank_idx]
    # sum label and emit log probs to get the normalized label log probs
    ref_label_emit_log_probs_seq = ref_label_log_probs_seq + ref_emit_log_probs_seq
    # get blank log probs
    ref_blank_log_probs_seq = ref_blank_log_prob[ref_align == blank_idx]
    # get seq log likelihood by summing over individual frames
    ref_output_log_prob = np.sum(ref_label_emit_log_probs_seq, axis=0) + np.sum(ref_blank_log_probs_seq, axis=0)

    search_dataset.load_seqs(seq_idx, seq_idx + 1)
    search_align = search_dataset.get_data(seq_idx, "alignment")
    if len(search_align[search_align != blank_idx]) == 0:
      only_blanks += 1
      seq_idx += 1
      continue
    search_out = rnn.engine.run_single(dataset=search_dataset, seq_idx=seq_idx, output_dict=output_dict)
    # get log probs for labels, blank and emit
    search_label_log_prob = search_out["label_log_prob-out"][0]  # [S, V]
    search_blank_log_prob = search_out["blank_log_prob-out"][0]  # [T, 1]
    search_emit_log_prob = search_out["emit_log_prob-out"][0]  # [T, 1]
    # choose the log probs for the labels in the alignment
    search_labels = search_align[search_align != blank_idx]
    search_label_log_probs_seq = np.take_along_axis(search_label_log_prob, search_labels[:, None], axis=-1)
    search_emit_log_probs_seq = search_emit_log_prob[search_align != blank_idx]
    # sum label and emit log probs to get the normalized label log probs
    search_label_emit_log_probs_seq = search_label_log_probs_seq + search_emit_log_probs_seq
    # get blank log probs
    search_blank_log_probs_seq = search_blank_log_prob[search_align == blank_idx]
    # get seq log likelihood by summing over individual frames
    search_output_log_prob = np.sum(search_label_emit_log_probs_seq, axis=0) + np.sum(search_blank_log_probs_seq, axis=0)

    # print("REF ALIGN: ", ref_align)
    # print("REF Labels: ", ref_labels)
    # print("REF LABEL LOG PROB SEQ: ", ref_label_log_probs_seq)
    # print("REF EMIT LOG PROB SEQ: ", ref_emit_log_probs_seq)
    # print("REF BLANK LOG PROB SEQ: ", ref_blank_log_probs_seq)
    # print("REF LABEL EMIT LOG PROB SEQ: ", ref_label_emit_log_probs_seq)
    # print("REF OUTPUT LOG PROB SEQ: ", ref_output_log_prob)

    with open("search_error_log", "a") as f:
      f.write("SEQ IDX: %s \n" % seq_idx)
      f.write("TAG: %s \n" % ref_dataset.get_tag(seq_idx))
      f.write("REF ALIGN: %s \n" % ref_align)
      f.write("REF FEED LOG SCORE: %s \n" % ref_output_log_prob)
      f.write("SEARCH ALIGN: %s \n" % search_align)
      f.write("SEARCH FEED LOG SCORE: %s \n" % search_output_log_prob)
      f.write("-----------------------------\n\n")

    if search_output_log_prob < ref_output_log_prob:
      num_search_errors += 1

    seq_idx += 1

  with open("search_errors", "w+") as f:
    f.write(str(num_search_errors / num_seqs))

  print("Number of only-blank sequences: ", only_blanks)


def net_dict_add_losses(net_dict):
  net_dict["label_model"]["unit"]["label_log_prob"]["is_output_layer"] = True
  net_dict["output"]["unit"]["blank_log_prob"]["is_output_layer"] = True
  net_dict["output"]["unit"]["emit_log_prob"]["is_output_layer"] = True

  return net_dict

def init(config_filename, segment_file, rasr_config_path, rasr_nn_trainer_exe, ref_align, search_align):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  rnn.init(
    config_filename=config_filename,
    config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=rnn.train_data)
  rnn.engine.init_network_from_config(net_dict_post_proc=net_dict_add_losses)

  args = ["--config=%s" % rasr_config_path, #"--*.corpus.segment-order-shuffle=true",
          # "--*.segment-order-sort-by-time-length=true"
          ]

  d = {
    "class": "ExternSprintDataset", "sprintTrainerExecPath": rasr_nn_trainer_exe, "sprintConfigStr": args,
    "suppress_load_seqs_print": True,  # less verbose
    "input_stddev": 3.}

  ref_align_opts = {
    "class": "HDFDataset", "files": [ref_align], "use_cache_manager": True,
    "seq_list_filter_file": segment_file,
    # "seq_ordering": "sorted",
    # "seq_order_seq_lens_file": "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
  }

  search_align_opts = {
    "class": "HDFDataset", "files": [search_align], "use_cache_manager": True, "seq_list_filter_file": segment_file,
    # "seq_ordering": "sorted",
    # "seq_order_seq_lens_file": "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
  }

  d_ref = {
    "class": "MetaDataset", "datasets": {"sprint": d, "align": ref_align_opts}, "data_map": {
      "data": ("sprint", "data"),
      "alignment": ("align", "data"),
    }, "seq_order_control_dataset": "align",
  }

  d_search = {
    "class": "MetaDataset", "datasets": {"sprint": d, "align": search_align_opts, "align2": ref_align_opts},
    "data_map": {
      "data": ("sprint", "data"),
      "alignment": ("align", "data"), "seq_order_dataset": ("align2", "data"),
    }, "seq_order_control_dataset": "align2",
  }

  ref_dataset = rnn.init_dataset(d_ref)
  search_dataset = rnn.init_dataset(d_search)

  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()

  return ref_dataset, search_dataset


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--rasr_config_path')
  arg_parser.add_argument('--rasr_nn_trainer_exe')
  arg_parser.add_argument('--segment_file')
  arg_parser.add_argument('--ref_align')
  arg_parser.add_argument('--search_align')
  arg_parser.add_argument('--blank_idx', type=int)
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  ref_dataset, search_dataset = init(
    config_filename=args.returnn_config, segment_file=args.segment_file, rasr_config_path=args.rasr_config_path,
    rasr_nn_trainer_exe=args.rasr_nn_trainer_exe,
    ref_align=args.ref_align, search_align=args.search_align)
  dump(ref_dataset, search_dataset, args.blank_idx)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
