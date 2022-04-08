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


def dump(hdf_dataset):
  output_dict = {}
  att_weights_layer = rnn.engine.network.get_layer("label_model/att_weights")
  seg_starts_layer = rnn.engine.network.get_layer("label_model/segment_starts")
  seg_lens_layer = rnn.engine.network.get_layer("label_model/segment_lens")
  output_dict["%s-out" % "att_weights"] = att_weights_layer.output.get_placeholder_as_batch_major()
  output_dict["%s-out" % "seg_starts"] = seg_starts_layer.output.get_placeholder_as_batch_major()
  output_dict["%s-out" % "seg_lens"] = seg_lens_layer.output.get_placeholder_as_batch_major()

  seq_idx = 0
  num_seqs = 0
  num_search_errors = 0

  while hdf_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= 0:
    num_seqs += 1
    out = rnn.engine.run_single(dataset=hdf_dataset, seq_idx=seq_idx, output_dict=output_dict)
    hdf_dataset.load_seqs(seq_idx, seq_idx + 1)
    hdf_align = hdf_dataset.get_data(seq_idx, "alignment")

    # get attention weights
    att_weights = out["att_weights-out"][0]  # [S, seg_len]
    seg_starts = out["seg_starts-out"][0]  # [S]
    seg_lens = out["seg_lens-out"][0]  # [S]

    np.savez(
      "data", weights=att_weights, seg_starts=seg_starts, seg_lens=seg_lens,
      align=hdf_align)

    seq_idx += 1

  with open("search_errors", "w+") as f:
    f.write(str(num_search_errors / num_seqs))


def net_dict_add_losses(net_dict):
  net_dict["label_model"]["unit"]["att_weights"]["is_output_layer"] = True
  net_dict["label_model"]["unit"]["segment_starts"]["is_output_layer"] = True
  net_dict["label_model"]["unit"]["segment_lens"]["is_output_layer"] = True

  return net_dict

def init(config_filename, segment_file, rasr_config_path, rasr_nn_trainer_exe, hdf_align):
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
          "--*.segment-order-sort-by-time-length=true"
          ]

  d = {
    "class": "ExternSprintDataset", "sprintTrainerExecPath": rasr_nn_trainer_exe, "sprintConfigStr": args,
    "suppress_load_seqs_print": True,  # less verbose
    # "seq_list_filter_file": segment_file,
    "input_stddev": 3.}

  hdf_align_opts = {
    "class": "HDFDataset", "files": [hdf_align], "use_cache_manager": True,
    "seq_list_filter_file": segment_file,
    # "seq_ordering": "sorted",
    # "seq_order_seq_lens_file": "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
  }

  d_meta = {
    "class": "MetaDataset", "datasets": {"sprint": d, "align": hdf_align_opts}, "data_map": {
      "data": ("sprint", "data"),
      "alignment": ("align", "data"),
    }, "seq_order_control_dataset": "align", "seq_list_filter_file": segment_file,
  }

  meta_dataset = rnn.init_dataset(d_meta)

  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()

  return meta_dataset


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--rasr_config_path')
  arg_parser.add_argument('--rasr_nn_trainer_exe')
  arg_parser.add_argument('--segment_file')
  arg_parser.add_argument('--hdf_align')
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  meta_dataset = init(
    config_filename=args.returnn_config, segment_file=args.segment_file, rasr_config_path=args.rasr_config_path,
    rasr_nn_trainer_exe=args.rasr_nn_trainer_exe,
    hdf_align=args.hdf_align)
  dump(meta_dataset)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
