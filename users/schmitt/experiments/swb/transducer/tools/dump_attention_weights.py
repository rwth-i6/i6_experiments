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


def dump(hdf_dataset, label_name, model_type, blank_idx, segment_file):
  output_dict = {}
  if model_type == "seg":
    att_weights_layer = rnn.engine.network.get_layer("label_model/att_weights")
    seg_starts_layer = rnn.engine.network.get_layer("label_model/segment_starts")
    seg_lens_layer = rnn.engine.network.get_layer("label_model/segment_lens")
    output_dict["%s-out" % "att_weights"] = att_weights_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "seg_starts"] = seg_starts_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "seg_lens"] = seg_lens_layer.output.get_placeholder_as_batch_major()
  elif model_type in ["seg_lab_dep", "global-import"]:
    # output_log_prob_layer = rnn.engine.network.get_layer("output/output_log_prob")
    # output_dict["%s-out" % "output_log_prob"] = output_log_prob_layer.output.get_placeholder_as_batch_major()
    att_weights_layer = rnn.engine.network.get_layer("label_model/att_weights")
    seg_starts_layer = rnn.engine.network.get_layer("label_model/segment_starts")
    seg_lens_layer = rnn.engine.network.get_layer("label_model/segment_lens")
    output_dict["%s-out" % "att_weights"] = att_weights_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "seg_starts"] = seg_starts_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "seg_lens"] = seg_lens_layer.output.get_placeholder_as_batch_major()
  else:
    assert model_type == "glob"
    att_weights_layer = rnn.engine.network.get_layer("output/att_weights")
    output_dict["%s-out" % "att_weights"] = att_weights_layer.output.get_placeholder_as_batch_major()

  seq_idx = 0
  num_seqs = 0
  num_search_errors = 0
  with open(segment_file, "r") as f:
    segment_tag = f.read().strip()
    print("Correct tag: ", segment_tag)

  while hdf_dataset.is_less_than_num_seqs(seq_idx):
    if hdf_dataset.get_tag(seq_idx) != segment_tag:
      print("wrong tag: ", hdf_dataset.get_tag(seq_idx))
      seq_idx += 1
      continue
    num_seqs += 1
    hdf_dataset.load_seqs(seq_idx, seq_idx + 1)
    hdf_targets = hdf_dataset.get_data(seq_idx, label_name)
    print("SEQ TAG: ", hdf_dataset.get_tag(seq_idx))
    print("HDF TARGETS: ", hdf_targets)
    print("HDF TARGETS SHAPE: ", hdf_targets.shape)
    features = hdf_dataset.get_data(seq_idx, "data")
    print("INPUT FEATURES: ", features)
    print("INPUT FEATURES SHAPE: ", features.shape)
    out = rnn.engine.run_single(dataset=hdf_dataset, seq_idx=seq_idx, output_dict=output_dict)
    hdf_dataset.load_seqs(seq_idx, seq_idx + 1)
    hdf_targets = hdf_dataset.get_data(seq_idx, label_name)
    weights = out["att_weights-out"][0]

    data = {}
    if model_type.startswith("seg") or model_type == "global-import":
      # if model_type == "global-import" and hdf_targets[-1] != blank_idx:
      #   hdf_targets = hdf_targets[:-1]
      hdf_targets_non_blank = hdf_targets[hdf_targets != blank_idx]
      seg_starts = out["seg_starts-out"][0]
      seg_lens = out["seg_lens-out"][0]
      if seg_starts.shape[0] != hdf_targets_non_blank.shape[0]:
        assert seg_starts.shape[0] > hdf_targets_non_blank.shape[0]
        assert seg_lens.shape[0] > hdf_targets_non_blank.shape[0]
        assert weights.shape[0] > hdf_targets_non_blank.shape[0]
        seg_starts = seg_starts[hdf_targets != blank_idx]
        seg_lens = seg_lens[hdf_targets != blank_idx]
        weights = weights[hdf_targets != blank_idx]

      hdf_targets = hdf_targets_non_blank

      data.update({
        "seg_starts": seg_starts, "seg_lens": seg_lens})

    data["weights"] = weights

    # hdf_targets = np.append(hdf_targets, [0], axis=-1)

    np.savez(
      "data", labels=hdf_targets, **data)

    seq_idx += 1
    break

  with open("search_errors", "w+") as f:
    f.write(str(num_search_errors / num_seqs))


def net_dict_add_losses(net_dict):
  if "mean_seg_lens" in net_dict or ("label_model" in net_dict and "att_ctx" in net_dict["label_model"]["unit"] and "name_scope" in net_dict["label_model"]["unit"]["att_ctx"]):
    # in this case, we have the segmental model with label dep length model
    net_dict["label_model"]["unit"]["att_weights"]["is_output_layer"] = True
    net_dict["label_model"]["unit"]["segment_starts"]["is_output_layer"] = True
    net_dict["label_model"]["unit"]["segment_lens"]["is_output_layer"] = True
    # net_dict["label_model"]["unit"]["label_prob"]["loss"] = None
    # net_dict["label_model"]["unit"]["label_prob"]["is_output_layer"] = False
    # net_dict["output"]["emit_blank_prob"]["loss"] = None
    # net_dict["output"]["unit"]["segment_starts"]["is_output_layer"] = True
    # net_dict["output"]["unit"]["segment_lens"]["is_output_layer"] = True
    # net_dict["output"]["target"] = "alignment"
    # net_dict["output"]["optimize_move_layers_out"] = True
    # net_dict["output"]["unit"]["output"]["target"] = "alignment"
    # net_dict["output"]["unit"]["dump_weights"] = {
    #   "class": "hdf_dump", "from": "att_weights0", "filename": "att_weight_dump"
    # }
    # net_dict["output"]["unit"]["att_weights"]["from"] = "dump_weights"
    # net_dict["output"]["unit"]["output_log_prob"]["is_output_layer"] = True
  elif "label_model" in net_dict:
    # in this case, we have a segmental model
    net_dict["label_model"]["unit"]["att_weights"]["is_output_layer"] = True
    net_dict["label_model"]["unit"]["segment_starts"]["is_output_layer"] = True
    net_dict["label_model"]["unit"]["segment_lens"]["is_output_layer"] = True
  else:
    # in this case, we have a global model
    net_dict["output"]["unit"]["att_weights"]["is_output_layer"] = True

  return net_dict

def init(
  config_filename, segment_file, rasr_config_path, rasr_nn_trainer_exe, hdf_targets, label_name, concat_seqs,
  concat_hdf, concat_seq_tags_file):

  rnn.init(
    config_filename=config_filename,
    config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=rnn.train_data)
  rnn.engine.init_network_from_config(net_dict_post_proc=net_dict_add_losses)

  args = ["--config=%s" % rasr_config_path, #"--*.corpus.segment-order-shuffle=true",
          "--*.segment-order-sort-by-time-length=false", "--*.corpus.segment-order-shuffle=false",
          # "--*.segment-order-sort-by-time-length=true"
          ]

  d = {
    "class": "ExternSprintDataset", "sprintTrainerExecPath": rasr_nn_trainer_exe, "sprintConfigStr": args,
    "suppress_load_seqs_print": True,  # less verbose
    # "seq_list_filter_file": segment_file,
    "input_stddev": 3.}

  if not concat_seqs:
    hdf_targets_opts = {
      "seq_list_filter_file": segment_file,
      "class": "HDFDataset", "files": [hdf_targets], "use_cache_manager": True,
      "seq_ordering": "sorted",
      "seq_order_seq_lens_file": "/u/zeyer/setups/switchboard/dataset/data/seq-lens.train.txt.gz"
    }

    d_meta = {
      "class": "MetaDataset", "datasets": {"sprint": d, "align": hdf_targets_opts}, "data_map": {
        "data": ("sprint", "data"),
        label_name: ("align", "data"),
      }, "seq_order_control_dataset": "align", "seq_list_filter_file": segment_file,
    }
  else:
    if not concat_hdf:
      hdf_targets_opts = {
        "class": "HDFDataset", "files": [
          hdf_targets],
        "use_cache_manager": True
      }

      d_concat = {
        "class": "ConcatSeqsDataset", "dataset": d, #"repeat_in_between_last_frame_up_to_multiple_of": {"data": 6},
        "seq_len_file": "/u/schmitt/experiments/transducer/alias/stm_files/cv-concat-20/output/orig_seq_lens.py.txt",
        "seq_list_file": concat_seq_tags_file,
        "seq_ordering": "sorted_reverse", # "seq_list_filter_file": segment_file
      }

      d_meta = {
        "class": "MetaDataset", "datasets": {"sprint": d_concat, "align": hdf_targets_opts}, "data_map": {
          "data": ("sprint", "data"), label_name: ("align", "data"), }, "seq_order_control_dataset": "align",
        "seq_list_filter_file": segment_file, }
    else:
      hdf_targets_opts = {
        "class": "HDFDataset", "files": [
          hdf_targets],
        "use_cache_manager": True, # "seq_list_filter_file": segment_file,
      }
      d_meta = {
        "class": "MetaDataset", "datasets": {"sprint": d, "align": hdf_targets_opts}, "data_map": {
          "data": ("sprint", "data"), label_name: ("align", "data"), }, "seq_order_control_dataset": "align",
        "seq_list_filter_file": segment_file, }

      d_meta = {
        "class": "ConcatSeqsDataset", "dataset": d_meta, "repeat_in_between_last_frame_up_to_multiple_of": {"data": 6},
        "seq_len_file": "/u/schmitt/experiments/transducer/alias/stm_files/cv-concat-20/output/orig_seq_lens.py.txt",
        "seq_list_file": concat_seq_tags_file,
        "seq_ordering": "sorted_reverse"
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
  arg_parser.add_argument('--blank_idx', type=int)
  arg_parser.add_argument('--hdf_targets')
  arg_parser.add_argument('--label_name')
  arg_parser.add_argument('--model_type')
  arg_parser.add_argument('--concat_seqs', action="store_true")
  arg_parser.add_argument('--concat_hdf', action="store_true")
  arg_parser.add_argument('--concat_seq_tags_file')
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  meta_dataset = init(
    config_filename=args.returnn_config, segment_file=args.segment_file, rasr_config_path=args.rasr_config_path,
    rasr_nn_trainer_exe=args.rasr_nn_trainer_exe, label_name=args.label_name,
    hdf_targets=args.hdf_targets, concat_seqs=args.concat_seqs, concat_hdf=args.concat_hdf,
    concat_seq_tags_file=args.concat_seq_tags_file)
  dump(
    meta_dataset, label_name=args.label_name, model_type=args.model_type, blank_idx=args.blank_idx,
    segment_file=args.segment_file)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
