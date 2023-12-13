#!/usr/bin/env python

"""
Take two HDF alignments of the same corpus and a model as input and produce a third alignment, which consists of the
alignments which result in the best model score.
"""

from __future__ import print_function

import sys
import argparse
import numpy as np
import tensorflow as tf


def hdf_dataset_init(dim: int, output_path: str):
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename=output_path, dim=dim, ndim=1)


def hdf_dataset_insert_seq(seq_len, tag, data, out_dataset):
  """
  Insert sequence into hdf dataset
  :param seq_len:
  :param tag:
  :param data:
  :param out_dataset:
  :return:
  """
  new_data = np.array(np.expand_dims(data, axis=0), dtype="int32")

  extra = {}
  seq_lens = {0: np.array([seq_len])}
  ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
  for dim in range(ndim_without_features):
    if dim not in seq_lens:
      seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
  batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
  for i, (axis, size) in enumerate(sorted(seq_lens.items())):
    batch_seq_sizes[:, i] = size
  extra["seq_sizes"] = batch_seq_sizes

  out_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)


def get_seq_log_score(label_log_probs, blank_log_probs, emit_log_probs, labels, blank_idx):
  """
  Get output log scores by combining label, blank and emit log scores
  :param label_log_probs:
  :param blank_log_probs:
  :param emit_log_probs:
  :param labels:
  :param blank_idx:
  :return:
  """
  blank_mask = labels == blank_idx
  non_blank_mask = labels != blank_idx
  labels_non_blank = labels[labels != blank_idx]
  label_log_probs_seq = np.take_along_axis(label_log_probs, labels_non_blank[:, None], axis=-1)
  emit_log_probs_seq = emit_log_probs[non_blank_mask]
  # sum label and emit log probs to get the normalized label log probs
  label_emit_log_probs_seq = label_log_probs_seq + emit_log_probs_seq
  # get blank log probs
  blank_log_probs_seq = blank_log_probs[blank_mask]
  # get seq log likelihood by summing over individual frames
  output_log_prob = np.sum(label_emit_log_probs_seq, axis=0) + np.sum(blank_log_probs_seq, axis=0)

  return output_log_prob


def dump_new_dataset(align1_dataset, align2_dataset, out_dataset, blank_idx, label_name):
  output_dict = {}
  label_log_prob_layer = rnn.engine.network.get_layer("label_model/label_log_prob")
  try:
    blank_log_prob_layer = rnn.engine.network.get_layer("output/blank_log_prob_scaled")
  except:
    blank_log_prob_layer = rnn.engine.network.get_layer("output/blank_log_prob")
  try:
    emit_log_prob_layer = rnn.engine.network.get_layer("output/emit_log_prob_scaled")
  except:
    emit_log_prob_layer = rnn.engine.network.get_layer("output/emit_log_prob")
  output_dict["%s-out" % "label_log_prob"] = label_log_prob_layer.output.get_placeholder_as_batch_major()
  output_dict["%s-out" % "blank_log_prob"] = blank_log_prob_layer.output.get_placeholder_as_batch_major()
  output_dict["%s-out" % "emit_log_prob"] = emit_log_prob_layer.output.get_placeholder_as_batch_major()

  seq_idx = 0
  num_seqs = 0

  while align1_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= float("inf"):
    assert align1_dataset.get_tag(seq_idx) == align2_dataset.get_tag(seq_idx)
    if seq_idx % 1000 == 0:
      complete_frac = align1_dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))

    align1_out = rnn.engine.run_single(dataset=align1_dataset, seq_idx=seq_idx, output_dict=output_dict)
    align1_dataset.load_seqs(seq_idx, seq_idx + 1)
    align1_labels = align1_dataset.get_data(seq_idx, label_name)
    align1_align = align1_dataset.get_data(seq_idx, label_name)

    align2_dataset.load_seqs(seq_idx, seq_idx + 1)
    align2_labels = align2_dataset.get_data(seq_idx, label_name)
    align2_align = align2_dataset.get_data(seq_idx, label_name)
    align2_out = rnn.engine.run_single(dataset=align2_dataset, seq_idx=seq_idx, output_dict=output_dict)

    print("REF ALIGN: ", align1_align)
    print("SEARCH ALIGN: ", align2_align)

    align1_output_log_prob = get_seq_log_score(
      label_log_probs=align1_out["label_log_prob-out"][0],
      emit_log_probs=align1_out["emit_log_prob-out"][0],
      blank_log_probs=align1_out["blank_log_prob-out"][0],
      blank_idx=blank_idx,
      labels=align1_labels,
    )

    align2_output_log_prob = get_seq_log_score(
      label_log_probs=align2_out["label_log_prob-out"][0],
      emit_log_probs=align2_out["emit_log_prob-out"][0],
      blank_log_probs=align2_out["blank_log_prob-out"][0],
      blank_idx=blank_idx,
      labels=align2_labels, )

    print("ALIGN 1 SCORE: ", align1_output_log_prob)
    print("ALIGN 2 SCORE: ", align2_output_log_prob)

    if align1_output_log_prob > align2_output_log_prob:
      keep_alignment = align1_align
    else:
      keep_alignment = align2_align

    print("KEEP ALIGN: ", keep_alignment)

    hdf_dataset_insert_seq(
      seq_len=align1_dataset.get_seq_length(seq_idx)[label_name],
      tag=align1_dataset.get_tag(seq_idx),
      data=keep_alignment,
      out_dataset=out_dataset
    )

    num_seqs += 1
    seq_idx += 1


def net_dict_add_output_layers(net_dict):
  """
  Set the log prob layers as output layers in order to be able to retrieve the scores.
  :param net_dict:
  :return:
  """
  net_dict["label_model"]["unit"]["label_log_prob"]["is_output_layer"] = True
  net_dict["output"]["unit"]["blank_log_prob"]["is_output_layer"] = True
  net_dict["output"]["unit"]["emit_log_prob"]["is_output_layer"] = True
  if "blank_log_prob_scaled" in net_dict["output"]["unit"]:
    net_dict["output"]["unit"]["blank_log_prob_scaled"]["is_output_layer"] = True
  if "emit_log_prob_scaled" in net_dict["output"]["unit"]:
    net_dict["output"]["unit"]["emit_log_prob_scaled"]["is_output_layer"] = True

  return net_dict


def init(
        config_filename,
        segment_path,
        rasr_config_path,
        rasr_nn_trainer_exe,
        align1_hdf_path,
        align2_hdf_path,
        label_name
):
  rnn.init(
    config_filename=config_filename,
    config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  rnn.engine.init_train_from_config(config=rnn.config, train_data=rnn.train_data)
  rnn.engine.init_network_from_config(net_dict_post_proc=net_dict_add_output_layers)

  # sprint dataset
  d = {
    "class": "ExternSprintDataset",
    "sprintTrainerExecPath": rasr_nn_trainer_exe,
    "sprintConfigStr": "--config=%s" % rasr_config_path,
    "suppress_load_seqs_print": True,
    "input_stddev": 3.}

  # hdf dataset 1
  align1_align_opts = {
    "class": "HDFDataset",
    "files": [align1_hdf_path],
    "use_cache_manager": True,
    "seq_list_filter_file": segment_path,
  }

  # hdf dataset 2
  align2_align_opts = {
    "class": "HDFDataset",
    "files": [align2_hdf_path],
    "use_cache_manager": True,
    "seq_list_filter_file": segment_path,
  }

  # combine sprint dataset + hdf dataset 1
  d_ref = {
    "class": "MetaDataset",
    "datasets": {"sprint": d, "align": align1_align_opts, "align2": align2_align_opts},
    "data_map": {
      "data": ("sprint", "data"),
      label_name: ("align", "data"), "seq_order_dataset": ("align2", "data")
    },
    "seq_order_control_dataset": "align2",
  }

  # combine sprint dataset + hdf dataset 2
  d_search = {
    "class": "MetaDataset",
    "datasets": {"sprint": d, "align": align2_align_opts},
    "data_map": {
      "data": ("sprint", "data"),
      label_name: ("align", "data"),
    },
    "seq_order_control_dataset": "align",
  }

  align1_dataset = rnn.init_dataset(d_ref)
  align2_dataset = rnn.init_dataset(d_search)

  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()

  return align1_dataset, align2_dataset


def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--rasr_config_path')
  arg_parser.add_argument('--rasr_nn_trainer_exe')
  arg_parser.add_argument('--segment_path')
  arg_parser.add_argument('--align1_hdf_path')
  arg_parser.add_argument('--align2_hdf_path')
  arg_parser.add_argument('--label_name')
  arg_parser.add_argument('--blank_idx', type=int)
  arg_parser.add_argument('--output_path')
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn

  align1_dataset, align2_dataset = init(
    config_filename=args.returnn_config,
    segment_path=args.segment_path,
    rasr_config_path=args.rasr_config_path,
    rasr_nn_trainer_exe=args.rasr_nn_trainer_exe,
    label_name=args.label_name,
    align1_hdf_path=args.align1_hdf_path,
    align2_hdf_path=args.align2_hdf_path)
  out_dataset = hdf_dataset_init(dim=align1_dataset.get_data_dim(args.label_name), output_path=args.output_path)
  try:
    dump_new_dataset(
      align1_dataset=align1_dataset,
      align2_dataset=align2_dataset,
      out_dataset=out_dataset,
      blank_idx=args.blank_idx,
      label_name=args.label_name)
    out_dataset.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit()
  finally:
    rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
