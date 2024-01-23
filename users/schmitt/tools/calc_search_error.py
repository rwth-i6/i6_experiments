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

global eos_idx


def dump(ref_dataset, search_dataset, blank_idx, label_name, model_type, max_seg_len, length_norm):
  output_dict = {}
  if model_type == "seg":
    label_log_prob_layer = rnn.engine.network.get_layer("label_model/label_log_prob")
    # dirty fix: try to get the sil_log_prob layer (which only exists in case of the separate silence model)
    # when it fails, just get the blank_log_prob layer which exists in any case
    # later, just check the name of the retrieved layer and prepend the sil log prob to the label log prob, only
    # when the sil_log_prob layer exists
    try:
      sil_log_prob_layer = rnn.engine.network.get_layer("label_model/sil_log_prob")
    except:
      sil_log_prob_layer = rnn.engine.network.get_layer("output/blank_log_prob")
    try:
      blank_log_prob_layer = rnn.engine.network.get_layer("output/blank_log_prob_scaled")
    except:
      blank_log_prob_layer = rnn.engine.network.get_layer("output/blank_log_prob")
    try:
      emit_log_prob_layer = rnn.engine.network.get_layer("output/emit_log_prob_scaled")
    except:
      emit_log_prob_layer = rnn.engine.network.get_layer("output/emit_log_prob")
    output_dict["%s-out" % "label_log_prob"] = label_log_prob_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "sil_log_prob"] = sil_log_prob_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "blank_log_prob"] = blank_log_prob_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "emit_log_prob"] = emit_log_prob_layer.output.get_placeholder_as_batch_major()
  elif model_type in ["seg_lab_dep", "global-import"]:
    output_log_prob_layer = rnn.engine.network.get_layer("output/output_log_prob")
    label_log_prob_layer = rnn.engine.network.get_layer("output/label_log_prob0")
    emit_log_prob_layer = rnn.engine.network.get_layer("output/emit_log_prob")
    output_dict["%s-out" % "output_log_prob"] = output_log_prob_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "label_log_prob"] = label_log_prob_layer.output.get_placeholder_as_batch_major()
    output_dict["%s-out" % "emit_log_prob"] = emit_log_prob_layer.output.get_placeholder_as_batch_major()
  else:
    assert model_type == "glob"
    label_prob_layer = rnn.engine.network.get_layer("output/label_prob")
    output_dict["%s-out" % "label_prob"] = label_prob_layer.output.get_placeholder_as_batch_major()

  seq_idx = 0
  num_seqs = 0
  num_diff_seqs = 0
  num_search_errors = 0
  num_search_errors_seg_bounds = 0
  only_blanks = 0
  num_over_max_seg_len = 0

  with open("search_error_log", "w+") as f:
    f.write("Search error log\n")
    f.write("-----------------------------\n\n")

  while ref_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= float("inf"):
    assert ref_dataset.get_tag(seq_idx) == search_dataset.get_tag(seq_idx)

    ref_out = rnn.engine.run_single(dataset=ref_dataset, seq_idx=seq_idx, output_dict=output_dict)
    ref_dataset.load_seqs(seq_idx, seq_idx + 1)
    ref_labels = ref_dataset.get_data(seq_idx, label_name)

    search_dataset.load_seqs(seq_idx, seq_idx + 1)
    search_labels = search_dataset.get_data(seq_idx, label_name)
    if model_type.startswith("seg") and len(search_labels[search_labels != blank_idx]) == 0:
      # in this case, the found seq just contains blanks and we just skip the seq
      only_blanks += 1
      seq_idx += 1
      continue

    if model_type.startswith("seg") and max_seg_len != -1:
      # in this case, the search was done with a maximum allowed segment size
      # if the reference seq contains segments which are longer, we need to skip bc the model had no way of finding
      # this segment during search
      label_positions = np.where(ref_labels != blank_idx)[0]
      label_positions_zero_prepend = np.concatenate([np.array([0]), label_positions])
      label_positions_len_append = np.concatenate([label_positions, np.array([len(ref_labels) - 1])])
      segment_lens = (label_positions_len_append - label_positions_zero_prepend) + 1
      if np.max(segment_lens) > max_seg_len:
        num_over_max_seg_len += 1
        seq_idx += 1
        continue

    if model_type == "seg":
      # get log probs for labels, blank and emit
      ref_label_log_prob = ref_out["label_log_prob-out"][0]  # [S, V]
      if sil_log_prob_layer.name == "sil_log_prob":
        # prepend the sil log prob to the label log prob (sil idx is always 0)
        ref_sil_log_prob = ref_out["sil_log_prob-out"][0]  # [S, 1]
        ref_label_log_prob = np.concatenate((ref_sil_log_prob, ref_label_log_prob), axis=-1)
      ref_blank_log_prob = ref_out["blank_log_prob-out"][0]  # [T, 1]
      ref_emit_log_prob = ref_out["emit_log_prob-out"][0]  # [T, 1]
      # choose the log probs for the labels in the alignment
      blank_mask = ref_labels == blank_idx
      non_blank_mask = ref_labels != blank_idx
      ref_labels_non_blank = ref_labels[ref_labels != blank_idx]
      ref_label_log_probs_seq = np.take_along_axis(ref_label_log_prob, ref_labels_non_blank[:, None], axis=-1)
      ref_emit_log_probs_seq = ref_emit_log_prob[non_blank_mask]
      # sum label and emit log probs to get the normalized label log probs
      ref_label_emit_log_probs_seq = ref_label_log_probs_seq + ref_emit_log_probs_seq
      # get blank log probs
      ref_blank_log_probs_seq = ref_blank_log_prob[blank_mask]
      # get seq log likelihood by summing over individual frames
      ref_output_log_prob = np.sum(ref_label_emit_log_probs_seq, axis=0) + np.sum(ref_blank_log_probs_seq, axis=0)

      # get log probs for labels, blank and emit
      search_label_log_prob = search_out["label_log_prob-out"][0]  # [S, V]
      if sil_log_prob_layer.name == "sil_log_prob":
        # prepend the sil log prob to the label log prob (sil idx is always 0)
        search_sil_log_prob = search_out["sil_log_prob-out"][0]  # [S, 1]
        search_label_log_prob = np.concatenate((search_sil_log_prob, search_label_log_prob), axis=-1)
      search_blank_log_prob = search_out["blank_log_prob-out"][0]  # [T, 1]
      search_emit_log_prob = search_out["emit_log_prob-out"][0]  # [T, 1]
      # choose the log probs for the labels in the alignment
      blank_mask = search_labels == blank_idx
      non_blank_mask = search_labels != blank_idx
      search_labels_non_blank = search_labels[search_labels != blank_idx]
      search_label_log_probs_seq = np.take_along_axis(search_label_log_prob, search_labels_non_blank[:, None], axis=-1)
      search_emit_log_probs_seq = search_emit_log_prob[non_blank_mask]
      # sum label and emit log probs to get the normalized label log probs
      search_label_emit_log_probs_seq = search_label_log_probs_seq + search_emit_log_probs_seq
      # get blank log probs
      search_blank_log_probs_seq = search_blank_log_prob[blank_mask]
      # get seq log likelihood by summing over individual frames
      search_output_log_prob = np.sum(search_label_emit_log_probs_seq, axis=0) + np.sum(search_blank_log_probs_seq, axis=0)

      ref_label_log_probs_seq = np.take_along_axis(ref_label_log_prob, ref_labels_non_blank[:, None], axis=-1)
      ref_label_log_prob_sum = np.sum(ref_label_log_probs_seq, axis=0)
      print("REF FEED LABEL LOG SCORES: %s \n" % ref_label_log_probs_seq)
      print("REF FEED LABEL LOG SCORES SUM: %s \n" % ref_label_log_prob_sum)
      print("REF FEED LABEL LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(ref_label_log_prob, axis=-1), np.max(ref_label_log_prob, axis=-1)))
      print("REF FEED EMIT LOG SCORES: %s \n" % ref_emit_log_probs_seq)
      print("REF FEED EMIT LOG SCORES SUM: %s \n" % np.sum(ref_emit_log_probs_seq, axis=0))
      print("REF FEED EMIT LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(ref_emit_log_prob, axis=-1), np.max(ref_emit_log_prob, axis=-1)))
      print("REF FEED BLANK LOG SCORES: %s \n" % ref_blank_log_probs_seq)
      print("REF FEED BLANK LOG SCORES SUM: %s \n" % np.sum(ref_blank_log_probs_seq, axis=0))
      print("REF FEED BLANK LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(ref_blank_log_prob, axis=-1), np.max(ref_blank_log_prob, axis=-1)))

      search_label_log_probs_seq = np.take_along_axis(search_label_log_prob, search_labels_non_blank[:, None], axis=-1)
      search_label_log_prob_sum = np.sum(search_label_log_probs_seq, axis=0)
      print("SEARCH FEED LABEL LOG SCORES: %s \n" % search_label_log_probs_seq)
      print("SEARCH FEED LABEL LOG SCORES SUM: %s \n" % search_label_log_prob_sum)
      print("SEARCH FEED LABEL LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(search_label_log_prob, axis=-1), np.max(search_label_log_prob, axis=-1)))
      print("SEARCH FEED EMIT LOG SCORES: %s \n" % search_emit_log_probs_seq)
      print("SEARCH FEED EMIT LOG SCORES SUM: %s \n" % np.sum(search_emit_log_probs_seq, axis=0))
      print("SEARCH FEED EMIT LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(search_emit_log_prob, axis=-1), np.max(search_emit_log_prob, axis=-1)))
      print("SEARCH FEED BLANK LOG SCORES: %s \n" % search_blank_log_probs_seq)
      print("SEARCH FEED BLANK LOG SCORES SUM: %s \n" % np.sum(search_blank_log_probs_seq, axis=0))
      print("SEARCH FEED BLANK LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(search_blank_log_prob, axis=-1), np.max(search_blank_log_prob, axis=-1)))
    elif model_type in ["seg_lab_dep", "global-import"]:
      if model_type == "global-import":
        if blank_idx == 1030:
          # if w/o silence, the last label needs to be EOS
          ref_labels[-1] = blank_idx + 1
          search_labels[-1] = blank_idx + 1
        elif blank_idx == 1031:
          if ref_labels[-1] != 0:
            # if w/ silence, we allow EOS and silence (0) as last label
            ref_labels[-1] = blank_idx + 1
          if search_labels[-1] != 0:
            # if w/ silence, we allow EOS and silence (0) as last label
            search_labels[-1] = blank_idx + 1
      ref_labels_non_blank = ref_labels[ref_labels != blank_idx]
      # get log probs for labels, blank and emit
      ref_label_prob = ref_out["output_log_prob-out"][0]  # [S, V]
      # choose the log probs for the labels in the alignment
      ref_label_log_probs_seq = np.take_along_axis(ref_label_prob, ref_labels[:, None], axis=-1)
      print("WHOLE REF SEQ PROBS: ", ref_label_log_probs_seq)
      # get seq log likelihood by summing over individual labels
      ref_output_log_prob = np.sum(ref_label_log_probs_seq, axis=0)

      search_labels_non_blank = search_labels[search_labels != blank_idx]
      # get log probs for labels, blank and emit
      search_label_prob = search_out["output_log_prob-out"][0]  # [S, V]
      # choose the log probs for the labels in the alignment
      search_label_log_probs_seq = np.take_along_axis(search_label_prob, search_labels[:, None], axis=-1)
      print("WHOLE SEARCH SEQ PROBS: ", search_label_log_probs_seq)
      # get seq log likelihood by summing over individual frames
      search_output_log_prob = np.sum(search_label_log_probs_seq, axis=0)

      if model_type == "global-import":
        if ref_labels[-1] == blank_idx + 1:
          ref_labels[-1] = 0 if blank_idx == 1030 else 1030
        if search_labels[-1] == blank_idx + 1:
          search_labels[-1] = 0 if blank_idx == 1030 else 1030
        ref_labels_non_blank = ref_labels[ref_labels != blank_idx]
        search_labels_non_blank = search_labels[search_labels != blank_idx]

      non_blank_mask = ref_labels != blank_idx
      ref_label_prob = ref_out["label_log_prob-out"][0][non_blank_mask]  # [S, V]
      ref_label_log_probs_seq = np.take_along_axis(ref_label_prob, ref_labels_non_blank[:, None], axis=-1)
      ref_label_log_prob_sum = np.sum(ref_label_log_probs_seq, axis=0)
      ref_emit_prob = ref_out["emit_log_prob-out"][0][non_blank_mask]  # [S, V]
      if ref_emit_prob.shape[-1] == 1:
        ref_emit_log_probs_seq = np.zeros(ref_emit_prob.shape)
      else:
        ref_emit_log_probs_seq = np.take_along_axis(ref_emit_prob, ref_labels_non_blank[:, None], axis=-1)
      ref_emit_log_prob_sum = np.sum(ref_emit_log_probs_seq, axis=0)
      print("REF FEED LABEL LOG SCORES: %s \n" % ref_label_log_probs_seq)
      print("REF FEED LABEL LOG SCORES SUM: %s \n" % ref_label_log_prob_sum)
      print("REF FEED LABEL LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(ref_label_prob, axis=-1), np.max(ref_label_prob, axis=-1)))
      print("REF FEED EMIT LOG SCORES: %s \n" % ref_emit_log_probs_seq)
      print("REF FEED EMIT LOG SCORES SUM: %s \n" % ref_emit_log_prob_sum)
      print("REF FEED EMIT LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(ref_emit_prob, axis=-1), np.max(ref_emit_prob, axis=-1)))

      non_blank_mask = search_labels != blank_idx
      search_label_prob = search_out["label_log_prob-out"][0][non_blank_mask]  # [S, V]
      search_label_log_probs_seq = np.take_along_axis(search_label_prob, search_labels_non_blank[:, None], axis=-1)
      search_label_log_prob_sum = np.sum(search_label_log_probs_seq, axis=0)
      search_emit_prob = search_out["emit_log_prob-out"][0][non_blank_mask]  # [S, V]
      if search_emit_prob.shape[-1] == 1:
        search_emit_log_probs_seq = np.zeros(search_emit_prob.shape)
      else:
        search_emit_log_probs_seq = np.take_along_axis(search_emit_prob, search_labels_non_blank[:, None], axis=-1)
      search_emit_log_prob_sum = np.sum(search_emit_log_probs_seq, axis=0)
      print("SEARCH FEED LABEL LOG SCORES: %s \n" % search_label_log_probs_seq)
      print("SEARCH FEED LABEL LOG SCORES SUM: %s \n" % search_label_log_prob_sum)
      print("SEARCH FEED LABEL LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(search_label_prob, axis=-1), np.max(search_label_prob, axis=-1)))
      print("SEARCH FEED EMIT LOG SCORES: %s \n" % search_emit_log_probs_seq)
      print("SEARCH FEED EMIT LOG SCORES SUM: %s \n" % search_emit_log_prob_sum)
      print("SEARCH FEED EMIT LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(search_emit_prob, axis=-1), np.max(search_emit_prob, axis=-1)))

      print("REF FEED LABEL LOG SCORE FOR FIRST SEARCH LABEL: %s \n" % ref_label_prob[0, search_labels_non_blank[0]])
      print("REF FEED LABEL LOG SCORE FOR FIRST REF LABEL: %s \n" % ref_label_prob[0, ref_labels_non_blank[0]])
      if ref_emit_prob.shape[-1] != 1:
        print("REF FEED EMIT LOG SCORE FOR FIRST SEARCH LABEL: %s \n" % ref_emit_prob[0, search_labels_non_blank[0]])
        print("REF FEED EMIT LOG SCORE FOR FIRST REF LABEL: %s \n" % ref_emit_prob[0, ref_labels_non_blank[0]])

      print("SEARCH FEED LABEL LOG SCORE FOR FIRST REF LABEL: %s \n" % search_label_prob[0, ref_labels_non_blank[0]])
      print("SEARCH FEED LABEL LOG SCORE FOR FIRST SEARCH LABEL: %s \n" % search_label_prob[0, search_labels_non_blank[0]])
      if search_emit_prob.shape[-1] != 1:
        print("SEARCH FEED EMIT LOG SCORE FOR FIRST REF LABEL: %s \n" % search_emit_prob[0, ref_labels_non_blank[0]])
        print("SEARCH FEED EMIT LOG SCORE FOR FIRST SEARCH LABEL: %s \n" % search_emit_prob[0, search_labels_non_blank[0]])
    else:
      ref_labels_non_blank = np.concatenate([ref_labels, [eos_idx]], axis=-1)
      # get log probs for labels, blank and emit
      ref_label_prob = ref_out["label_prob-out"][0]  # [S, V]
      # choose the log probs for the labels in the alignment
      ref_label_probs_seq = np.take_along_axis(ref_label_prob, ref_labels_non_blank[:, None], axis=-1)
      ref_label_log_probs_seq = np.log(ref_label_probs_seq)
      # get seq log likelihood by summing over individual labels
      ref_output_log_prob = np.sum(ref_label_log_probs_seq, axis=0)
      print("REF FEED LABEL LOG SCORES: %s \n" % ref_label_log_probs_seq)
      print("REF FEED LABEL LOG SCORES SUM: %s \n" % ref_output_log_prob)
      print("REF FEED LABEL LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(ref_label_prob, axis=-1), np.max(ref_label_prob, axis=-1)))

      search_labels_non_blank = np.concatenate([search_labels, [eos_idx]], axis=-1)
      # get log probs for labels, blank and emit
      search_label_prob = search_out["label_prob-out"][0]  # [S, V]
      # choose the log probs for the labels in the alignment
      search_label_probs_seq = np.take_along_axis(search_label_prob, search_labels_non_blank[:, None], axis=-1)
      search_label_log_probs_seq = np.log(search_label_probs_seq)
      # get seq log likelihood by summing over individual frames
      search_output_log_prob = np.sum(search_label_log_probs_seq, axis=0)
      print("SEARCH FEED LABEL LOG SCORES: %s \n" % search_label_log_probs_seq)
      print("SEARCH FEED LABEL LOG SCORES SUM: %s \n" % search_output_log_prob)
      print("SEARCH FEED LABEL LOG SCORES ARGMAX: %s, SCORE: %s \n" % (np.argmax(search_label_prob, axis=-1), np.max(search_label_prob, axis=-1)))

    num_seqs += 1

    if length_norm:
      search_output_log_prob /= len(search_labels_non_blank)
      ref_output_log_prob /= len(ref_labels_non_blank)

    print("SEQ IDX: %s \n" % seq_idx)
    print("TAG: %s \n" % ref_dataset.get_tag(seq_idx))
    print("REF ALIGN: %s \n" % ref_labels)
    print("REF FEED LOG SCORE: %s \n" % ref_output_log_prob)
    print("SEARCH ALIGN: %s \n" % search_labels)
    print("SEARCH FEED LOG SCORE: %s \n" % search_output_log_prob)
    print("-------------------------------------------------------------------------------")

    if list(search_labels_non_blank) == list(ref_labels_non_blank):
      if search_output_log_prob < ref_output_log_prob:
        # in this case, the found alignment is worse than the ref alignment but the non-blank labels are the same
        # so we do not count this as search errors
        with open("search_error_log", "a") as f:
          f.write("SEQ IDX: %s \n" % seq_idx)
          f.write("TAG: %s \n" % ref_dataset.get_tag(seq_idx))
          f.write("REF ALIGN: %s \n" % ref_labels)
          f.write("REF FEED LOG SCORE: %s \n" % ref_output_log_prob)
          f.write("SEARCH ALIGN: %s \n" % search_labels)
          f.write("SEARCH FEED LOG SCORE: %s \n" % search_output_log_prob)
          f.write("----> SAME LABEL SEQS, do not count as search error")
          f.write("-----------------------------\n\n")
        num_search_errors_seg_bounds += 1
    else:
      # in this case, the non-blank labels are different
      num_diff_seqs += 1
      if search_output_log_prob < ref_output_log_prob:
        # if the found seq is now worse than the ref seq, we have a search error
        with open("search_error_log", "a") as f:
          f.write("SEQ IDX: %s \n" % seq_idx)
          f.write("TAG: %s \n" % ref_dataset.get_tag(seq_idx))
          f.write("REF ALIGN: %s \n" % ref_labels)
          f.write("REF FEED LOG SCORE: %s \n" % ref_output_log_prob)
          f.write("SEARCH ALIGN: %s \n" % search_labels)
          f.write("SEARCH FEED LOG SCORE: %s \n" % search_output_log_prob)
          f.write("----> DIFFERENT LABEL SEQS, count as search error")
          f.write("-----------------------------\n\n")
        num_search_errors += 1

    seq_idx += 1

  with open("search_error_log", "a") as f:
    f.write("\n\n")
    f.write("NUMBER OF SEARCH ERRORS: " + str(num_search_errors) + "\n")
    f.write("NUMBER OF SEARCH ERRORS INCLUDING DIFFERENT SEG BOUNDARIES: " + str(num_search_errors + num_search_errors_seg_bounds) + "\n")
    f.write("RELATIVE SEARCH ERRORS (#search_errors/num_seqs): " + str(num_search_errors / num_seqs) + "\n")
    f.write("RELATIVE SEARCH ERRORS INCLUDING SEG BOUND ERRORS: " + str((num_search_errors + num_search_errors_seg_bounds) / num_seqs) + "\n")

  with open("search_errors", "w+") as f:
    f.write("SEARCH ERRORS: " + str(num_search_errors / num_seqs) + "\n")
    f.write("SEARCH ERRORS INCLUDING SEG BOUND ERRORS: " + str((num_search_errors + num_search_errors_seg_bounds) / num_seqs))

  print("Number of only-blank sequences: ", only_blanks)
  print("Number of sequences with too long segments: ", num_over_max_seg_len)
  print("Total number of seqs: ", num_seqs)


def net_dict_add_losses(net_dict):
  if "mean_seg_lens" in net_dict or ("att_ctx" in net_dict["output"]["unit"] and "name_scope" in net_dict["output"]["unit"]["att_ctx"]):
    # in this case, we have the segmental model with label dep length model
    net_dict["output"]["unit"]["output_log_prob"]["is_output_layer"] = True
    net_dict["output"]["unit"]["label_log_prob0"]["is_output_layer"] = True
    net_dict["output"]["unit"]["emit_log_prob"]["is_output_layer"] = True
    net_dict["output"]["target"] = "alignment"
    net_dict["output"]["optimize_move_layers_out"] = False
    net_dict["output"]["unit"]["output"]["target"] = "alignment"
  elif "label_model" in net_dict:
    # in this case, we have the segmental model
    net_dict["label_model"]["unit"]["label_log_prob"]["is_output_layer"] = True
    net_dict["output"]["unit"]["blank_log_prob"]["is_output_layer"] = True
    net_dict["output"]["unit"]["emit_log_prob"]["is_output_layer"] = True
    if "blank_log_prob_scaled" in net_dict["output"]["unit"]:
      net_dict["output"]["unit"]["blank_log_prob_scaled"]["is_output_layer"] = True
    if "emit_log_prob_scaled" in net_dict["output"]["unit"]:
      net_dict["output"]["unit"]["emit_log_prob_scaled"]["is_output_layer"] = True
  else:
    # in this case, we have the global model
    net_dict["output"]["unit"]["label_prob"]["is_output_layer"] = True

    global eos_idx
    eos_idx = net_dict["output"]["unit"]["output"]["initial_output"]

  return net_dict

def init(
  config_filename, segment_file, rasr_config_path, rasr_nn_trainer_exe, ref_targets, search_targets, label_name,
  concat_seqs, concat_seq_tags_file, model_type):
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

  if not concat_seqs:
    ref_align_opts = {
      "class": "HDFDataset", "files": [ref_targets], "use_cache_manager": True,
      "seq_list_filter_file": segment_file,
    }

    search_align_opts = {
      "class": "HDFDataset", "files": [search_targets], "use_cache_manager": True, "seq_list_filter_file": segment_file,
    }

    d_ref = {
      "class": "MetaDataset", "datasets": {"sprint": d, "align": ref_align_opts, "align2": search_align_opts},
      "data_map": {
        "data": ("sprint", "data"),
        label_name: ("align", "data"), "seq_order_dataset": ("align2", "data")
      }, "seq_order_control_dataset": "align2",
    }

    d_search = {
      "class": "MetaDataset", "datasets": {"sprint": d, "align": search_align_opts},
      "data_map": {
        "data": ("sprint", "data"),
        label_name: ("align", "data"),
      }, "seq_order_control_dataset": "align",
    }
  else:
    search_align_opts = {
      "class": "HDFDataset", "files": [search_targets],
      "use_cache_manager": True
    }

    d_search_concat = {
      "class": "ConcatSeqsDataset", "dataset": d,  # "repeat_in_between_last_frame_up_to_multiple_of": {"data": 6},
      "seq_len_file": "/u/schmitt/experiments/transducer/alias/stm_files/cv-concat-20/output/orig_seq_lens.py.txt",
      "seq_list_file": concat_seq_tags_file,
      "seq_ordering": "sorted_reverse",  # "seq_list_filter_file": segment_file
    }

    d_search = {
      "class": "MetaDataset", "datasets": {"sprint": d_search_concat, "align": search_align_opts}, "data_map": {
        "data": ("sprint", "data"), label_name: ("align", "data"), }, "seq_order_control_dataset": "align",
      "seq_list_filter_file": segment_file, }

    ref_align_opts = {
      "class": "HDFDataset", "files": [ref_targets],
      "use_cache_manager": True,  # "seq_list_filter_file": segment_file,
    }
    d_ref_meta = {
      "class": "MetaDataset", "datasets": {"sprint": d, "align": ref_align_opts}, "data_map": {
        "data": ("sprint", "data"), label_name: ("align", "data"), }, "seq_order_control_dataset": "align",
      "seq_list_filter_file": segment_file, }

    d_ref = {
      "class": "ConcatSeqsDataset", "dataset": d_ref_meta,
      "seq_len_file": "/u/schmitt/experiments/transducer/alias/stm_files/cv-concat-20/output/orig_seq_lens.py.txt",
      "seq_list_file": concat_seq_tags_file,
      "seq_ordering": "sorted_reverse"
    }

    if model_type != "glob":
      d_ref["repeat_in_between_last_frame_up_to_multiple_of"] = {"data": 6}

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
  arg_parser.add_argument('--ref_targets')
  arg_parser.add_argument('--search_targets')
  arg_parser.add_argument('--max_seg_len', type=int)
  arg_parser.add_argument('--label_name')
  arg_parser.add_argument('--model_type')
  arg_parser.add_argument('--length_norm', action="store_true")
  arg_parser.add_argument('--blank_idx', type=int)
  arg_parser.add_argument('--concat_seqs', action="store_true")
  arg_parser.add_argument('--concat_seq_tags_file')
  arg_parser.add_argument("--returnn_root", help="path to returnn root")
  args = arg_parser.parse_args(argv[1:])
  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn
  ref_dataset, search_dataset = init(
    config_filename=args.returnn_config, segment_file=args.segment_file, rasr_config_path=args.rasr_config_path,
    rasr_nn_trainer_exe=args.rasr_nn_trainer_exe, label_name=args.label_name,
    ref_targets=args.ref_targets, search_targets=args.search_targets, concat_seqs=args.concat_seqs,
    concat_seq_tags_file=args.concat_seq_tags_file, model_type=args.model_type)
  dump(
    ref_dataset, search_dataset, args.blank_idx, label_name=args.label_name, model_type=args.model_type,
    max_seg_len=args.max_seg_len, length_norm=args.length_norm)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
