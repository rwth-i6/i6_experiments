#!/usr/bin/env python

"""
For debugging, go through some dataset, forward it through the net, and output the layer activations on stdout.
"""

from __future__ import print_function

import sys
import numpy as np
import os
import ast
import subprocess
import tensorflow as tf
import pickle

# import _setup_returnn_env  # noqa
# import returnn.tf.layers.base
# from returnn.log import log
import argparse
from matplotlib import pyplot as plt
from matplotlib import ticker
# from returnn.util.basic import pretty_print


def dump(dataset, options):
  """
  :type dataset: Dataset.Dataset
  :param options: argparse.Namespace
  """
  print("Epoch: %i" % options.epoch, file=returnn.log.log.v3)
  dataset.init_seq_order(options.epoch)
  print("LABELS: ", dataset.labels)

  layers = ["encoder"]
  rec_layers = ["att_weights", "att_query", "att_ctx", "att_val", "att", "att_energy", "output", "am", "s", "lm", "readout", "prev_out_embed"]
  if rnn.config.typed_value("att_area") == "seg":
    rec_layers.append("segment_starts")
    # rec_layers.append("att_unmasked")
  np.set_printoptions(threshold=np.inf)

  if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)

  output_dict = {}
  # get the output of layers outside the rec loop
  for name in layers:
    layer = rnn.engine.network.get_layer(name)
    output_dict["%s-out" % name] = layer.output.get_placeholder_as_batch_major()
    with open(os.path.join(options.out_dir, name + "-shape"), "w+") as f:
      f.write(repr(layer.output.dim_tags))
  # get the output of layers inside the rec loop
  for name in rec_layers:
    layer = rnn.engine.network.get_layer("output/" + name)
    output_dict["%s-out" % name] = layer.output.get_placeholder_as_batch_major()
    with open(os.path.join(options.out_dir, name + "-shape"), "w+") as f:
      f.write(repr(layer.output.dim_tags))

  # lm_masked_layer = rnn.engine.network.get_layer("output/lm_masked")
  # if isinstance(lm_masked_layer, returnn.tf.layers.base.WrappedInternalLayer):
  #   lm_embed_layer = lm_masked_layer.base_layer.sub_layer.subnetwork.layers["input_embed"]
  # else:
  #   lm_embed_layer = rnn.engine.network.get_layer("output/lm_masked").sub_layer.subnetwork.layers["input_embed"]
  # output_dict["lm_embed-out"] = lm_embed_layer.output.get_placeholder_as_batch_major()

  # get additional information
  output_dict["output_len"] = rnn.engine.network.get_layer("output").output.get_sequence_lengths()
  output_dict["encoder_len"] = rnn.engine.network.get_layer("encoder").output.get_sequence_lengths()
  output_dict["seq_idx"] = rnn.engine.network.get_extern_data("seq_idx").get_placeholder_as_batch_major()
  output_dict["seq_tag"] = rnn.engine.network.get_extern_data("seq_tag").get_placeholder_as_batch_major()
  output_dict["target_data"] = rnn.engine.network.get_extern_data(
    rnn.engine.network.extern_data.default_input).get_placeholder_as_batch_major()
  output_dict["target_classes"] = rnn.engine.network.get_extern_data("targetb").get_placeholder_as_batch_major()

  # get attention weight axis information
  weight_layer = rnn.engine.network.get_layer("output/att_weights").output.copy_as_batch_major()
  weight_batch_axis = weight_layer.batch_dim_axis
  weight_time_axis = weight_layer.time_dim_axis
  weight_head_axis = weight_layer.get_axis_from_description("stag:att_heads")
  weight_att_axis = weight_layer.get_axis_from_description("stag:att_t")

  energy_layer = rnn.engine.network.get_layer("output/att_energy").output.copy_as_batch_major()
  energy_batch_axis = energy_layer.batch_dim_axis
  energy_time_axis = energy_layer.time_dim_axis
  energy_head_axis = energy_layer.get_axis_from_description("stag:att_heads")
  energy_att_axis = energy_layer.get_axis_from_description("stag:att_t")

  if rnn.config.typed_value("att_area") == "seg":
    output_dict["energy_len"] = rnn.engine.network.get_layer("output/att_energy").output.dim_tags[energy_att_axis].dyn_size

  # get losses
  loss_dict = rnn.engine.network.losses_dict
  total_err = loss_dict["output/output_prob"].loss.get_error()
  non_blank_err = loss_dict["output_loss_non_blank"].loss.get_error()
  blank_err = loss_dict["output_loss_blank"].loss.get_error()
  total_loss = loss_dict["output/output_prob"].loss.get_value()
  non_blank_loss = loss_dict["output_loss_non_blank"].loss.get_value()
  blank_loss = loss_dict["output_loss_blank"].loss.get_value()
  output_dict["total_err"] = total_err
  output_dict["non_blank_err"] = non_blank_err
  output_dict["blank_err"] = blank_err
  output_dict["total_loss"] = total_loss
  output_dict["non_blank_loss"] = non_blank_loss
  output_dict["blank_loss"] = blank_loss
  output_dict["non_blank_len"] = rnn.engine.network.get_layer("output_loss_non_blank").output.get_sequence_lengths()
  output_dict["blank_len"] = rnn.engine.network.get_layer("output_loss_blank").output.get_sequence_lengths()

  # get readout weights
  params_dict = rnn.engine.network.get_param_values_dict(rnn.engine.tf_session)
  readout_weights = params_dict["output/readout_in"]["W"]
  emit_prob_weights = params_dict["output/emit_prob0"]["W"]
  lm_weights = params_dict["output/lm_masked"]["lstm0/rec/W"]
  s_weights = params_dict["output/s"]["W"]

  with open(os.path.join(options.out_dir, "layer_weights"), "w+") as f:
    for layer_name, layer_weights in zip(
      ["output/readout_in", "output/emit_prob0", "output/lm_masked", "output/s"],
      [readout_weights, emit_prob_weights, lm_weights, s_weights]):
      layer = rnn.engine.network.get_layer(layer_name)
      if layer_name == "output/lm_masked":
        if isinstance(layer, returnn.tf.layers.base.WrappedInternalLayer):
          layer = layer.base_layer.sub_layer.subnetwork.layers["lstm0"]
        else:
          layer = layer.sub_layer.subnetwork.layers["lstm0"]
      layer_inps = layer.sources

      start_dim = 0
      f.write(layer_name + "_weights:\n")
      for inp in layer_inps:
        feat_dims = inp.output.dim_tags[inp.output.feature_dim_axis].dimension
        mean_act = np.mean(layer_weights[start_dim:start_dim + feat_dims])
        start_dim += feat_dims
        f.write("\t" + inp.name + ": " + str(mean_act) + "\n")
      f.write("\n\n")

  readout_layer = rnn.engine.network.get_layer("output/readout_in")
  readout_inps = readout_layer.sources
  with open(os.path.join(options.out_dir, "readout-weights"), "w+") as f:
    start_dim = 0
    for inp in readout_inps:
      feat_dims = inp.output.dim_tags[inp.output.feature_dim_axis].dimension
      mean_act = np.mean(readout_weights[start_dim:start_dim+feat_dims])
      start_dim += feat_dims
      f.write(inp.name + ": " + str(mean_act) + "\n")

  # go through the specified sequences and dump the outputs from output_dict into separate files
  seq_idx = options.startseq
  if options.endseq < 0:
    options.endseq = float("inf")
  if not os.path.exists(options.out_dir):
    os.mkdir(options.out_dir)
  plot_dir = os.path.join(options.out_dir, "att_plots")
  if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

  att_means_sum = 0
  am_means_sum = 0
  prev_out_embed_means_sum = 0
  # prev_out_non_blank_means_sum = 0
  s_means_sum = 0
  lm_means_sum = 0
  readout_means_sum = 0
  num_seqs = 0
  total_loss = 0
  total_err = 0
  total_len = 0
  non_blank_loss = 0
  non_blank_len = 0
  non_blank_err = 0
  blank_loss = 0
  blank_len = 0
  blank_err = 0
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= options.endseq:
    out = rnn.engine.run_single(dataset=dataset, seq_idx=seq_idx, output_dict=output_dict)

    num_seqs += 1
    att_means_sum += np.mean(out["att-out"])
    am_means_sum += np.mean(out["am-out"])
    lm_means_sum += np.mean(out["lm-out"])
    s_means_sum += np.mean(out["s-out"])
    readout_means_sum += np.mean(out["readout-out"])
    prev_out_embed_means_sum += np.mean(out["prev_out_embed-out"])
    # prev_out_non_blank_means_sum += np.mean(out["lm_embed-out"])


    total_loss += out["total_loss"]
    total_err += out["total_err"]
    total_len += out["encoder_len"][0]
    non_blank_loss += out["non_blank_loss"]
    non_blank_err += out["non_blank_err"]
    non_blank_len += out["non_blank_len"][0]
    blank_loss += out["blank_loss"]
    blank_err += out["blank_err"]
    blank_len += out["blank_len"][0]

    if seq_idx in options.plot_seqs:
      assert options.plot_weights or options.plot_energies
      seq_tag = out["seq_tag"]
      encoder_len = out["encoder_len"][0]

      hmm_align = subprocess.check_output(
        ["/u/schmitt/experiments/transducer/config/sprint-executables/archiver", "--mode", "show", "--type", "align",
          "--allophone-file", "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/allophones",
          "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/tuske__2016_01_28__align.combined.train",
          seq_tag[0]])
      hmm_align = hmm_align.splitlines()
      hmm_align = [row.decode("utf-8").strip() for row in hmm_align]
      hmm_align = [row for row in hmm_align if row.startswith("time")]
      hmm_align = [row.split("\t")[5].split("{")[0] for row in hmm_align]
      hmm_align = [phon if phon != "[SILENCE]" else "[S]" for phon in hmm_align]

      # hmm_align = list(np.random.randint(0, 10, (6 * encoder_len - 3,)))

      hmm_align_borders = [i+1 for i, x in enumerate(hmm_align) if i < len(hmm_align)-1 and hmm_align[i] != hmm_align[i+1]]
      if hmm_align_borders[-1] != len(hmm_align):
        hmm_align_borders += [len(hmm_align)]
      hmm_align_major = [hmm_align[i-1] for i in range(1, len(hmm_align) + 1) if i in hmm_align_borders]
      if hmm_align_borders[0] != 0:
        hmm_align_borders = [0] + hmm_align_borders
      # print(hmm_align)
      # print(len(hmm_align))
      # print(hmm_align_major)
      # print(hmm_align_borders)
      # hmm_align_major = [hmm_align[i] for i in range(len(hmm_align)) if i in hmm_align_borders]
      hmm_align_borders_center = [(i+j) / 2 for i, j in zip(hmm_align_borders[:-1], hmm_align_borders[1:])]

      last_label_rep = len(hmm_align) - (encoder_len-1) * 6
      target_align = out["target_classes"][0][:]

      targetb_blank_idx = rnn.config.typed_value("targetb_blank_idx")

      weights = np.transpose(
        out["att_weights-out"],
        (weight_batch_axis, weight_time_axis, weight_att_axis, weight_head_axis))

      print("WEIGHTS SHAPE: ", weights.shape)

      energies = np.transpose(
        out["att_energy-out"],
        (energy_batch_axis, energy_time_axis, energy_att_axis, energy_head_axis))

      fig, axs = plt.subplots(
        2 if options.plot_weights and options.plot_energies else 1,
        weights.shape[3],
        figsize=(weights.shape[3] * 30, 35 if options.plot_weights and options.plot_energies else 5),
        constrained_layout=True)
      if not (options.plot_weights and options.plot_energies):
        axs = [axs]

      for head in range(weights.shape[3]):
        weights_ = weights[0, :, :, head]  # (B, dec, enc, heads) -> (dec, enc)
        energies_ = energies[0, :, :, head]
        if rnn.config.typed_value("att_area") == "win" and rnn.config.typed_value("att_win_size") == "full":
          pass

        elif rnn.config.typed_value("att_area") == "win":
          excess_size = int(weights_.shape[1] / 2)
          zeros = np.zeros(
            (weights_.shape[0], (encoder_len - weights_.shape[1]) + excess_size * 2))  # (dec, (enc - win) + 2*(win//2))
          ones = np.ones_like(zeros)
          # append zeros to the weights to fill up the whole encoder length
          weights_ = np.concatenate([weights_, zeros], axis=1)  # (dec, enc + 2*(win//2))
          energies_ = np.concatenate([energies_, ones * np.min(energies_)], axis=1)
          # roll the weights to their corresponding position in the encoder sequence (sliding window)
          weights_ = np.array([np.roll(row, i) for i, row in enumerate(weights_)])[:, excess_size:-excess_size]
          energies_ = np.array([np.roll(row, i) for i, row in enumerate(energies_)])[:, excess_size:-excess_size]
        else:
          assert rnn.config.typed_value("att_area") == "seg"
          energy_len = out["energy_len"]
          segment_starts = out["segment_starts-out"][0]
          print("WEIGHTS SHAPE: ", segment_starts.shape)
          energies_ = np.where(np.arange(0, energies_.shape[1])[None, :] >= energy_len, np.min(energies_), energies_)
          zeros = np.zeros((weights_.shape[0], encoder_len - weights_.shape[1]))  # (dec, enc - win)
          ones = np.ones_like(zeros)
          weights_ = np.concatenate([weights_, zeros], axis=1)  # (dec, enc)
          energies_ = np.concatenate([energies_, ones * np.min(energies_)], axis=1)
          weights_ = np.array([np.roll(row, start) for start, row in zip(segment_starts, weights_)])
          energies_ = np.array([np.roll(row, start) for start, row in zip(segment_starts, energies_)])

        target_align_borders_expanded = [0] + [
          (i + 1) * 6 - 1 if i != len(target_align) - 1 else (i + 1) * 6 - 1 - (6 - last_label_rep) for i, x in
          enumerate(target_align) if x != targetb_blank_idx]
        target_align_borders_expanded_center = [(i + j) / 2 for i, j in
          zip(target_align_borders_expanded[:-1], target_align_borders_expanded[1:])]
        target_label_idxs = [i for i, x in enumerate(target_align) if x != targetb_blank_idx]
        label_ticks_major = [i+1 for i in range(len(target_label_idxs))]
        label_ticks_center = [(i + j) / 2 for i, j in
                              zip(([0] + label_ticks_major)[:-1], ([0] + label_ticks_major)[1:])]
        target_align_major = [x for x in target_align if x != targetb_blank_idx]

        weights_ = np.concatenate(
          [np.repeat(weights_[:, :-1], 6, axis=-1), np.repeat(weights_[:, -1:], last_label_rep, axis=-1)], axis=-1)
        weights_ = weights_[target_label_idxs]

        energies_ = np.concatenate(
          [np.repeat(energies_[:, :-1], 6, axis=-1), np.repeat(energies_[:, -1:], last_label_rep, axis=-1)], axis=-1)
        energies_ = energies_[target_label_idxs]

        assert os.path.exists(options.vocab_file)
        with open(options.vocab_file) as f:
          vocab = f.read()
        vocab = ast.literal_eval(vocab)
        vocab = {v: k for k, v in vocab.items()}
        target_align_major = [vocab[c] for c in target_align_major]

        matrices = [
          matrix for matrix, flag in
          zip([weights_, energies_], [options.plot_weights, options.plot_energies]) if flag is True]

        titles = [
          title for title, flag in zip(["weights", "energies"], [options.plot_weights, options.plot_energies])
          if flag is True]

        bpe_yticks = [x - .5 for x in label_ticks_major]
        bpe_yticks_minor = [x - .5 for x in label_ticks_center]
        bpe_xticks = [x - .5 if x == 0 else x + .5 for x in target_align_borders_expanded]
        bpe_xticks_minor = [x if i == 0 else x + .5 for i, x in enumerate(target_align_borders_expanded_center)]

        hmm_xticks = [x - .5 for x in hmm_align_borders]
        hmm_xticks_minor = [x - .5 for x in hmm_align_borders_center]

        time_xticks = [x - .5 for x in range(0, len(hmm_align), 10)]
        time_xticks_labels = [x for x in range(0, len(hmm_align), 10)]

        for ax, matrix, title in zip(axs[:, head] if weights.shape[3] > 1 else axs, matrices, titles):
          # plot matrix
          matshow = ax.matshow(matrix, aspect="auto", cmap=plt.cm.get_cmap("Blues"))
          # create second x axis for hmm alignment labels and plot same matrix
          hmm_ax = ax.twiny()
          time_ax = ax.twiny()
          # set x and y axis for target alignment axis
          if not options.presentation_mode:
            ax.set_title(title + "_head" + str(head), y=1.1)

          ax.set_yticks(bpe_yticks)
          ax.yaxis.set_major_formatter(ticker.NullFormatter())
          ax.set_yticks(bpe_yticks_minor, minor=True)
          ax.set_yticklabels(target_align_major, fontsize=17, minor=True)
          ax.tick_params(axis="y", which="minor", length=0)
          ax.tick_params(axis="y", which="major", length=10)

          ax.set_xticks(bpe_xticks)
          ax.xaxis.set_major_formatter(ticker.NullFormatter())
          ax.set_xticks(bpe_xticks_minor, minor=True)
          ax.set_xticklabels(target_align_major, minor=True)
          ax.tick_params(axis="x", which="minor", length=0, labelsize=17)
          ax.tick_params(axis="x", which="major", length=10)
          ax.set_xlabel("RNA BPE Alignment", fontsize=18)
          ax.set_ylabel("Output RNA BPE Labels", fontsize=18)
          ax.xaxis.tick_top()
          ax.xaxis.set_label_position('top')
          ax.spines['top'].set_position(('outward', 50))
          # for tick in ax.xaxis.get_minor_ticks():
          #   tick.label1.set_horizontalalignment("center")

          # set x ticks and labels and positions for hmm axis
          hmm_ax.set_xticks(hmm_xticks)
          hmm_ax.xaxis.set_major_formatter(ticker.NullFormatter())
          hmm_ax.set_xticks(hmm_xticks_minor, minor=True)
          hmm_ax.set_xticklabels(hmm_align_major, fontsize=17, minor=True, rotation="vertical")
          hmm_ax.tick_params(axis="x", which="minor", length=0)
          hmm_ax.tick_params(axis="x", which="major", length=0)
          hmm_ax.xaxis.set_ticks_position('top')
          hmm_ax.xaxis.set_label_position('top')
          hmm_ax.set_xlabel("HMM Phoneme Alignment", fontsize=18)

          time_ax.set_xlabel("Input Time Frames", fontsize=18)
          time_ax.xaxis.tick_bottom()
          time_ax.xaxis.set_label_position('bottom')
          time_ax.set_xlim(ax.get_xlim())
          time_ax.set_xticks(time_xticks)
          time_ax.set_xticklabels(time_xticks_labels, fontsize=17)

          for idx in hmm_align_borders:
            ax.axvline(x=idx - .5, ymin=0, ymax=1, color="k", linestyle="--", linewidth=.5)
          for idx in label_ticks_major:
            ax.axhline(y=idx - .5, xmin=0, xmax=1, color="k", linewidth=.5)

          cbar = plt.colorbar(matshow, ax=ax)
          cbar.ax.tick_params(labelsize=17)

        if options.store_plot_data:
          plot_data = {
            "bpe_yticks": bpe_yticks,
            "bpe_yticks_minor": bpe_yticks_minor,
            "bpe_labels": target_align_major,
            "bpe_xticks": bpe_xticks,
            "bpe_xticks_minor": bpe_xticks_minor,
            "hmm_xticks": hmm_xticks,
            "hmm_xticks_minor": hmm_xticks_minor,
            "hmm_labels": hmm_align_major,
            "time_xticks": time_xticks,
            "time_labels": time_xticks_labels,
            "weights": weights_
          }

          plot_data_dir = os.path.join(options.out_dir, "att_plots_data")
          if not os.path.exists(plot_data_dir):
            os.mkdir(plot_data_dir)

          with open(os.path.join(plot_data_dir, "seq-" + str(seq_idx)), "wb+") as f:
            pickle.dump(plot_data, f)

      if not options.presentation_mode:
        suptitle = options.model_name + "\n" + seq_tag[0]
        fig.suptitle(suptitle)
      # fig.tight_layout()
      # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)
      plt.savefig(os.path.join(plot_dir, options.model_name + "_seq-" + str(seq_idx) + ".pdf"))
      plt.close()

    seq_idx += 1

    if seq_idx % 20 == 0:
      print("PROGRESS: %d/%d" % (seq_idx, dataset.estimated_num_seqs if options.endseq == float("inf") else options.endseq))

  with open(os.path.join(options.out_dir, "activations"), "w+") as f:
    for name, mean_sum in zip(
      ["att", "am", "s", "lm", "readout", "prev_out_embed"],
      [att_means_sum, am_means_sum, s_means_sum, lm_means_sum, readout_means_sum, prev_out_embed_means_sum]):
      f.write(name + ": " + str(mean_sum / num_seqs) + "\n")

  with open(os.path.join(options.out_dir, "errors_losses"), "w+") as f:
    f.write("Total Loss: " + str(total_loss))
    f.write("\nTotal Error: " + str(total_err))
    f.write("\nTotal Len: " + str(total_len))
    f.write("\nTotal Mean Loss: " + str(total_loss / total_len))
    f.write("\nTotal Mean Error: " + str(total_err / total_len))

    f.write("\n\nBlank Loss: " + str(blank_loss))
    f.write("\nBlank Error: " + str(blank_err))
    f.write("\nBlank Len: " + str(blank_len))
    f.write("\nBlank Mean Loss: " + str(blank_loss / blank_len))
    f.write("\nBlank Mean Error: " + str(blank_err / blank_len))

    f.write("\n\nNon Blank Loss: " + str(non_blank_loss))
    f.write("\nNon Blank Error: " + str(non_blank_err))
    f.write("\nNon Blank Len: " + str(non_blank_len))
    f.write("\nNon Blank Mean Loss: " + str(non_blank_loss / blank_len))
    f.write("\nNon Blank Mean Error: " + str(non_blank_err / blank_len))

  print("Done. More seqs which we did not dumped: %s" % dataset.is_less_than_num_seqs(seq_idx), file=returnn.log.log.v1)


def net_dict_add_losses(net_dict):
  net_dict["output"]["unit"]["lm_masked"]["is_output_layer"] = True
  net_dict["output"]["unit"]["lm_masked"]["unit"]["subnetwork"]["input_embed"]["is_output_layer"] = True
  net_dict["output"]["unit"]["lm_masked"]["unit"]["is_output_layer"] = True
  net_dict["output"]["unit"]["output_is_blank"] = {
          "class": "compare", "from": "output_", "value": rnn.config.typed_value("targetb_blank_idx"),
          "kind": "equal",
          "initial_output": False, "is_output_layer": True}
  net_dict.update({
    "_target_masked_blank": {
      "class": "masked_computation", "mask": "output/output_is_blank",
      "from": "data:targetb", "unit": {"class": "copy"}},
    "5_target_masked": {
      "class": "copy", "from": "_target_masked_blank", "register_as_extern_data": "targetb_masked_blank"},
    "4_target_masked": {
      "class": "copy", "from": "_target_masked",
      "register_as_extern_data": "targetb_masked_1031"},
    "output_prob_non_blank": {
      "class": "masked_computation", "mask": "output/output_emit", "from": "output/output_prob",
      "unit": {"class": "copy", "from": ["data"]}},
    "output_loss_non_blank": {
      "class": "copy", "from": "output_prob_non_blank", "target": "targetb_masked_1031", "loss": "ce",
      "loss_opts": {"focal_loss_factor": 2.0, "scale": 1.0}},
    "output_prob_blank": {
      "class": "masked_computation", "mask": "output/output_is_blank", "from": "output/output_prob",
      "unit": {"class": "copy", "from": ["data"]}},
    "output_loss_blank": {
      "class": "copy", "from": "output_prob_blank", "target": "targetb_masked_blank", "loss": "ce",
      "loss_opts": {"focal_loss_factor": 2.0, "scale": 1.0}}
  })
    # "slow_readout_in": {
    #   "class": "linear", "from": ["slow_rnn"], "activation": None, "n_out": 1000, "reuse_params": "output/readout_in"},
    # "slow_readout": {"class": "reduce_out", "mode": "max", "num_pieces": 2, "from": ["slow_readout_in"]},
    # "slow_log_prob": {
    #   "class": "linear", "from": "slow_readout", "activation": "log_softmax", "dropout": 0.1,
    #   "n_out": target_num_labels}, "slow_prob": {
    #   "class": "activation", "from": "slow_log_prob", "activation": "exp",
    #   "target": "targetb_masked" if task == "train" else None, "loss": "ce" if task == "train" else None,
    #   "loss_opts": {"focal_loss_factor": 2.0, "label_smoothing": _label_smoothing, "scale": 4.0}}, })

  return net_dict


def init(config_filename, command_line_options):
  """
  :param str config_filename:
  :param list[str] command_line_options:
  """
  rnn.init(config_filename=config_filename, command_line_options=command_line_options, config_updates={"log": None},
    extra_greeting="RETURNN dump-forward starting up.")
  print("CONFIG: ", rnn.config)
  rnn.engine.init_train_from_config(config=rnn.config,
                                    train_data=rnn.train_data)  # rnn.engine.init_network_from_config(rnn.config)
  rnn.engine.init_network_from_config(net_dict_post_proc=net_dict_add_losses)



def main(argv):
  """
  Main entry.
  """
  arg_parser = argparse.ArgumentParser(description='Forward something and dump it.')
  arg_parser.add_argument('returnn_config')
  arg_parser.add_argument('--epoch', type=int, default=1)
  arg_parser.add_argument('--startseq', type=int, default=0, help='start seq idx (inclusive) (default: 0)')
  arg_parser.add_argument('--endseq', type=int, default=10, help='end seq idx (inclusive) or -1 (default: 10)')
  arg_parser.add_argument('--out_dir', type=str)
  arg_parser.add_argument('--model_name', type=str, default="model")
  arg_parser.add_argument('--vocab_file', type=str)
  arg_parser.add_argument('--plot_seqs', action="append", type=int)
  arg_parser.add_argument('--plot_weights', action="store_true")
  arg_parser.add_argument('--plot_energies', action="store_true")
  arg_parser.add_argument('--presentation_mode', action="store_true")
  arg_parser.add_argument('--store_plot_data', action="store_true")
  arg_parser.add_argument('--returnn_root', type=str)

  args = arg_parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  global returnn
  import returnn.__main__ as rnn
  import returnn

  init(config_filename=args.returnn_config, command_line_options=[])
  dump(rnn.train_data, args)
  rnn.finalize()


if __name__ == '__main__':
  main(sys.argv)
