import matplotlib.pyplot as plt
import h5py
import ast
import numpy as np
import os

"""
This script can be used to plot the attention weights and segment boundaries of multiple models for the same utterance.
"""


model1_folder = "/u/schmitt/experiments/segmental_models_2021_22/alias/models/ls_conformer/import_glob.conformer.mohammad.5.6/center_window_train_global_recog/win-size-129/w-weight-feedback/w-eos/10-epochs_const-lr-0.000100/center_window_att_train_recog/returnn_decoding/analysis/att_weights/dev-other/search/dump_hdf/output"
model2_folder = "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2021_22/i6_core/returnn/forward/ReturnnForwardJob.8VnoYBjNApTh/output"
model_folders = [model1_folder, model2_folder]

att_weight_hdfs = [
  model_folder + "/att_weights.hdf" for model_folder in model_folders
]
targets_hdfs = [
  model_folder + "/targets.hdf" for model_folder in model_folders
]
seg_starts_hdfs = [
  model_folder + "/seg_starts.hdf" for model_folder in model_folders
]
seg_lens_hdfs = [
  model_folder + "/seg_lens.hdf" for model_folder in model_folders
]
center_positions_hdfs = [
  model_folder + "/center_positions.hdf" for model_folder in model_folders
]
ref_alignment_hdf = "/u/schmitt/experiments/segmental_models_2021_22/alias/models/ls_conformer/import_glob.conformer.mohammad.5.6/center-window_att_global_ctc_align/att_weight_penalty/penalty_in_train/win-size-129_10-epochs_0.000100-const-lr/mult-weight-0.005000_exp-weight-2.000000/loss-scale-1.000000/w_penalty/returnn_decoding/analysis/att_weights/dev-other/search/plot/input/i6_core_returnn_forward_ReturnnForwardJob.1fohfY7LLczN/output/alignments.hdf"
json_vocab_path = "/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
target_blank_idx = 10025
ref_alignment_blank_idx = 10025


def plot():
  out_folder = os.path.join(os.path.dirname(__file__), "plots")
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)

  weights_dicts = []
  targets_dicts = []
  seg_starts_dicts = []
  seg_lens_dicts = []
  center_positions_dicts = []

  # first, we load all the necessary data (att_weights, targets, segment starts/lens (if provided)) from the hdf files
  for att_weight_hdf in att_weight_hdfs:
    with h5py.File(att_weight_hdf, "r") as f:
      # load tags once here and reuse for remaining hdfs
      seq_tags = f["seqTags"][()]
      seq_tags = [str(tag) for tag in seq_tags]  # convert from byte to string

      seq_lens = f["seqLengths"][()][:, 0]  # total num frames (num_seqs * num_labels * num_frames)
      shape_data = f["targets"]["data"]["sizes"][
        ()]  # shape of att weights for each seq, stored alternating (num_labels, num_frames, ..., num_labels, num_frames)
      weights_data = f["inputs"][()]  # all att weights as 1D array
      weights_dict = {}
      for i, seq_len in enumerate(seq_lens):
        weights_dict[seq_tags[i]] = weights_data[:seq_len]  # store weights in dict under corresponding tag
        weights_data = weights_data[seq_len:]  # cut the stored weights from the remaining weights
        shape = shape_data[i * 2: i * 2 + 2]  # get shape for current weight matrix
        weights_dict[seq_tags[i]] = weights_dict[seq_tags[i]].reshape(shape)  # reshape corresponding entry in dict
      weights_dicts.append(weights_dict)

  # follows same principle as the loading of att weights
  for targets_hdf in targets_hdfs:
    with h5py.File(targets_hdf, "r") as f:
      targets_data = f["inputs"][()]
      seq_lens = f["seqLengths"][()][:, 0]
      targets_dict = {}
      for i, seq_len in enumerate(seq_lens):
        targets_dict[seq_tags[i]] = targets_data[:seq_len]
        targets_data = targets_data[seq_len:]
      targets_dicts.append(targets_dict)

  if target_blank_idx is not None:
    for seg_starts_hdf in seg_starts_hdfs:
      # follows same principle as the loading of att weights
      with h5py.File(seg_starts_hdf, "r") as f:
        seg_starts_data = f["inputs"][()]
        seq_lens = f["seqLengths"][()][:, 0]
        seg_starts_dict = {}
        for i, seq_len in enumerate(seq_lens):
          seg_starts_dict[seq_tags[i]] = seg_starts_data[:seq_len]
          seg_starts_data = seg_starts_data[seq_len:]
        seg_starts_dicts.append(seg_starts_dict)

    for seg_lens_hdf in seg_lens_hdfs:
      # follows same principle as the loading of att weights
      with h5py.File(seg_lens_hdf, "r") as f:
        seg_lens_data = f["inputs"][()]
        seq_lens = f["seqLengths"][()][:, 0]
        seg_lens_dict = {}
        for i, seq_len in enumerate(seq_lens):
          seg_lens_dict[seq_tags[i]] = seg_lens_data[:seq_len]
          seg_lens_data = seg_lens_data[seq_len:]
        seg_lens_dicts.append(seg_lens_dict)

  if center_positions_hdfs is not None:
    for center_positions_hdf in center_positions_hdfs:
      # follows same principle as the loading of att weights
      with h5py.File(center_positions_hdf, "r") as f:
        center_positions_data = f["inputs"][()]
        seq_lens = f["seqLengths"][()][:, 0]
        center_positions_dict = {}
        for i, seq_len in enumerate(seq_lens):
          center_positions_dict[seq_tags[i]] = center_positions_data[:seq_len]
          center_positions_data = center_positions_data[seq_len:]
        center_positions_dicts.append(center_positions_dict)

  # follows same principle as the loading of att weights
  with h5py.File(ref_alignment_hdf, "r") as f:
    # here we need to load the seq tags again because the ref alignment might contain more seqs than what we dumped
    # into the hdfs
    seq_tags_ref_alignment = f["seqTags"][()]
    seq_tags_ref_alignment = [str(tag) for tag in seq_tags_ref_alignment]  # convert from byte to string

    ref_alignment_data = f["inputs"][()]
    seq_lens = f["seqLengths"][()][:, 0]
    ref_alignment_dict = {}
    for i, seq_len in enumerate(seq_lens):
      # only store the ref alignment if the corresponding seq tag is among the ones for which we dumped att weights
      if seq_tags_ref_alignment[i] in seq_tags:
        ref_alignment_dict[seq_tags_ref_alignment[i]] = ref_alignment_data[:seq_len]
      ref_alignment_data = ref_alignment_data[seq_len:]

  # load vocabulary as dictionary
  with open(json_vocab_path, "r") as f:
    json_data = f.read()
    vocab = ast.literal_eval(json_data)
    vocab = {v: k for k, v in vocab.items()}

  # for each seq tag, plot the corresponding att weights
  for seq_tag in seq_tags:
    if seq_tag != "b'dev-other/7697-105815-0015/7697-105815-0015'":
      continue
    seg_starts = seg_starts_dicts[0][seq_tag] if target_blank_idx is not None else None
    seg_lens = seg_lens_dicts[0][seq_tag] if target_blank_idx is not None else None
    center_positions0 = center_positions_dicts[0][seq_tag] if center_positions_hdfs is not None else None
    center_positions1 = center_positions_dicts[1][seq_tag] if center_positions_hdfs is not None else None
    ref_alignment = ref_alignment_dict[seq_tag]
    targets = targets_dicts[0][seq_tag]
    weights0 = weights_dicts[0][seq_tag]
    weights1 = weights_dicts[1][seq_tag]

    num_labels = weights0.shape[0]
    num_frames = weights0.shape[1]
    fig_width = num_frames / 8
    fig_height = num_labels / 4
    figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.matshow(weights0, cmap=plt.cm.get_cmap("Blues"), aspect="auto")

    ref_label_positions = np.where(ref_alignment != ref_alignment_blank_idx)[
                            -1] + 1  # +1 bc plt starts at 1, not at 0
    ref_labels = ref_alignment[ref_alignment != ref_alignment_blank_idx]
    ref_labels = [vocab[idx] for idx in ref_labels]  # the corresponding bpe label
    labels = targets[
      targets != target_blank_idx] if target_blank_idx is not None else targets  # the alignment labels which are not blank (in case of global att model, just use `targets`)
    labels = [vocab[idx] for idx in labels]  # the corresponding bpe label
    # x axis
    ax.set_xticks([tick - 1.0 for tick in ref_label_positions])
    ax.set_xticklabels(ref_labels, rotation=90)
    # y axis
    yticks = [tick for tick in range(num_labels)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    for ytick in yticks:
      ax.axhline(y=ytick - .5, xmin=0, xmax=1, color="k", linewidth=.5)

    if target_blank_idx is not None:
      # red delimiters to indicate segment boundaries
      for i, (seg_start, seg_len) in enumerate(zip(seg_starts, seg_lens)):
        ymin = (num_labels - i) / num_labels
        ymax = (num_labels - i - 1) / num_labels
        ax.axvline(x=seg_start - 0.5, ymin=ymin, ymax=ymax, color="r")
        ax.axvline(x=seg_start + seg_len - 1.5, ymin=ymin, ymax=ymax, color="r")

    if center_positions0 is not None:
      # green markers to indicate center positions
      for i, center_position0 in enumerate(center_positions0):
        ymin = (num_labels - i) / num_labels
        ymax = (num_labels - i - 1) / num_labels
        ax.axvline(x=center_position0 - .5, ymin=ymin, ymax=ymax, color="lime")
        ax.axvline(x=center_position0 + .5, ymin=ymin, ymax=ymax, color="lime")

        if i < len(center_positions1):
          center_position1 = center_positions1[i]
          if center_position1 != center_position0:
            ax.axvline(x=center_position1 - .5, ymin=ymin, ymax=ymax, color="red", alpha=0.3)
            ax.axvline(x=center_position1 + .5, ymin=ymin, ymax=ymax, color="red", alpha=0.3)

    filename = "plot.%s" % seq_tag.replace("/", "_")
    plt.savefig(out_folder + "/" + filename + ".png")
    plt.savefig(out_folder + "/" + filename + ".pdf")


if __name__ == "__main__":
  plot()
