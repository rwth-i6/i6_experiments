#!/usr/bin/env python3

"""
Creates a HDF file, which can be read by :class:`HDFDataset`.
The input is any other dataset (:class:`Dataset`).
"""

from __future__ import print_function

import json
import sys
import argparse
import numpy as np
import tensorflow as tf


def hdf_dataset_init(dim):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_align", dim=dim, ndim=1)


def hdf_dump_from_dataset(dataset, phon_to_id, id_to_phon, id_to_multiphone, id_to_state, hdf_dataset, parser_args):
  """
  :param Dataset dataset: could be any dataset implemented as child of Dataset
  :param hdf_dataset_mod.HDFDatasetWriter hdf_dataset:
  :param parser_args: argparse object from main()
  """
  seq_idx = 0
  end_idx = float("inf")
  target_dim = dataset.get_data_dim("classes")
  blank_idx = target_dim + 1  # idx target_dim is reserved for SOS token
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_idx:
    try:
      dataset.load_seqs(seq_idx, seq_idx + 1)
    except:
      print("ERROR WHILE LOADING SEQ")
      continue
    data = dataset.get_data(seq_idx, "classes")
    print("TAG: ", dataset.get_tag(seq_idx))
    # print("SEQ IDX: ", seq_idx)
    # print("DATA: ", data)

    ### first, do a sanity check if the alignment from RASR complies with our expectation
    ### we expect an alignment, where, for every phoneme except silence, there are three different idxs (three states)
    ### silence only has one state

    print("DATA: ", data)
    print("TXT DATA: ", np.array([id_to_multiphone[idx] for idx in data]))

    # init state with 0 and prev idx with first idx in data
    prev_idx = data[0]
    prev_phon = id_to_multiphone[prev_idx]
    prev_state = id_to_state[prev_idx]
    new_data = []
    # go through the alignment, starting from the second idx
    for idx in data[1:]:
      curr_state = id_to_state[idx]
      # if prev phoneme was silence and the current phoneme is not, then the prev frame was a silence seg boundary
      if prev_phon == "[SILENCE]" and prev_idx != idx:
        new_data.append(phon_to_id["[SILENCE]"])
      # if the prev state is greater than the current one, the prev frame was a segment boundary
      elif prev_state > curr_state:
        new_data.append(phon_to_id[prev_phon])
      # otherwise, the prev frame was not a segment boundary
      else:
        new_data.append(blank_idx)

      # update variables
      prev_state = curr_state
      prev_phon = id_to_multiphone[idx]
      prev_idx = idx

    new_data.append(phon_to_id[prev_phon])
    new_data = np.array(new_data)

    print("NEW DATA: ", new_data)
    print("TXT NEW DATA: ", np.array([id_to_phon[idx] if idx != blank_idx else blank_idx for idx in new_data]))
    # blank_mask = [True if i == j else False for i, j in zip(data[:-1], data[1:])] + [False]
    # # sil_mask = [True if id_to_multiphone[idx] == "[SILENCE]" else False for idx in data]
    # # sil_end_mask = [False if i == j or i is False else True for i, j in zip(sil_mask[:-1], sil_mask[1:])] + [sil_mask[-1]]
    # data[blank_mask] = blank_idx
    # blank_mask = [False if idx in end_state_ids or id_to_multiphone[idx] == "[SILENCE]" else True for idx in data]
    # data[blank_mask] = blank_idx
    # print("NEW DATA: ", np.array([id_to_multiphone[idx] if idx != blank_idx else blank_idx for idx in data]))
    # data = np.array([phon_to_id[id_to_multiphone[idx]] if idx != blank_idx else blank_idx for idx in data])
    # # data[sil_end_mask] = phon_to_id["[SILENCE]"]

    # print("NEW DATA: ", np.array([id_to_phon[idx] if idx != blank_idx else blank_idx for idx in data]))
    # print("NEW DATA: ", data)
    print("\n")

    seq_len = dataset.get_seq_length(seq_idx)["classes"]
    tag = dataset.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(new_data, axis=0), dtype="int32")

    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1 #- (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)
    seq_idx += 1


def hdf_close(hdf_dataset):
  """
  :param HDFDataset.HDFDatasetWriter hdf_dataset: to close
  """
  hdf_dataset.close()


def init(rasr_config_path, time_red, rasr_exe):
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
    "--*.LOGFILE=nn-trainer.train.log", "--*.TASK=1", "--*.corpus.segment-order-shuffle=true",
  ]

  dataset_dict = {
    'class': 'ExternSprintDataset',
    "reduce_target_factor": time_red,
    'sprintConfigStr': sprint_args,
    'sprintTrainerExecPath': '/u/schmitt/src/rasr-dev/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard',
    "partition_epoch": epoch_split}
  dataset = rnn.datasets.init_dataset(dataset_dict)
  print("Source dataset:", dataset.len_info(), file=log.v3)

  id_to_multiphone = {}
  end_state_ids = []
  mid_state_ids = []
  beg_state_ids = []
  id_to_state = {}
  phon_to_id = {}
  phon_vocab_w_ctx = {}
  print("BEFORE OPEN FILE")
  with open("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/tuske-phoneme-align/state-tying_mono-eow_3-states", "r") as state_tying_file:
    relevant_lines = []
    for i, line in enumerate(state_tying_file):
      if line.split("{")[1].startswith("#+#"):
        relevant_lines.append(line)

  for line in relevant_lines:
    allophone, id = line.split()
    id = int(id)
    monophone, ctx_state = allophone.split("{")
    ctx, state = ctx_state.split("}", maxsplit=1)

    if state.endswith(".2") and id not in end_state_ids:
      end_state_ids.append(id)
      id_to_state[id] = 2
    elif state.endswith(".1") and id not in mid_state_ids:
      mid_state_ids.append(id)
      id_to_state[id] = 1
    elif state.endswith(".0") and id not in beg_state_ids:
      beg_state_ids.append(id)
      id_to_state[id] = 0

    monophone_w_ctx = monophone + "{" + ctx + "}"
    if state.startswith("@f") and monophone not in ["[SILENCE]", "[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"]:
      monophone_w_ctx += "@f"
      monophone += "@f"
    monophone_w_ctx += ".0"

    if monophone not in phon_to_id:
      new_id = len(phon_to_id)
      phon_to_id[monophone] = new_id
      phon_vocab_w_ctx[monophone_w_ctx] = new_id

    if id not in id_to_multiphone:
      id_to_multiphone[id] = monophone

  id_to_phon = {v: k for k, v in phon_to_id.items()}
  id_to_phon_w_ctx = {v: k for k, v in phon_vocab_w_ctx.items()}
  phon = id_to_phon[0]
  phon_w_ctx = id_to_phon_w_ctx[0]
  if phon != "[SILENCE]":
    sil_id = phon_to_id["[SILENCE]"]
    phon_to_id["[SILENCE]"] = 0
    phon_vocab_w_ctx["[SILENCE]{#+#}.0"] = 0
    phon_to_id[phon] = sil_id
    phon_vocab_w_ctx[phon_w_ctx] = sil_id
    id_to_phon[sil_id] = phon
    id_to_phon[0] = "[SILENCE]"

  with open("phoneme_vocab", "w+") as f:
    json.dump(phon_vocab_w_ctx, f)

  print("VOCAB: ", phon_to_id)
  print("MULTI PHON VOCAB: ", id_to_multiphone)
  print("LEN MULTI PHON VOCAB: ", len(id_to_multiphone))
  print("END STATES: ", end_state_ids)
  print("MID STATES: ", mid_state_ids)
  print("BEG STATES: ", beg_state_ids)
  print("LEN END STATES: ", len(end_state_ids))
  print("VOCAB W CTX: ", phon_vocab_w_ctx)

  return dataset, phon_to_id, id_to_phon, id_to_multiphone, id_to_state


def main(argv):
  """
  Main entry.
  """
  parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset into external HDF dataset")
  parser.add_argument('rasr_config', type=str,
                      help="Config file for RETURNN, or directly the dataset init string")
  parser.add_argument('--rasr_exe', type=str)
  parser.add_argument('--time_red', type=int)
  parser.add_argument('--returnn_root', help="Returnn root to use for imports")

  args = parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  dataset, phon_to_id, id_to_phon, id_to_multiphone, id_to_state = init(
    rasr_config_path=args.rasr_config, time_red=args.time_red, rasr_exe=args.rasr_exe)
  hdf_dataset = hdf_dataset_init(dim=90)
  hdf_dump_from_dataset(dataset, phon_to_id, id_to_phon, id_to_multiphone, id_to_state, hdf_dataset, args)
  hdf_close(hdf_dataset)

  rnn_main.finalize()


if __name__ == '__main__':
  main(sys.argv)
