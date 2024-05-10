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
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_align.hdf", dim=dim, ndim=1)


def dump_phoneme_align_into_hdf(sprint_dataset, hdf_dataset, dump_type):
  seq_idx = 0
  end_idx = float("inf")
  target_dim = sprint_dataset.get_data_dim("classes")
  blank_idx = target_dim
  while sprint_dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= end_idx:
    # try:
    sprint_dataset.load_seqs(seq_idx, seq_idx + 1)
    # except:
    #   print("ERROR WHILE LOADING SEQ")
    #   continue
    data = sprint_dataset.get_data(seq_idx, "classes")
    if seq_idx % 1000 == 0:
      print("Progress: Seq Idx: ", seq_idx)

    if dump_type == "remove_repetitions":
      data_blank_mask = np.append(data[1:] == data[:-1], False)
      data[data_blank_mask] = blank_idx
    else:
      assert dump_type == "as_is"

    seq_len = sprint_dataset.get_seq_length(seq_idx)["classes"]
    tag = sprint_dataset.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(data, axis=0), dtype="int32")

    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    hdf_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)
    seq_idx += 1


def init_returnn():
  from returnn.log import log
  from returnn.__main__ import init_better_exchook, init_thread_join_hack, init_faulthandler
  init_better_exchook()
  init_thread_join_hack()
  log.initialize(verbosity=[5])
  print("Returnn dump_phoneme_align starting up.", file=log.v3)
  init_faulthandler()


def get_extern_sprint_dataset(rasr_config_path, time_red_factor, rasr_exe, output_dim):
  sprint_args = [
    "--config=%s" % rasr_config_path,
    "--*.LOGFILE=nn-trainer.train.log",
    f"--*.trainer-output-dimension={output_dim}",
  ]
  dataset_dict = {
    'class': 'ExternSprintDataset',
    "reduce_target_factor": time_red_factor,
    'sprintConfigStr': sprint_args,
    'sprintTrainerExecPath': rasr_exe,
    "partition_epoch": 1,
    "suppress_load_seqs_print": True,
  }
  dataset = rnn.datasets.init_dataset(dataset_dict)

  return dataset


def get_state_tying_as_dict(state_tying_path):
  # load state_tying
  state_tying = {}
  with open(state_tying_path, "r") as f:
    for line in f:
      state, idx = line.split()
      state_tying[state] = int(idx)

  # write state_tying to file
  with open("phoneme_vocab", "w+") as f:
    f.write("{\n")
    for k, v in state_tying.items():
      f.write(f"\"{k}\": {v},\n")
    f.write("}")

  return {v: k for k, v in state_tying.items()}


def main(argv):
  """
  Main entry.
  """
  parser = argparse.ArgumentParser(description="Dump dataset or subset of dataset into external HDF dataset")
  parser.add_argument('rasr_config', type=str,
                      help="Config file for RETURNN, or directly the dataset init string")
  parser.add_argument('--rasr_exe', type=str)
  parser.add_argument('--state_tying', type=str)
  parser.add_argument('--dump_type', help="'as_is' or 'remove_repetitions'", type=str)
  parser.add_argument('--time_red_factor', type=int)
  parser.add_argument('--returnn_root', help="Returnn root to use for imports")

  args = parser.parse_args(argv[1:])

  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  init_returnn()
  state_tying = get_state_tying_as_dict(args.state_tying)
  max_idx = max(state_tying.keys())
  sprint_dataset = get_extern_sprint_dataset(
    rasr_exe=args.rasr_exe,
    rasr_config_path=args.rasr_config,
    time_red_factor=args.time_red_factor,
    output_dim=max_idx
  )
  hdf_dataset = hdf_dataset_init(dim=max_idx + 1)

  dump_phoneme_align_into_hdf(sprint_dataset, hdf_dataset, args.dump_type)
  hdf_dataset.close()

  rnn_main.finalize()


if __name__ == '__main__':
  main(sys.argv)
