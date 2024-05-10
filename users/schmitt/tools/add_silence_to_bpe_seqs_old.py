import argparse
import ast
import sys
import os
import numpy as np
import tensorflow as tf
import time
import h5py
import subprocess


def get_hdf_out_dataset(out_dim):
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_hdf.hdf", dim=out_dim, ndim=1)


def get_allophone_word_ends(archiver_tool, allophone_file, phoneme_align_cache, seq_tag):
  """
  Get the indices of allophones which end with @f (word ends) from the phoneme alignment cache.
  This includes silence tokens ([SILENCE]{#+#}@i@f).
  :param archiver_tool:
  :param allophone_file:
  :param phoneme_align_cache:
  :param seq_tag:
  :return:
  """
  # archiver shell command
  command = [
    archiver_tool,
    "--mode", "show",
    "--type", "align",
    "--allophone-file", allophone_file,
    phoneme_align_cache,
    seq_tag
  ]
  result = subprocess.run(command, capture_output=True, text=True)
  result_lines = result.stdout.split("\n")
  # parse archiver output
  allophone_idxs = []
  allophones = []
  for line in result_lines:
    if line.startswith("<") or line == "":
      continue
    # e.g.: time= 565     emission=       1524    allophone=      AH{K+T} index=  1524    state=  0
    _, _, _, _, _, allophone, _, allophone_idx, _, _ = line.split("\t")
    allophone_idxs.append(int(allophone_idx))
    allophones.append(allophone)
  # remove repetitions (e.g. collapse A{...}A{...}A{...} to A{...})
  allophone_idxs = np.array(allophone_idxs)
  allophone_reps = np.append(allophone_idxs[1:] == allophone_idxs[:-1], False)
  allophone_idxs_no_reps = allophone_idxs[~allophone_reps]
  allophones = np.array(allophones)
  allophones_no_reps = allophones[~allophone_reps]
  # get word ends (allophones which end with @f, e.g. T{AH+HH}@f)
  is_word_end = np.char.endswith(allophones_no_reps, "@f")
  word_ends = allophone_idxs_no_reps[is_word_end]
  return word_ends


def get_allophone_word_ends2(allophone_idxs, state_tying_vocab, silence_idx):
  """
  Get the indices of allophones which end with @f (word ends) from the phoneme alignment cache.
  This includes silence tokens ([SILENCE]{#+#}@i@f).
  :param archiver_tool:
  :param allophone_file:
  :param phoneme_align_cache:
  :param seq_tag:
  :return:
  """
  # remove repetitions (e.g. collapse A{...}A{...}A{...} to A{...})
  allophone_idxs = np.array(allophone_idxs)
  allophone_idxs = allophone_idxs[allophone_idxs != 0]
  allophone_reps = np.append(allophone_idxs[1:] == allophone_idxs[:-1], False)
  allophone_idxs = allophone_idxs[~allophone_reps]
  final_states = np.array([int(state_tying_vocab[idx][-1]) for idx in allophone_idxs])
  is_silence = allophone_idxs == silence_idx
  is_word_end = np.append(final_states[:-1] > final_states[1:], [True])
  is_word_end = np.logical_or(is_word_end, is_silence)
  word_ends = allophone_idxs[is_word_end]
  return word_ends


def add_silence_to_labels(
        hdf_data,
        seq_tags,
        seq_lens,
        out_dataset,
        phoneme_silence_idx,
        bpe_vocab,
        bpe_silence_idx,
        sprint_cache_dataset,
        state_tying_vocab,
        archiver_tool,
        allophone_file,
        phoneme_align_cache
):
  seq_idx = 0

  sprint_cache_dataset.init_seq_order(1)
  print("SILENCE IDX: ", phoneme_silence_idx)

  while sprint_cache_dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    sprint_cache_dataset.load_seqs(seq_idx, seq_idx + 1)
    tag = sprint_cache_dataset.get_tag(seq_idx)
    allophone_idxs = sprint_cache_dataset.get_data(seq_idx, "data")
    allophones = [state_tying_vocab[idx] for idx in allophone_idxs]
    word_ends = get_allophone_word_ends2(allophone_idxs, state_tying_vocab, phoneme_silence_idx)
    print([state_tying_vocab[idx] for idx in word_ends])
    print(len(word_ends))
    break

  for seq_tag, seq_len in zip(seq_tags, seq_lens):
    if seq_idx % 1000 == 0:
      print(f"Progress: {seq_idx / len(seq_tags) * 100:.2f}% ({seq_idx}/{len(seq_tags)})")
    bpe_seq = hdf_data[:seq_len]
    # cut off the data that was just read
    hdf_data = hdf_data[seq_len:]
    bpe_string_seq = np.array([bpe_vocab[idx] for idx in bpe_seq])
    bpe_word_end_positions = np.where(~np.char.endswith(bpe_string_seq, "@@"))[0]

    allophone_word_ends = get_allophone_word_ends(archiver_tool, allophone_file, phoneme_align_cache, seq_tag)
    allophone_word_ends_no_silence = allophone_word_ends[allophone_word_ends != 0]
    assert len(allophone_word_ends_no_silence) == len(bpe_word_end_positions), (
      "Number of word ends in BPE and phoneme alignment do not match")

    print(bpe_string_seq)
    print(len(allophone_word_ends))
    exit()

    # insert silence token into BPE sequence at word ends
    bpe_seq_w_silence = np.array(bpe_seq)
    word_end_idx = -1
    for i, idx in enumerate(allophone_word_ends):
      if idx == phoneme_silence_idx:
        # in case the phoneme alignment starts with silence, insert silence token at the beginning
        if i == 0:
          bpe_seq_w_silence = np.insert(bpe_seq_w_silence, 0, bpe_silence_idx)
        # insert silence behind the current word end in the BPE alignment
        else:
          bpe_seq_w_silence = np.insert(bpe_seq_w_silence, bpe_word_end_positions[word_end_idx] + 1, bpe_silence_idx)
        bpe_word_end_positions += 1
      else:
        # move to the next word end in the BPE alignment (each BPE word end corresponds to a phoneme word end)
        word_end_idx += 1

    # dump new alignment into hdf file
    new_data = tf.constant(np.expand_dims(bpe_seq_w_silence, axis=0), dtype="int32")
    new_seq_len = len(bpe_seq_w_silence)
    extra = {}
    seq_lens = {0: tf.constant([new_seq_len]).numpy()}
    ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    out_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[seq_tag], extra=extra)

    seq_idx += 1


def init_returnn():
  from returnn.log import log
  from returnn.util.debug import init_better_exchook, init_faulthandler
  from returnn.util.basic import init_thread_join_hack

  init_better_exchook()
  init_thread_join_hack()
  rnn_main.init_config(config_filename=None, default_config={"cache_size": 0})
  config = rnn_main.config
  config.set("log", None)
  rnn_main.init_log()
  print("Returnn add_silence_to_bpe_seqs starting up", file=rnn_main.log.v2)
  rnn_main.returnn_greeting()
  init_faulthandler()


def get_hdf_data(bpe_hdf):
  with h5py.File(bpe_hdf, "r") as f:
    hdf_data = f["inputs"][()]
    seq_tags = f["seqTags"][()]
    seq_lens = f["seqLengths"][()][:, 0]

  return hdf_data, seq_tags, seq_lens


def get_sprint_cache_dataset(alignment_cache, silence_phone, allophone_file, state_tying_file):
  from returnn import datasets
  dataset_dict = {
    "class": "SprintCacheDataset",
    "data": {
      "data": {
        "filename": alignment_cache,
        "data_type": "align",
        "allophone_labeling": {
          "silence_phone": silence_phone,
          "allophone_file": allophone_file,
          "state_tying_file": "state-tying",
        },
      }
    },
    # "seq_ordering": "sorted",
  }
  dataset = datasets.init_dataset(dataset_dict)
  return dataset


def get_bpe_vocab(bpe_vocab_path):
  with open(bpe_vocab_path, "r") as f:
    bpe_vocab = ast.literal_eval(f.read())
  bpe_vocab = {v: k for k, v in bpe_vocab.items()}
  return bpe_vocab


def get_state_tying_vocab(state_tying_path):
  # load state_tying
  state_tying = {}
  with open(state_tying_path, "r") as f:
    for line in f:
      state, idx = line.split()
      state_tying[state] = int(idx)

  return {v: k for k, v in state_tying.items()}


def get_state_tying_vocab2(allophone_file):
  # load state_tying
  state_tying = {}
  with open(allophone_file, "r") as f:
    non_final_idx = 0
    i = 1
    for line in f:
      if line.startswith("#"):
        continue
      if line[-2] != "f":
        for j in range(3):
          state_tying[f"{line.strip()}.{j}"] = non_final_idx
      else:
        for j in range(3):
          state_tying[f"{line.strip()}.{j}"] = i
          i += 1
  with open("state-tying", "w+") as f:
    for k, v in state_tying.items():
      f.write(f"{k} {v}\n")

  silence_idx = state_tying["[SILENCE]{#+#}@i@f.0"]
  return {v: k for k, v in state_tying.items()}, silence_idx


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("--bpe_seqs_hdf", help="hdf file which contains BPE sequences")
  arg_parser.add_argument("--bpe_vocab", help="path to BPE vocab")
  arg_parser.add_argument("--phoneme_align_cache", help="RASR alignment cache with phoneme alignments")
  arg_parser.add_argument("--allophone_file", help="RASR allophone file")
  arg_parser.add_argument("--state_tying_file", help="RASR state-tying file")
  arg_parser.add_argument("--silence_phone", help="Phoneme representing silence, e.g. [SILENCE]")
  arg_parser.add_argument("--archiver_tool", help="RASR archiver tool")
  arg_parser.add_argument("--phoneme_silence_idx", help="the silence index in the phoneme alignment", type=int)
  arg_parser.add_argument("--segment_file", help="segment whitelist", type=str, default=None)
  arg_parser.add_argument("--returnn_root", type=str)
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn_main
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  init_returnn()
  hdf_data, seq_tags, seq_lens = get_hdf_data(args.bpe_seqs_hdf)
  bpe_vocab = get_bpe_vocab(args.bpe_vocab)
  bpe_silence_idx = max(bpe_vocab.keys()) + 1

  sprint_cache_dataset = get_sprint_cache_dataset(
    args.phoneme_align_cache,
    args.silence_phone,
    args.allophone_file,
    args.state_tying_file
  )
  state_tying_vocab = get_state_tying_vocab(args.state_tying_file)
  state_tying_vocab2, silence_idx = get_state_tying_vocab2(args.allophone_file)

  out_dataset = get_hdf_out_dataset(out_dim=bpe_silence_idx + 1)

  try:
    add_silence_to_labels(
      hdf_data,
      seq_tags,
      seq_lens,
      out_dataset,
      silence_idx,
      bpe_vocab,
      bpe_silence_idx,
      sprint_cache_dataset,
      state_tying_vocab2,
      args.archiver_tool,
      args.allophone_file,
      args.phoneme_align_cache
    )
    out_dataset.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn_main.finalize()


if __name__ == "__main__":
  main()
