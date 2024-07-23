import argparse
import ast
import json
import sys
import numpy as np
import os
import tensorflow as tf
import time
import h5py
from collections import Counter


def get_bpe_word_end_positions(bpe_align, bpe_blank_idx, bpe_vocab):
  bpe_positions = np.arange(len(bpe_align))
  bpe_blank_mask = bpe_align == bpe_blank_idx
  bpe_non_blanks = bpe_align[~bpe_blank_mask]
  bpe_positions = bpe_positions[~bpe_blank_mask]
  bpe_strings = np.array([bpe_vocab[idx] for idx in bpe_non_blanks])
  bpe_word_end_mask = ~np.char.endswith(bpe_strings, "@@")
  bpe_word_end_positions = bpe_positions[bpe_word_end_mask]
  return bpe_word_end_positions


def get_bpe_to_word_idx_mapping(bpe_align, blank_idx):
  bpe_labels = bpe_align[bpe_align != blank_idx]
  word_idx = 0
  word_indices = []
  for token in bpe_labels:
    word_indices.append(word_idx)
    if not token.endswith("@@"):
      word_idx += 1
  return word_indices


def get_allophone_word_end_positions(allophone_idxs, state_tying_vocab, silence_idx, non_final_idx):
  """
  Get indices of allophones that correspond to word ends, including silence.
  :return:
  """
  allophone_idxs = np.array(allophone_idxs)
  allophone_positions = np.arange(len(allophone_idxs))
  # remove non-final states
  final_state_mask = allophone_idxs != non_final_idx
  allophone_idxs = allophone_idxs[final_state_mask]
  allophone_positions = allophone_positions[final_state_mask]
  # remove repetitions
  allophone_reps = np.append(allophone_idxs[1:] == allophone_idxs[:-1], False)
  allophone_idxs = allophone_idxs[~allophone_reps]
  allophone_positions = allophone_positions[~allophone_reps]
  # get final states, e.g. for A{...}@f.0, the final state is 0
  final_states = np.array([int(state_tying_vocab[idx][-1]) for idx in allophone_idxs])
  is_silence = allophone_idxs == silence_idx
  # if final state is greater than next final state, there is a word end
  # e.g. A{...}@f.0 A{...}@f.1 A{...}@f.2 B{...}@f.0 -> word ends after A{...}@f.2
  is_word_end = np.append(final_states[:-1] > final_states[1:], [True])
  # silence is always [SILENCE]@i@f.0,
  # so we need to add it to the word ends because it is not counted in the previous line
  is_word_end = np.logical_or(is_word_end, is_silence)
  word_ends = allophone_idxs[is_word_end]
  word_end_positions = allophone_positions[is_word_end]
  return word_end_positions


def calculate_max_word_duration_deviation(
        phoneme_align,
        bpe_align,
        bpe_blank_idx,
        bpe_vocab,
        bpe_downsampling_factor,
        state_tying_vocab,
        phoneme_silence_idx,
        phoneme_non_final_idx,
        phoneme_downsampling_factor,
        frame_len
):
  # get word end positions
  phoneme_word_end_positions = get_allophone_word_end_positions(
    phoneme_align, state_tying_vocab, phoneme_silence_idx, phoneme_non_final_idx)
  bpe_word_end_positions = get_bpe_word_end_positions(bpe_align, bpe_blank_idx, bpe_vocab)
  assert len(phoneme_word_end_positions) == len(bpe_word_end_positions), "Number of word ends do not match"
  # get phoneme word durations
  phoneme_word_end_positions = np.append([-1], phoneme_word_end_positions)
  phoneme_align_word_durations = np.diff(phoneme_word_end_positions) * phoneme_downsampling_factor * frame_len
  # get BPE word durations
  bpe_word_end_positions = np.append([-1], bpe_word_end_positions)
  bpe_align_word_durations = np.diff(bpe_word_end_positions) * bpe_downsampling_factor * frame_len
  # calculate maximum word duration deviation
  max_word_duration_deviation = np.max(np.absolute(phoneme_align_word_durations - bpe_align_word_durations))
  return max_word_duration_deviation


def calculate_bpe_word_boundary_deviation(
        bpe_positions,
        bpe_to_word_idx_mapping,
        phoneme_word_end_positions,
):
  """
  Calculate the deviation between BPE word boundaries and phoneme word boundaries.
  :param bpe_positions:
  :param bpe_to_word_idx_mapping:
  :param phoneme_word_end_positions:
  :return:
  """
  bpe_word_end_positions = bpe_positions[bpe_to_word_idx_mapping]
  bpe_word_end_positions = np.append([-1], bpe_word_end_positions)
  bpe_word_durations = np.diff(bpe_word_end_positions)
  phoneme_word_durations = np.diff(phoneme_word_end_positions)
  word_boundary_deviation = np.max(np.absolute(bpe_word_durations - phoneme_word_durations))
  return word_boundary_deviation



def write_word_duration_deviation_statistics(max_word_duration_deviation_counter):
  def _get_percent_word_duration_deviations_below(max_deviation, num_seqs):
    return sum([v for k, v in max_word_duration_deviation_counter.items() if k < max_deviation]) / num_seqs * 100
  num_seqs = sum(max_word_duration_deviation_counter.values())
  percent_max_word_duration_deviations_below = {
    i: _get_percent_word_duration_deviations_below(i, num_seqs) for i in range(0, 1000, 100)
  }
  with open("statistics", "w+") as f:
    f.write(f"BPE word durations vs. GMM word durations\n")
    f.write(" Percent of max word duration deviations below <X>ms:\n")
    for k, v in percent_max_word_duration_deviations_below.items():
      f.write(f"  {k}s: {v:.2f}%\n")


def compare_alignments(
        sprint_cache_dataset,
        hdf_data,
        seq_tags,
        seq_lens,
        bpe_downsampling_factor,
        bpe_blank_idx,
        bpe_vocab,
        phoneme_downsampling_factor,
        phoneme_non_final_idx,
        phoneme_silence_idx,
        state_tying_vocab,
        frame_len=10,
):
  """

  :param sprint_cache_dataset:
  :param hdf_data:
  :param seq_tags:
  :param seq_lens:
  :param bpe_downsampling_factor:
  :param bpe_blank_idx:
  :param bpe_vocab:
  :param phoneme_downsampling_factor:
  :param phoneme_non_final_idx:
  :param phoneme_silence_idx:
  :param state_tying_vocab:
  :param frame_len: in ms
  :return:
  """
  seq_idx = 0
  max_word_duration_deviation_counter = Counter()
  while sprint_cache_dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    if seq_idx % 1000 == 0:
      complete_frac = sprint_cache_dataset.get_complete_frac(seq_idx)
      print(f"Progress: {complete_frac * 100:.2f}% ({seq_idx}/{len(seq_tags)})")

    if seq_idx > 100:
      break

    # find corresponding BPE alignment for current phoneme alignment
    sprint_seq_tag = sprint_cache_dataset.get_tag(seq_idx)
    i = 0
    for hdf_seq_tag in seq_tags:
      if hdf_seq_tag == sprint_seq_tag:
        break
      i += 1
    bpe_seq_len = seq_lens[i]
    # load BPE alignment
    bpe_align = hdf_data[sum(seq_lens[:i]):sum(seq_lens[:i + 1])]
    assert len(bpe_align) == bpe_seq_len, "Something went wrong when aligning the two datasets"
    # load phoneme alignment
    sprint_cache_dataset.load_seqs(seq_idx, seq_idx + 1)
    phoneme_align = sprint_cache_dataset.get_data(seq_idx, "data")
    max_word_duration_deviation = calculate_max_word_duration_deviation(
      phoneme_align,
      bpe_align,
      bpe_blank_idx,
      bpe_vocab,
      bpe_downsampling_factor,
      state_tying_vocab,
      phoneme_silence_idx,
      phoneme_non_final_idx,
      phoneme_downsampling_factor,
      frame_len
    )
    max_word_duration_deviation_counter[max_word_duration_deviation] += 1

    seq_idx += 1

  write_word_duration_deviation_statistics(max_word_duration_deviation_counter)


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


def get_sprint_cache_dataset(alignment_cache, silence_phone, allophone_file, state_tying_file, segment_file):
  from returnn import datasets
  sprint_cache_dataset_dict = {
    "class": "SprintCacheDataset",
    "data": {
      "data": {
        "filename": alignment_cache,
        "data_type": "align",
        "allophone_labeling": {
          "silence_phone": silence_phone,
          "allophone_file": allophone_file,
          "state_tying_file": state_tying_file,
        },
      }
    },
    "seq_list_filter_file": segment_file
  }

  dataset = datasets.init_dataset(sprint_cache_dataset_dict)
  dataset.init_seq_order(1)
  return dataset


def get_hdf_data(bpe_hdf):
  with h5py.File(bpe_hdf, "r") as f:
    hdf_data = f["inputs"][()]
    seq_tags = f["seqTags"][()]
    seq_lens = f["seqLengths"][()][:, 0]

  # convert from byte to string
  seq_tags = [tag.decode("utf-8") for tag in seq_tags]

  return hdf_data, seq_tags, seq_lens


def get_bpe_vocab(bpe_vocab_path):
  with open(bpe_vocab_path, "r") as f:
    bpe_vocab = ast.literal_eval(f.read())
  bpe_vocab = {v: k for k, v in bpe_vocab.items()}
  return bpe_vocab


def write_state_tying(allophone_file):
  """
  Write state tying to file and return state tying as dict. We map all allophone states, which don't correspond
  to word ends (i.e. don't end with "f"), to the same index. This is because we are only interested in word ends and
  this makes it easy to remove all other, irrelevant states from the allophone seqs.
  :param allophone_file:
  :return:
  """
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
  return "state-tying", {v: k for k, v in state_tying.items()}, silence_idx, non_final_idx


def main():
  arg_parser = argparse.ArgumentParser(description="Compare BPE and GMM alignments.")
  arg_parser.add_argument("--bpe_align_hdf", help="hdf file which contains the extracted bpe alignments")
  arg_parser.add_argument("--bpe_vocab", help="path to BPE vocab")
  arg_parser.add_argument("--bpe_blank_idx", help="the blank index in the bpe alignment", type=int)
  arg_parser.add_argument("--bpe_downsampling_factor", type=int)
  arg_parser.add_argument("--phoneme_align_cache", help="RASR alignment cache with phoneme alignments")
  arg_parser.add_argument("--phoneme_downsampling_factor", type=int)
  arg_parser.add_argument("--allophone_file", help="RASR allophone file")
  arg_parser.add_argument("--silence_phone", help="Phoneme representing silence, e.g. [SILENCE]")
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
  state_tying_path, state_tying_vocab, phoneme_silence_idx, phoneme_non_final_idx = write_state_tying(args.allophone_file)
  sprint_cache_dataset = get_sprint_cache_dataset(
    args.phoneme_align_cache,
    args.silence_phone,
    args.allophone_file,
    state_tying_path,
    args.segment_file
  )
  hdf_data, seq_tags, seq_lens = get_hdf_data(args.bpe_align_hdf)

  try:
    compare_alignments(
      sprint_cache_dataset,
      hdf_data,
      seq_tags,
      seq_lens,
      args.bpe_downsampling_factor,
      args.bpe_blank_idx,
      get_bpe_vocab(args.bpe_vocab),
      args.phoneme_downsampling_factor,
      phoneme_non_final_idx,
      phoneme_silence_idx,
      state_tying_vocab
    )
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn_main.finalize()


if __name__ == "__main__":
  main()
