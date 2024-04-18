"""
This script takes an HDF file with BPE sequences (without explicit silence) and adds silence tokens to the sequences.
It does this by using a RASR alignment cache (with phoneme labels) and finding the position of silence in between
words. The silence token is then inserted into the BPE sequence at the corresponding position.
"""


import argparse
import ast
import sys
import numpy as np
import tensorflow as tf
import h5py


def get_hdf_out_dataset(out_dim):
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_hdf.hdf", dim=out_dim, ndim=1)


def get_allophone_word_ends(allophone_idxs, state_tying_vocab, silence_idx, non_final_idx):
  """
  Get indices of allophones that correspond to word ends, including silence.
  :param archiver_tool:
  :param allophone_file:
  :param phoneme_align_cache:
  :param seq_tag:
  :return:
  """
  allophone_idxs = np.array(allophone_idxs)
  # remove non-final states
  allophone_idxs = allophone_idxs[allophone_idxs != non_final_idx]
  # remove repetitions
  allophone_reps = np.append(allophone_idxs[1:] == allophone_idxs[:-1], False)
  allophone_idxs = allophone_idxs[~allophone_reps]
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
  return word_ends


def add_silence_to_labels(
        hdf_data,
        seq_tags,
        bpe_seq_lens,
        out_dataset,
        phoneme_silence_idx,
        non_final_idx,
        bpe_vocab,
        bpe_silence_idx,
        sprint_cache_dataset,
        state_tying_vocab,
):
  seq_idx = 0

  sprint_cache_dataset.init_seq_order(1)

  while sprint_cache_dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    if seq_idx % 1000 == 0:
      print(f"Progress: {seq_idx / len(seq_tags) * 100:.2f}% ({seq_idx}/{len(seq_tags)})")

    # check whether datasets align
    seq_tag = sprint_cache_dataset.get_tag(seq_idx)
    assert seq_tag == seq_tags[seq_idx], "Sequence tags do not match"
    bpe_seq_len = bpe_seq_lens[seq_idx]

    # load allophone sequence and find word ends in the phoneme alignment
    sprint_cache_dataset.load_seqs(seq_idx, seq_idx + 1)
    allophone_idxs = sprint_cache_dataset.get_data(seq_idx, "data")
    allophone_word_ends = get_allophone_word_ends(
      allophone_idxs,
      state_tying_vocab,
      phoneme_silence_idx,
      non_final_idx
    )
    allophone_word_ends_no_silence = allophone_word_ends[allophone_word_ends != phoneme_silence_idx]

    # load BPE data
    bpe_seq = hdf_data[:bpe_seq_len]
    # cut off the data that was just read
    hdf_data = hdf_data[bpe_seq_len:]
    bpe_string_seq = np.array([bpe_vocab[idx] for idx in bpe_seq])
    bpe_word_end_positions = np.where(~np.char.endswith(bpe_string_seq, "@@"))[0]

    assert len(allophone_word_ends_no_silence) == len(bpe_word_end_positions), (
      "Number of word ends in BPE and phoneme alignment do not match")

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
    new_seq_lens = {0: tf.constant([new_seq_len]).numpy()}
    ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in new_seq_lens:
        new_seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(new_seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(new_seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    out_dataset.insert_batch(new_data, seq_len=new_seq_lens, seq_tag=[seq_tag], extra=extra)

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

  # convert from byte to string
  seq_tags = [tag.decode("utf-8") for tag in seq_tags]

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
          "state_tying_file": state_tying_file,
        },
      }
    },
  }
  dataset = datasets.init_dataset(dataset_dict)
  return dataset


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
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("--bpe_seqs_hdf", help="hdf file which contains BPE sequences")
  arg_parser.add_argument("--bpe_vocab", help="path to BPE vocab")
  arg_parser.add_argument("--phoneme_align_cache", help="RASR alignment cache with phoneme alignments")
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
  hdf_data, seq_tags, seq_lens = get_hdf_data(args.bpe_seqs_hdf)
  bpe_vocab = get_bpe_vocab(args.bpe_vocab)
  bpe_silence_idx = max(bpe_vocab.keys()) + 1
  out_dataset = get_hdf_out_dataset(out_dim=bpe_silence_idx + 1)

  state_tying_path, state_tying_vocab, silence_idx, non_final_idx = write_state_tying(args.allophone_file)
  sprint_cache_dataset = get_sprint_cache_dataset(
    args.phoneme_align_cache,
    args.silence_phone,
    args.allophone_file,
    state_tying_path
  )

  try:
    add_silence_to_labels(
      hdf_data,
      seq_tags,
      seq_lens,
      out_dataset,
      silence_idx,
      non_final_idx,
      bpe_vocab,
      bpe_silence_idx,
      sprint_cache_dataset,
      state_tying_vocab,
    )
    out_dataset.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn_main.finalize()


if __name__ == "__main__":
  main()
