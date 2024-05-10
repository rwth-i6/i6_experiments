import argparse
import ast
import json
import sys
import numpy as np
import os
import xml, gzip
import xml.etree.ElementTree as ET
from xml import etree
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time


def hdf_dataset_init(out_dim):
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_align.hdf", dim=out_dim, ndim=1)


def create_augmented_alignment(
        meta_dataset,
        out_dataset,
        bpe_downsampling_factor,
        bpe_blank_idx,
        phoneme_blank_idx,
        phoneme_sil_idx
):
  meta_dataset.init_seq_order()
  seq_idx = 0

  while meta_dataset.is_less_than_num_seqs(seq_idx) and seq_idx < float("inf"):
    if seq_idx % 1000 == 0:
      complete_frac = meta_dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))

    meta_dataset.load_seqs(seq_idx, seq_idx + 1)
    bpe_align = meta_dataset.get_data(seq_idx, "bpe_align")
    if np.all(bpe_align == bpe_blank_idx):
      raise ValueError("BPE alignment only consists of blanks")
    phoneme_align = meta_dataset.get_data(seq_idx, "data")
    phoneme_align_silence = np.copy(phoneme_align)
    blank_or_silence_mask = np.logical_or(phoneme_align == phoneme_blank_idx, phoneme_align == phoneme_sil_idx)
    phoneme_align_silence[~blank_or_silence_mask] = phoneme_blank_idx

    # downsample phoneme alignment by replacing patches of size bpe_downsampling_factor with single labels (silence or blank)
    phoneme_align_silence_downsampled = []
    i = 0
    while i < len(phoneme_align_silence):
      if i <= len(phoneme_align_silence) - bpe_downsampling_factor:
        align_patch = phoneme_align_silence[i:i + bpe_downsampling_factor]
      else:
        align_patch = phoneme_align_silence[i:]

      num_silence_segments = np.sum(align_patch == phoneme_sil_idx)
      if num_silence_segments == 0:
        phoneme_align_silence_downsampled.append(phoneme_blank_idx)
      elif num_silence_segments == 1:
        phoneme_align_silence_downsampled.append(phoneme_sil_idx)
      else:
        # assert that there is only one silence segment in the patch, otherwise the downsampled alignment would be ambiguous
        assert np.sum(align_patch == phoneme_sil_idx) <= 1, f"More than one silence segment in patch: {align_patch}"

      i += bpe_downsampling_factor

    # extend or cut the downsampled phoneme alignment to match the length of the BPE alignment (very crude fix)
    if len(phoneme_align_silence_downsampled) != len(bpe_align):
      len_diff = len(bpe_align) - len(phoneme_align_silence_downsampled)
      if len_diff > 0:
        phoneme_align_silence_downsampled[-1] = phoneme_blank_idx
        phoneme_align_silence_downsampled += [phoneme_blank_idx] * len_diff
      else:
        phoneme_align_silence_downsampled = phoneme_align_silence_downsampled[:len(bpe_align)]
    # add silence in last frame if in original alignment
    if phoneme_align[-1] == phoneme_sil_idx:
      phoneme_align_silence_downsampled[-1] = phoneme_sil_idx

    phoneme_align_silence_downsampled = np.array(phoneme_align_silence_downsampled)
    assert len(phoneme_align_silence_downsampled) == len(bpe_align), "Length of downsampled phoneme alignment does not match length of BPE alignment"

    # add silence from the downsampled phoneme alignment to the BPE alignment
    bpe_sil_idx = meta_dataset.get_data_dim("bpe_align")
    silence_mask = phoneme_align_silence_downsampled == phoneme_sil_idx
    bpe_non_blank_mask = bpe_align != bpe_blank_idx
    bpe_sil_align = np.copy(bpe_align)
    bpe_sil_align[np.logical_and(silence_mask, bpe_non_blank_mask)] = bpe_sil_idx

    # dump new alignment into hdf file
    seq_len = len(bpe_sil_align)
    tag = meta_dataset.get_tag(seq_idx)
    new_data = tf.constant(np.expand_dims(bpe_sil_align, axis=0), dtype="int32")
    extra = {}
    seq_lens = {0: tf.constant([seq_len]).numpy()}
    ndim_without_features = 1  # - (0 if data_obj.sparse or data_obj.feature_dim_axis is None else 1)
    for dim in range(ndim_without_features):
      if dim not in seq_lens:
        seq_lens[dim] = np.array([new_data.shape[dim + 1]] * 1, dtype="int32")
    batch_seq_sizes = np.zeros((1, len(seq_lens)), dtype="int32")
    for i, (axis, size) in enumerate(sorted(seq_lens.items())):
      batch_seq_sizes[:, i] = size
    extra["seq_sizes"] = batch_seq_sizes

    out_dataset.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)

    seq_idx += 1


def init_returnn():
  global config

  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  config = rnn.config
  config.set("log", None)
  rnn.init_log()
  print("Returnn augment_bpe_align starting up", file=rnn.log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()


def get_meta_dataset(bpe_hdf, phoneme_hdf, segment_file):
  # init bpe and phoneme align datasets
  bpe_dataset_dict = {
    "class": "HDFDataset",
    "files": [bpe_hdf],
    "use_cache_manager": True,
    "partition_epoch": 1,
    "seq_list_filter_file": segment_file
  }
  phoneme_dataset_dict = {
    "class": "HDFDataset",
    "files": [phoneme_hdf],
    "use_cache_manager": True,
    "partition_epoch": 1,
    'seq_list_filter_file': segment_file
  }

  dataset_dict = {
    'class': 'MetaDataset',
    'data_map': {'bpe_align': ('bpe_align', 'data'), 'data': ('data', 'data')},
    'datasets': {'bpe_align': bpe_dataset_dict, "data": phoneme_dataset_dict},
    'seq_order_control_dataset': 'data'
  }

  dataset = rnn.init_dataset(dataset_dict)
  return dataset


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("--bpe_align_hdf", help="hdf file which contains the extracted bpe alignments")
  arg_parser.add_argument("--phoneme_align_hdf", help="hdf file which contains the extracted phoneme alignments")
  arg_parser.add_argument("--bpe_blank_idx", help="the blank index in the bpe alignment", type=int)
  arg_parser.add_argument("--phoneme_blank_idx", help="the blank index in the phoneme alignment", type=int)
  arg_parser.add_argument("--phoneme_sil_idx", help="the blank index in the phoneme alignment", type=int)
  arg_parser.add_argument("--segment_file", help="segment whitelist", type=str)
  arg_parser.add_argument("--bpe_downsampling_factor", type=int)
  arg_parser.add_argument("--returnn_root", type=str)
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  init_returnn()
  meta_dataset = get_meta_dataset(args.bpe_align_hdf, args.phoneme_align_hdf, args.segment_file)
  out_dataset = hdf_dataset_init(out_dim=meta_dataset.get_data_dim("bpe_align") + 1)

  try:
    create_augmented_alignment(
      meta_dataset,
      out_dataset,
      args.bpe_downsampling_factor,
      args.bpe_blank_idx,
      args.phoneme_blank_idx,
      args.phoneme_sil_idx
    )
    out_dataset.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
