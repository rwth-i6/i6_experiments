import argparse
import sys
import numpy as np
import tensorflow as tf


def hdf_dataset_init(dim):
  """
  :param str file_name: filename of hdf dataset file in the filesystem
  :rtype: hdf_dataset_mod.HDFDatasetWriter
  """
  import returnn.datasets.hdf as hdf_dataset_mod
  return hdf_dataset_mod.SimpleHDFWriter(
    filename="out_hdf_align", dim=dim, ndim=1)


def dump_into_hdf(align_seqs, tags, hdf_dataset_out):
  for i in range(len(align_seqs)):
    align = np.array(align_seqs[i])
    seq_len = len(align)
    tag = tags[i]
    new_data = tf.constant(np.expand_dims(align, axis=0), dtype="int32")

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

    hdf_dataset_out.insert_batch(new_data, seq_len=seq_lens, seq_tag=[tag], extra=extra)


def init(align_txt_file, segment_file):
  with open(align_txt_file, "r") as f:
    align_seqs = []
    for line in f:
      if line != "":
        align_seq = [int(idx) for idx in line.strip().split(" ")]
        align_seqs.append(align_seq)

  with open(segment_file, "r") as f:
    tags = []
    for line in f:
      if line != "":
        tags.append(line.strip())

  assert len(align_seqs) == len(tags)

  return align_seqs, tags


def main():
  arg_parser = argparse.ArgumentParser(description="Dump non-blanks of alignment into hdf file.")
  arg_parser.add_argument("align_txt_file", help="file with one align seq sep by blanks per line")
  arg_parser.add_argument("--segment_file", help="file with one seq tag per line")
  arg_parser.add_argument("--num_classes", help="number of classes in alignment", type=int)
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  align_seqs, tags = init(args.align_txt_file, args.segment_file)

  hdf_dataset_out = hdf_dataset_init(dim=args.num_classes)

  try:
    dump_into_hdf(align_seqs,  tags, hdf_dataset_out)
    hdf_dataset_out.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
