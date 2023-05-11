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
    filename="out_alignment", dim=dim, ndim=1)


def switch_label_and_dump(hdf_dataset_in, hdf_dataset_out, new_idx, orig_idx):

  hdf_dataset_in.init_seq_order()
  seq_idx = 0

  while hdf_dataset_in.is_less_than_num_seqs(seq_idx):
    # progress indication
    if seq_idx % 1000 == 0:
      complete_frac = hdf_dataset_in.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    hdf_dataset_in.load_seqs(seq_idx, seq_idx + 1)
    data = hdf_dataset_in.get_data(seq_idx, "data")

    print("ORIGINAL DATA: ", data)

    orig_idx_mask = data == orig_idx
    new_data = data
    new_data[orig_idx_mask] = new_idx

    print("NEW DATA: ", new_data)

    seq_len = len(new_data)
    tag = hdf_dataset_in.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(new_data, axis=0), dtype="int32")

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

    seq_idx += 1


def init(hdf_file):
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  dataset_dict = {
    "class": "HDFDataset", "files": [hdf_file], "use_cache_manager": True
  }

  rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  global config
  config = rnn.config
  config.set("log", None)
  dataset = rnn.init_dataset(dataset_dict)
  rnn.init_log()
  print("Returnn segment-statistics starting up", file=rnn.log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()

  return dataset


def main():
  arg_parser = argparse.ArgumentParser(description="Replace specific index with blank index in alignment.")
  arg_parser.add_argument("hdf_file", help="hdf file which contains the extracted alignments of some corpus")
  arg_parser.add_argument("--orig_idx", help="the index which should be switched", type=int)
  arg_parser.add_argument("--new_idx", help="the index which should be used instead", type=int)
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  hdf_dataset_in = init(args.hdf_file)
  hdf_dataset_out = hdf_dataset_init(dim=hdf_dataset_in.get_data_dim("data"))

  try:
    switch_label_and_dump(
      hdf_dataset_in, hdf_dataset_out, new_idx=args.new_idx, orig_idx=args.orig_idx)
    hdf_dataset_out.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
