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


def dump_into_hdf(align_seqs, tags, hdf_dataset_out, num_classes):
  for i in range(len(align_seqs)):
    align = np.array(align_seqs[i])
    if align[-1] == num_classes:
      if num_classes == 1032:
        align[-1] = 1030
      else:
        assert num_classes == 1031
        align[-1] = 0
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


def hdf_dump_from_dataset(dataset, hdf_dataset):
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
    # try:
    dataset.load_seqs(seq_idx, seq_idx + 1)
    # except:
    #   print("ERROR WHILE LOADING SEQ")
    #   continue
    data = dataset.get_data(seq_idx, "classes")
    # print("TAG: ", dataset.get_tag(seq_idx))
    #
    # print("DATA: ", data)

    seq_len = dataset.get_seq_length(seq_idx)["classes"]
    tag = dataset.get_tag(seq_idx)

    new_data = tf.constant(np.expand_dims(data, axis=0), dtype="int32")

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


def init(
  rasr_config_file):
  sprint_args = [
    "--config=%s" % rasr_config_file,
    "--*.LOGFILE=nn-trainer.train.log", "--*.TASK=1", "--*.corpus.segment-order-shuffle=true",
  ]

  align_dataset_dict = {
    "class": "ExternSprintDataset",
    "reduce_target_factor": 6,
    "sprintConfigStr": sprint_args,
    "sprintTrainerExecPath": "/u/schmitt/src/rasr/arch/linux-x86_64-standard/nn-trainer.linux-x86_64-standard",
    "suppress_load_seqs_print": True,
  }

  align_dataset = rnn.datasets.init_dataset(align_dataset_dict)
  print("Source dataset:", align_dataset.get_data_keys())

  return align_dataset


def main():
  arg_parser = argparse.ArgumentParser(description="Dump non-blanks of alignment into hdf file.")
  arg_parser.add_argument("--rasr_config_file")
  arg_parser.add_argument("--num_classes", help="number of classes in alignment", type=int)
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn as rnn
  import returnn.__main__ as rnn_main
  import returnn.tf.compat as tf_compat
  tf_compat.v1.enable_eager_execution()

  align_dataset = init(
    rasr_config_file=args.rasr_config_file,
  )

  hdf_dataset_out = hdf_dataset_init(dim=args.num_classes)

  try:
    # dump_into_hdf(align_seqs,  tags, hdf_dataset_out, num_classes=args.num_classes)
    hdf_dump_from_dataset(dataset=align_dataset, hdf_dataset=hdf_dataset_out)
    hdf_dataset_out.close()
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn_main.finalize()


if __name__ == "__main__":
  main()
