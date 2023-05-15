import argparse
import sys
import numpy as np
import os

import _setup_returnn_env  # noqa
from returnn import __main__ as rnn
from returnn.datasets import init_dataset, Dataset
from returnn.log import log

dataset = None


def calc_segment_stats(blank_idx, segment):
  print("inside calc_segment_stats")
  if segment == "total":
    segment_idxs = ":"
    initial_idx = "0"
  elif segment == "first":
    segment_idxs = ":1"
    initial_idx = "0"
  elif segment == "except_first":
    segment_idxs = "1:"
    initial_idx = "non_blank_idxs[0]"
  else:
    raise ValueError("segment definition unknown")
  print("before seq order")
  dataset.init_seq_order()
  print("after seq order")
  seq_idx = 0
  seg_total_len = 0
  num_segs = 0
  num_blanks = 0
  num_non_blanks = 0
  print("before loop")
  while dataset.is_less_than_num_seqs(seq_idx) and seq_idx <= 10:
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "classes")
    # print(data)
    non_blank_idxs = np.where(data != blank_idx)[0]
    num_non_blanks += len(non_blank_idxs)
    num_blanks += len(data) - len(non_blank_idxs)
    if non_blank_idxs.size == 0:
      seq_idx += 1
      continue
    else:
      # non_blank_idxs = np.append(non_blank_idxs)
      prev_i = eval(initial_idx)
      try:
        for i in eval("non_blank_idxs[" + segment_idxs + "]"):
          # each non-blank idx corresponds to one segment
          num_segs += 1
          # the segment length is the difference from the previous border to the current one
          # for the first and last segment, the result needs to be corrected (see after while)
          seg_total_len += i - prev_i
          prev_i = i
      except IndexError:
        continue

    seq_idx += 1
  print("after loop")
  # the first segment is always 1 too short
  if segment == "first":
    seg_total_len += num_segs

  mean_seg_len = seg_total_len / num_segs

  filename = "mean.seg.len"
  mode = "w"
  if os.path.exists(filename):
    mode = "a"
  with open(filename, mode) as f:
    f.write(segment + ": " + str(mean_seg_len) + "\n")

  filename = "blank_ratio"
  if not os.path.exists(filename):
    with open(filename, "w+") as f:
      f.write("Blanks: " + str(num_blanks) + "\n")
      f.write("Non Blanks: " + str(num_non_blanks) + "\n")

config = None

def init(returnn_config, seq_list_filter_file):
  # rnn.init_better_exchook()
  # rnn.init_thread_join_hack()
  # dataset_dict = {
  #   'class': 'ExternSprintDataset',
  #   'sprintConfigStr': '--config=/u/schmitt/experiments/transducer/config/rasr-configs/zhou-phon-trans.config --*.LOGFILE=nn-trainer.train.log --*.TASK=1 --*.corpus.segment-order-shuffle=true',
  #   'sprintTrainerExecPath': '/u/zhou/rasr-dev/arch/linux-x86_64-standard-label_sync_decoding/nn-trainer.linux-x86_64-standard-label_sync_decoding'}
  #
  # rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  # global config
  # config = rnn.config
  # config.set("log", None)
  # global dataset
  # dataset = rnn.init_dataset(dataset_dict)
  # rnn.init_log()
  # print("Returnn segment-statistics starting up", file=rnn.log.v2)
  # rnn.returnn_greeting()
  # rnn.init_faulthandler()
  # rnn.init_config_json_network()
  global dataset
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  config_filename = returnn_config
  print("Using config file %r." % config_filename)
  assert os.path.exists(config_filename)
  rnn.init_config(config_filename=config_filename, default_config={"cache_size": "0"})
  global config
  config = rnn.config
  config.set("log", None)

  print("Use train dataset from config.")
  assert config.value("train", None)
  dataset = init_dataset("config:train")
  rnn.init_log()
  print("Returnn dump-dataset starting up.", file=log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()
  print("Dataset:", file=log.v2)
  print("  input:", dataset.num_inputs, "x", dataset.window, file=log.v2)
  print("  output:", dataset.num_outputs, file=log.v2)
  print("before dataset len info")
  print(" ", dataset.len_info() or "no info", file=log.v2)
  print("at the end of init")


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("returnn_config", help="hdf file which contains the extracted alignments of some corpus")
  arg_parser.add_argument("--seq-list-filter-file", help="whitelist of sequences to use", default=None)
  arg_parser.add_argument("--blank-idx", help="the blank index in the alignment", default=0, type=int)
  arg_parser.add_argument("--segment", help="over which segments to calculate the statistics: 'total', 'first', "
                                            "'except_first', 'all' (default: 'all')", default="all")
  args = arg_parser.parse_args()
  assert args.segment in ["first", "except_first", "all", "total"]

  print("enter init")
  init(args.returnn_config, args.seq_list_filter_file)
  print("leave init")

  try:
    # print("dumping")
    # filename = "mean.seg.len"
    # with open(filename, "w+") as f:
    #   f.write("test")
    #
    # filename = "blank_ratio"
    # with open(filename, "w+") as f:
    #   f.write("test")
    if args.segment == "all":
      for seg in ["total", "first", "except_first"]:
        print("enter calc_segment_stats")
        calc_segment_stats(args.blank_idx, seg)
    else:
      calc_segment_stats(args.blank_idx, args.segment)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
