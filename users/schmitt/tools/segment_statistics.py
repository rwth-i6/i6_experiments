import argparse
import ast
import json
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def calc_segment_stats_with_sil(blank_idx, sil_idx, vocab):
  dataset.init_seq_order()
  seq_idx = 0

  # silence segment stats
  # init, inter, final segment lens
  inter_sil_seg_len = 0
  init_sil_seg_len = 0
  final_sil_seg_len = 0
  # init, inter, final segment counts
  num_sil_segs = 0
  num_init_sil_segs = 0
  num_final_sil_segs = 0
  # counts of segment lengths
  map_sil_seg_len_to_count = Counter()

  # label segment stats
  # number of non-blank segments
  num_label_segs = 0
  # total segment length over all non-blank labels
  label_seg_len = 0
  # segment lengths per non-blank label
  label_dependent_seg_lens = Counter()
  label_dependent_num_segs = Counter()
  # counts of segment lengths
  map_non_sil_seg_len_to_count = Counter()

  # number of blank frames
  num_blank_frames = 0
  # number of sequences
  num_seqs = 0
  # number of words
  num_words_except_first = 0
  num_words_total = 0

  # max segment length
  max_seg_len = 0
  # max sequence length
  max_seq_len = 0

  # label repetition statistics
  num_label_transitions = 0
  num_label_repetitions = 0
  num_seqs_w_label_repetitions = 0
  num_words_w_label_repetitions = 0
  # character repetition statistics
  num_character_transitions = 0
  num_character_repetitions = 0
  num_seqs_w_character_repetitions = 0
  num_words_w_character_repetitions = 0

  while dataset.is_less_than_num_seqs(seq_idx):
    num_seqs += 1
    # progress indication
    if seq_idx % 1000 == 0:
      complete_frac = dataset.get_complete_frac(seq_idx)
      print("Progress: %.02f" % (complete_frac * 100))
    dataset.load_seqs(seq_idx, seq_idx + 1)
    data = dataset.get_data(seq_idx, "data")

    # find non-blanks and silence
    non_blank_idxs = np.where(data != blank_idx)[0]
    sil_idxs = np.where(data == sil_idx)[0]

    non_blank_data = data[data != blank_idx]

    # find percentage of label repetitions
    repetitions = non_blank_data[1:] == non_blank_data[:-1]
    num_repetitions = np.sum(repetitions)
    num_label_repetitions += num_repetitions
    num_label_transitions += len(non_blank_data) - 1
    num_seqs_w_label_repetitions += np.sum(repetitions) > 0
    if num_repetitions > 0:
      print("Seq with label repetitions: ", [vocab[idx] for idx in non_blank_data])
      print("\n")
    # needed to count repetitions per word (below) (needs to be reset for each seq)
    num_label_repetitions_cur_word = 0
    num_labels_cur_word = 0
    num_char_repetitions_cur_seq = 0
    cur_word = ""
    seq_str = ""

    # update max sequence length
    if len(data) > max_seq_len:
      max_seq_len = len(data)

    # count number of segments and number of blank frames
    num_label_segs += len(non_blank_idxs) - len(sil_idxs)
    num_sil_segs += len(sil_idxs)
    num_blank_frames += len(data) - len(non_blank_idxs)

    # if there are only blanks, we skip the seq as there are no segments
    if non_blank_idxs.size == 0:
      seq_idx += 1
      continue
    else:
      prev_idx = 0
      try:
        # go through non blanks and count segment len
        # differ between sil_beginning, sil_mid, sil_end and non-sil
        for i, idx in enumerate(non_blank_idxs):
          seg_len = idx - prev_idx
          # first segment is always one too short because of prev_idx = 0
          if prev_idx == 0:
            seg_len += 1

          if seg_len > max_seg_len:
            max_seg_len = seg_len

          label_dependent_seg_lens.update({non_blank_data[i]: seg_len})
          label_dependent_num_segs.update([non_blank_data[i]])

          # count repetitions per word
          if i > 0 and non_blank_data[i] == non_blank_data[i - 1]:
            num_label_repetitions_cur_word += 1

          label_str = vocab[non_blank_data[i]]
          cur_word += label_str.replace("@", "")
          seq_str += label_str.replace("@", "")
          # either whole word or last bpe of a word -> no @@ in label
          if "@@" not in label_str:
            num_words_total += 1
            if i > 0:
              num_words_except_first += 1
            if num_label_repetitions_cur_word > 0:
              num_words_w_label_repetitions += 1

            num_label_repetitions_cur_word = 0

            # count repetitions per character in word
            cur_word = np.array(list(cur_word))
            char_repetitions_in_word = np.sum(cur_word[1:] == cur_word[:-1]) * (len(cur_word) > 1)
            num_character_repetitions += char_repetitions_in_word
            num_char_repetitions_cur_seq += char_repetitions_in_word
            num_character_transitions += len(cur_word) - 1
            num_words_w_character_repetitions += char_repetitions_in_word > 0

            cur_word = ""
            seq_str += " "

          if idx in sil_idxs:
            map_sil_seg_len_to_count.update([seg_len])
            if i == 0:
              init_sil_seg_len += seg_len
              num_init_sil_segs += 1
            elif i == len(non_blank_idxs) - 1:
              final_sil_seg_len += seg_len
              num_final_sil_segs += 1
            else:
              inter_sil_seg_len += seg_len
          else:
            label_seg_len += seg_len
            map_non_sil_seg_len_to_count.update([seg_len])

          prev_idx = idx
      except IndexError:
        continue

    # count repetitions per character in sequence
    if num_char_repetitions_cur_seq > 0:
      num_seqs_w_character_repetitions += 1
      print("Seq with character repetitions: ", seq_str)
      print("\n")

    seq_idx += 1

  mean_init_sil_len = init_sil_seg_len / num_init_sil_segs if num_init_sil_segs > 0 else 0
  mean_final_sil_len = final_sil_seg_len / num_final_sil_segs if num_final_sil_segs > 0 else 0
  mean_inter_sil_len = inter_sil_seg_len / (num_sil_segs - num_init_sil_segs - num_final_sil_segs) if inter_sil_seg_len > 0 else 0
  mean_total_sil_len = (init_sil_seg_len + final_sil_seg_len + inter_sil_seg_len) / num_seqs

  mean_label_len = label_seg_len / num_label_segs
  mean_total_label_len = label_seg_len / num_seqs

  label_dependent_mean_seg_lens = {int(idx): label_dependent_seg_lens[idx] / label_dependent_num_segs[idx] for idx in label_dependent_seg_lens }
  label_dependent_mean_seg_lens.update({idx: mean_label_len for idx in range(blank_idx) if idx not in label_dependent_mean_seg_lens})

  mean_seq_len = (num_blank_frames + num_sil_segs + num_label_segs) / num_seqs

  num_segments_shorter2 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 2])
  num_segments_shorter4 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 4])
  num_segments_shorter8 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 8])
  num_segments_shorter16 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 16])
  num_segments_shorter21 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 21])
  num_segments_shorter32 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 32])
  num_segments_shorter64 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 64])
  num_segments_shorter128 = sum([count for seg_len, count in map_non_sil_seg_len_to_count.items() if seg_len < 128])

  filename = "statistics"
  with open(filename, "w+") as f:
    f.write("Segment statistics: \n\n")
    f.write("\tSilence: \n")
    f.write("\t\tInitial:\n")
    f.write("\t\t\tMean length: %f \n" % mean_init_sil_len)
    f.write("\t\t\tNum segments: %f \n" % num_init_sil_segs)
    f.write("\t\tIntermediate:\n")
    f.write("\t\t\tMean length: %f \n" % mean_inter_sil_len)
    f.write("\t\t\tNum segments: %f \n" % (num_sil_segs - num_init_sil_segs - num_final_sil_segs))
    f.write("\t\tFinal:\n")
    f.write("\t\t\tMean length: %f \n" % mean_final_sil_len)
    f.write("\t\t\tNum segments: %f \n" % num_final_sil_segs)
    f.write("\t\tTotal per sequence:\n")
    f.write("\t\t\tMean length: %f \n" % mean_total_sil_len)
    f.write("\t\t\tNum segments: %f \n" % num_sil_segs)
    f.write("\n")
    f.write("\tNon-silence: \n")
    f.write("\t\tMean length per segment: %f \n" % mean_label_len)
    f.write("\t\tMean length per sequence: %f \n" % mean_total_label_len)
    f.write("\t\tNum segments: %f \n" % num_label_segs)
    f.write("\t\tPercent segments shorter than x frames: \n")
    f.write("\t\tx = 2: %f \n" % (num_segments_shorter2 / num_label_segs))
    f.write("\t\tx = 4: %f \n" % (num_segments_shorter4 / num_label_segs))
    f.write("\t\tx = 8: %f \n" % (num_segments_shorter8 / num_label_segs))
    f.write("\t\tx = 16: %f \n" % (num_segments_shorter16 / num_label_segs))
    f.write("\t\tx = 21: %f \n" % (num_segments_shorter21 / num_label_segs))
    f.write("\t\tx = 32: %f \n" % (num_segments_shorter32 / num_label_segs))
    f.write("\t\tx = 64: %f \n" % (num_segments_shorter64 / num_label_segs))
    f.write("\t\tx = 128: %f \n" % (num_segments_shorter128 / num_label_segs))
    f.write("\n")
    f.write("Overall maximum segment length: %d \n" % max_seg_len)
    f.write("\n")
    f.write("\n")
    f.write("Sequence statistics: \n\n")
    f.write("\tMean length: %f \n" % mean_seq_len)
    f.write("\tMax length: %f \n" % max_seq_len)
    f.write("\tNum sequences: %f \n" % num_seqs)
    f.write("\n")
    f.write("\n")
    f.write("Whole word statistics: \n\n")
    f.write("\tNum words: %f \n" % num_words_total)
    f.write("\n")
    f.write("\n")
    f.write("Label repetition statistics: \n\n")
    f.write("\tPercent of BPE repetitions: %f%% \n" % (num_label_repetitions / num_label_transitions * 100))
    f.write("\tPercent of words with BPE repetitions: %f%% \n" % (num_words_w_label_repetitions / num_words_except_first * 100))
    f.write("\tPercent of seqs with BPE repetitions: %f%% \n" % (num_seqs_w_label_repetitions / num_seqs * 100))
    f.write("\n")
    f.write("\n")
    f.write("Character repetition statistics: \n\n")
    f.write("\tPercent of character repetitions: %f%% \n" % (num_character_repetitions / num_character_transitions * 100))
    f.write("\tPercent of words with character repetitions: %f%% \n" % (num_words_w_character_repetitions / num_words_total * 100))
    f.write("\tPercent of seqs with character repetitions: %f%% \n" % (num_seqs_w_character_repetitions / num_seqs * 100))

  filename = "mean_non_sil_len"
  with open(filename, "w+") as f:
    f.write(str(float(mean_label_len)))

  filename = "label_dep_mean_lens"
  with open(filename, "w+") as f:
    json.dump(label_dependent_mean_seg_lens, f)

  # plot histograms non-sil segment lens
  hist_data = [item for seg_len, count in map_non_sil_seg_len_to_count.items() for item in [seg_len] * count]
  plt.hist(hist_data, bins=30, range=(0, 50))
  ax = plt.gca()
  quantiles = [np.quantile(hist_data, q) for q in [.90, .95, .99]]
  for n, q in zip([90, 95, 99], quantiles):
    # write quantiles to file
    with open("percentile_%s" % n, "w+") as f:
      f.write(str(q))
    ax.axvline(q, color="r")
  plt.savefig("non_sil_histogram.pdf")
  plt.close()

  # plot histograms sil segment lens
  hist_data = [item for seg_len, count in map_sil_seg_len_to_count.items() for item in [seg_len] * count]
  plt.hist(hist_data, bins=40, range=(0, 100))
  if len(hist_data) != 0:
    ax = plt.gca()
    quantiles = [np.quantile(hist_data, q) for q in [.90, .95, .99]]
    for q in quantiles:
      ax.axvline(q, color="r")
  plt.savefig("sil_histogram.pdf")
  plt.close()


def init(hdf_file, seq_list_filter_file):
  rnn.init_better_exchook()
  rnn.init_thread_join_hack()
  dataset_dict = {
    "class": "HDFDataset", "files": [hdf_file], "use_cache_manager": True, "seq_list_filter_file": seq_list_filter_file
  }

  rnn.init_config(config_filename=None, default_config={"cache_size": 0})
  global config
  config = rnn.config
  config.set("log", None)
  global dataset
  dataset = rnn.init_dataset(dataset_dict)
  rnn.init_log()
  print("Returnn segment-statistics starting up", file=rnn.log.v2)
  rnn.returnn_greeting()
  rnn.init_faulthandler()
  rnn.init_config_json_network()


def main():
  arg_parser = argparse.ArgumentParser(description="Calculate segment statistics.")
  arg_parser.add_argument("hdf_file", help="hdf file which contains the extracted alignments of some corpus")
  arg_parser.add_argument("--seq-list-filter-file", help="whitelist of sequences to use", default=None)
  arg_parser.add_argument("--json-vocab", help="dict mapping idx to word", type=str)
  arg_parser.add_argument("--blank-idx", help="the blank index in the alignment", default=0, type=int)
  arg_parser.add_argument("--sil-idx", help="the blank index in the alignment", default=None, type=int)
  arg_parser.add_argument("--returnn-root", help="path to returnn root")
  args = arg_parser.parse_args()
  sys.path.insert(0, args.returnn_root)
  global rnn
  import returnn.__main__ as rnn

  init(args.hdf_file, args.seq_list_filter_file)

  with open(args.json_vocab, "r") as f:
    vocab = ast.literal_eval(f.read())
    vocab = {int(v): k for k, v in vocab.items()}

  try:
    calc_segment_stats_with_sil(args.blank_idx, args.sil_idx, vocab)
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
