import argparse
import ast
import json
import sys
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, List
import os


class SegmentStatistics:
  """
  Gather statistics about silence and non-silence segments.

  Attributes:
      sil_idx (Optional[int]): The index used to identify silence segments within the data. If None, silence is not
                               explicitly tracked.
      accum_segment_len_per_label (Counter): Accumulates the total length of segments for each non-silence label.
      accum_num_segments_per_label (Counter): Counts the number of segments for each non-silence label.
      accum_segment_len_silence (Counter): Accumulates the total length of silence segments, categorized into
                                           initial, intermediate, and final positions within a sequence.
      accum_num_segments_silence (Counter): Counts the number of silence segments, categorized similarly.
      max_segment_len (int): Tracks the maximum length observed across all segments.
      segment_len_count_statistics (SegmentLenCountStatistics): An instance to further analyze segment lengths,
                                                                including distributions and variance, for each label.

  Methods:
      update(labels, label_positions): Updates the accumulated statistics based on a new set of labels and their
      positions in the alignment.
      get_mean_segment_len_for_non_silence(): Computes the mean length of non-silence segments.
      get_mean_segment_len_for_silence(sil_position): Computes the mean length of silence segments, which can be
                                                      categorized as initial, intermediate, final, or total.
      __str__(): Provides a textual summary of the accumulated statistics.
      plot_histograms(): Generates histograms for the distribution of segment lengths, overall and per label.
  """
  def __init__(self, sil_idx: Optional[int], vocab: Dict[int, str]):
    self.sil_idx = sil_idx

    self.accum_segment_len_per_label = Counter()
    self.accum_num_segments_per_label = Counter()
    self.accum_segment_len_silence = Counter({"init": 0, "inter": 0, "final": 0})
    self.accum_num_segments_silence = Counter({"init": 0, "inter": 0, "final": 0})

    self.max_segment_len = 0

    self.segment_len_count_statistics = SegmentLenCountStatistics(vocab)

  def update(self, labels, label_positions):
    num_labels_in_seq = len(labels)
    for i, label in enumerate(labels):
      if i == 0:
        segment_len = label_positions[i] + 1
      else:
        segment_len = label_positions[i] - label_positions[i - 1]

      if label == self.sil_idx:
        if i == 0:
          self.accum_segment_len_silence["init"] += segment_len
          self.accum_num_segments_silence["init"] += 1
        elif i == num_labels_in_seq - 1:
          self.accum_segment_len_silence["final"] += segment_len
          self.accum_num_segments_silence["final"] += 1
        else:
          self.accum_segment_len_silence["inter"] += segment_len
          self.accum_num_segments_silence["inter"] += 1
      else:
        self.accum_segment_len_per_label[label] += segment_len
        self.accum_num_segments_per_label[label] += 1

      if segment_len > self.max_segment_len:
        self.max_segment_len = segment_len

      self.segment_len_count_statistics.update(label, segment_len)

  def get_mean_segment_len_for_non_silence(self):
    total_len = sum(self.accum_segment_len_per_label.values())
    total_num = sum(self.accum_num_segments_per_label.values())
    if total_num == 0:
      return 0
    return total_len / total_num

  def get_mean_segment_len_for_silence(self, sil_position: str):
    assert sil_position in ("init", "inter", "final", "total")
    if sil_position == "total":
      total_num_segments = sum(self.accum_num_segments_silence.values())
      if total_num_segments == 0:
        return 0
      return sum(self.accum_segment_len_silence.values()) / sum(self.accum_num_segments_silence.values())

    if self.accum_num_segments_silence[sil_position] == 0:
      return 0
    return self.accum_segment_len_silence[sil_position] / self.accum_num_segments_silence[sil_position]

  def __str__(self):
    string = "Segment statistics: \n"
    # silence
    string += "\tSilence: \n"
    if self.sil_idx is None:
      string += "\t\tNo explicit silence in alignment! \n"
    else:
      string += "\t\tInitial: \n"
      string += f"\t\t\tMean length: {self.get_mean_segment_len_for_silence('init')} \n"
      string += f"\t\t\tNum segments: {self.accum_num_segments_silence['init']} \n"
      string += "\t\tIntermediate: \n"
      string += f"\t\t\tMean length: {self.get_mean_segment_len_for_silence('inter')} \n"
      string += f"\t\t\tNum segments: {self.accum_num_segments_silence['inter']} \n"
      string += "\t\tFinal: \n"
      string += f"\t\t\tMean length: {self.get_mean_segment_len_for_silence('final')} \n"
      string += f"\t\t\tNum segments: {self.accum_num_segments_silence['final']} \n"
      string += "\t\tTotal per sequence: \n"
      string += f"\t\t\tMean length: {self.get_mean_segment_len_for_silence('total')} \n"
      string += f"\t\t\tNum segments: {sum(self.accum_num_segments_silence.values())} \n"
    string += "\n"
    # non-silence
    string += "\tNon-silence: \n"
    string += f"\t\tMean length per segment: {self.get_mean_segment_len_for_non_silence()} \n"
    string += f"\t\tNum segments: {sum(self.accum_num_segments_per_label.values())} \n\n"
    string += "\n"
    string += "\tOverall maximum segment length: %d \n" % self.max_segment_len
    string += "\n"
    return string

  def plot_histograms(self):
    self.segment_len_count_statistics.plot_histograms()

  def write_label_dependent_means_to_file(self):
    self.segment_len_count_statistics.write_label_dependent_means_to_file()


class SegmentLenCountStatistics:
  """
   Gather statistics specifically about the distribution of segment lengths in total and per label.

   Attributes:
       segment_len_counters (Dict[int, Counter]): A dictionary mapping each label to a Counter object that tracks
                                                  the frequency of each segment length occurring for that label.
       vocab (Dict[int, str]): A dictionary mapping label indices to their string representations, facilitating
                               human-readable output and analysis.

   Methods:
       update(label, segment_len): Updates the statistics for a given label with a new segment length.
       counter_to_hist_data(counter): Converts a Counter object into lists suitable for plotting histograms (x, y),
                                      where x is segment length and y is its normalized frequency.
       counter_to_individual_len_list(counter): Generates a list of segment lengths, repeated by their count,
                                                suitable for statistical analysis.
       get_label_variances_and_means(): Computes the variance and mean of segment lengths for each label, returning
                                        a dictionary for each statistic.
       plot_histograms(): Generates and saves histograms of segment lengths for all labels together and for the top
                          labels with the highest variance in segment lengths, providing insights into the
                          distribution and variability of segment lengths across labels.
   """
  def __init__(self, vocab: Dict[int, str]):
    self.segment_len_counters = {}
    self.vocab = vocab

  def update(self, label, segment_len):
    if label not in self.segment_len_counters:
      self.segment_len_counters[label] = Counter()

    self.segment_len_counters[label][segment_len] += 1

  @staticmethod
  def counter_to_hist_data(counter: Counter, remove_outliers=False):
    if remove_outliers:
      counter = SegmentLenCountStatistics.remove_outliers_from_counter(counter)
    x = np.array(list(counter.keys()))
    y = np.array(list(counter.values()))
    y = y / np.sum(y)  # normalize
    return x, y

  @staticmethod
  def remove_outliers_from_counter(x: Counter, num_std_devs=20):
    individual_lens = np.array([len_ for len_, count in x.items() for _ in range(count)])

    # deviation of median and median of deviation
    d = np.abs(individual_lens - np.median(individual_lens))
    mdev = np.median(d)
    # calculate how many median deviations away from the median each value is
    s = d / mdev if mdev else 0.
    # only keep values that are less than 3 median deviations away from the median
    mask = s < num_std_devs
    individual_lens = individual_lens[mask]
    counter_wo_outliers = Counter(individual_lens)
    return counter_wo_outliers

  @staticmethod
  def counter_to_individual_len_list(counter: Counter):
    return [len_ for len_, count in counter.items() for _ in range(count)]

  def get_label_variances_and_means(self):
    variances = {}
    means = {}
    for label, counter in self.segment_len_counters.items():
      if len(counter) > 1:
        variances[label] = np.var(self.counter_to_individual_len_list(counter))
        means[label] = np.mean(self.counter_to_individual_len_list(counter))

    return variances, means

  @staticmethod
  def plot_histogram(counter, title, filename, remove_outliers=False):
    x, y = SegmentLenCountStatistics.counter_to_hist_data(counter, remove_outliers=remove_outliers)
    plt.bar(x, y)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

  def plot_histograms(self):
    dirname = "histograms"
    if not os.path.exists(dirname):
      os.makedirs(dirname)

    self.plot_histogram(
      sum(self.segment_len_counters.values(),
          Counter()),
      "Histogram of segment lengths for all labels.",
      os.path.join(dirname, "all_segments_histogram.png"),
      remove_outliers=False
    )
    self.plot_histogram(
      sum(self.segment_len_counters.values(),
          Counter()),
      "Histogram of segment lengths for all labels without outliers.",
      os.path.join(dirname, "all_segments_histogram_wo_outliers.png"),
      remove_outliers=True
    )

    label_variances, label_means = self.get_label_variances_and_means()
    labels_sorted_by_variance = sorted(label_variances, key=label_variances.get, reverse=True)
    for i, label in enumerate(labels_sorted_by_variance[:10], 1):
      title = f"Segment length histogram for '{self.vocab[label]}'\n"
      title += f"Mean/Variance: {label_means[label]:.2f}/{label_variances[label]:.2f}"
      self.plot_histogram(
        self.segment_len_counters[label],
        title,
        os.path.join(dirname, f"top-{i}-var_histogram.png"),
        remove_outliers=False
      )

  def write_label_dependent_means_to_file(self):
    _, means = self.get_label_variances_and_means()
    total_mean = np.mean(list(means.values()))
    with open("label_dependent_mean_lens", "w+") as f:
      f.write("{\n")
      for label in self.vocab:
        f.write(f"'{label}': {means[label] if label in means else total_mean},\n")
      f.write("}\n")


class RepetitionStatistics(ABC):
  """
   The RepetitionStatistics class serves as an abstract base class designed to track and analyze repetition statistics
   within sequences of data. This class is intended to be subclassed to implement specific types of repetition tracking,
   such as by characters or BPE units.

   Attributes:
       num_transitions (int): Counts the number of transitions between units in sequences.
       num_repetitions (int): Counts the total number of repeated units.
       num_seqs_w_repetitions (int): Counts the number of sequences that contain at least one repetition.
       num_words_w_repetitions (int): Counts the number of words with at least one repetition.
       num_word_transitions (int): Counts the number of times we transition into a new word (not counting the first).
       num_seqs (int): Counts the total number of sequences processed.

   Methods:
       update(sequence): Updates the repetition statistics based on the provided sequence.
       prev_word_ended(unit): Abstract method that determines if the previous word has ended, based on the current unit.
       name(): Abstract method that returns the name of the specific repetition statistics being tracked.
       get_percent_repetitions(): Calculates the percentage of transitions that are repetitions.
       get_percent_words_w_repetitions(): Calculates the percentage of words with repetitions.
       get_percent_seqs_w_repetitions(): Calculates the percentage of sequences that contain repetitions.
       __str__(): Returns a string representation of the current repetition statistics.
   """
  def __init__(self):
    self.num_transitions = 0
    self.num_repetitions = 0
    self.num_seqs_w_repetitions = 0
    self.num_words_w_repetitions = 0
    self.num_word_transitions = 0

    self.num_seqs = 0

  def update(self, sequence: List[str]):
    repetition_in_seq_flag = False
    repetition_in_word_flag = False
    self.num_seqs += 1

    for i, unit in enumerate(sequence):
      if i == 0:
        prev_unit = None
      else:
        prev_unit = sequence[i - 1]
        self.num_transitions += 1

      if unit == prev_unit:
        self.num_repetitions += 1

        if not repetition_in_seq_flag:
          self.num_seqs_w_repetitions += 1
          repetition_in_seq_flag = True

        if not repetition_in_word_flag:
          self.num_words_w_repetitions += 1
          repetition_in_word_flag = True

      if self.prev_word_ended(unit):
        repetition_in_word_flag = False
        if i > 0:
          self.num_word_transitions += 1

  @abstractmethod
  def prev_word_ended(self, unit: str):
    pass

  @abstractmethod
  def name(self):
    pass

  def get_percent_repetitions(self):
    return self.num_repetitions / self.num_transitions * 100

  def get_percent_words_w_repetitions(self):
    return self.num_words_w_repetitions / self.num_word_transitions * 100

  def get_percent_seqs_w_repetitions(self):
    return self.num_seqs_w_repetitions / self.num_seqs * 100

  def __str__(self):
    string = f"{self.name()} statistics: \n"
    string += f"\tPercent of repetitions: {self.get_percent_repetitions()} \n"
    string += f"\tPercent of words with repetitions: {self.get_percent_words_w_repetitions()} \n"
    string += f"\tPercent of seqs with repetitions: {self.get_percent_seqs_w_repetitions()} \n\n"
    return string


class BPERepetitionStatistics(RepetitionStatistics):
  """
    Specialization of RepetitionStatistics for BPE labels.
  """
  def __init__(self):
    super().__init__()

  def name(self):
    return "BPE repetition statistics"

  def prev_word_ended(self, unit: str):
    return not unit.endswith("@@")


class CharRepetitionStatistics(RepetitionStatistics):
  """
    Specialization of RepetitionStatistics for characters.
  """
  def __init__(self):
    super().__init__()

  def name(self):
    return "Character repetition statistics"

  def prev_word_ended(self, unit: str):
    return unit == " "


class SequenceStatistics:
  """
    Gather statistics on the sequence level.
  """
  def __init__(self):
    self.num_seqs = 0
    self.num_all_blank_seqs = 0
    self.total_len = 0
    self.max_len = 0

  def update(self, alignment, labels):
    if len(labels) == 0:
      self.num_all_blank_seqs += 1
    self.num_seqs += 1
    alignment_len = len(alignment)
    self.total_len += alignment_len
    if alignment_len > self.max_len:
      self.max_len = alignment_len

  def __str__(self):
    string = "Sequence statistics: \n"
    string += f"\tNum sequences: {self.num_seqs} \n"
    string += f"\tNum all-blank sequences: {self.num_all_blank_seqs} \n"
    string += f"\tMean length: {self.total_len / self.num_seqs} \n"
    string += f"\tMax length: {self.max_len} \n\n"
    return string


class WordStatistics:
  """
    Gather statistics on the word level.
  """
  def __init__(self):
    self.num_words = 0

  def update(self, labels: List[str]):
    self.num_words += len([label for label in labels if not label.endswith("@@")])

  def __str__(self):
    string = "Word statistics: \n"
    string += f"\tNum words: {self.num_words} \n\n"
    return string


class AlignmentStatistics:
  """
    Employ the statistics classes defined above to gather different statistics about the given dataset of alignments.
  """
  def __init__(self, blank_idx: int, sil_idx: Optional[int], vocab: Dict[int, str]):
    self.blank_idx = blank_idx
    self.sil_idx = sil_idx
    self.vocab = vocab

    self.segment_statistics = SegmentStatistics(sil_idx, vocab)
    self.label_repetition_statistics = BPERepetitionStatistics()
    self.char_repetition_statistics = CharRepetitionStatistics()
    self.sequence_statistics = SequenceStatistics()
    self.word_statistics = WordStatistics()

  def process_dataset(self):
    seq_idx = 0
    while dataset.is_less_than_num_seqs(seq_idx):
      self.process_sequence(seq_idx)
      seq_idx += 1

  def process_sequence(self, seq_idx):
    seq_tag = dataset.get_tag(seq_idx)
    alignment = dataset.get_data(seq_idx, "data")
    labels = alignment[alignment != self.blank_idx]
    label_positions = np.where(alignment != self.blank_idx)[0]

    self.sequence_statistics.update(alignment, labels)
    self.word_statistics.update([self.vocab[label] for label in labels])
    self.segment_statistics.update(labels, label_positions)

    labels_str = [self.vocab[label] for label in labels]
    self.label_repetition_statistics.update(labels_str)

    seq_str = " ".join([self.vocab[label] for label in labels]).replace("@@ ", "")
    self.char_repetition_statistics.update(list(seq_str))

  def write_statistics_to_file(self, filename: str):
    with open(filename, "w+") as f:
      f.write(str(self.segment_statistics))
      f.write(str(self.sequence_statistics))
      f.write(str(self.word_statistics))
      f.write(str(self.label_repetition_statistics))
      f.write(str(self.char_repetition_statistics))

    self.segment_statistics.plot_histograms()
    self.segment_statistics.write_label_dependent_means_to_file()


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
    alignment_statistics = AlignmentStatistics(args.blank_idx, args.sil_idx, vocab)
    alignment_statistics.process_dataset()
    alignment_statistics.write_statistics_to_file("statistics")
  except KeyboardInterrupt:
    print("KeyboardInterrupt")
    sys.exit(1)
  finally:
    rnn.finalize()


if __name__ == "__main__":
  main()
