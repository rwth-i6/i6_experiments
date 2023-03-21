from sisyphus import *
import h5py
from i6_core.lib import corpus
import numpy as np
from typing import Optional


class CalculateVarianceFromFeaturesJob(Job):

  __sis_hash_exclude__ = {"segments": None, "mem": 16}
  def __init__(self, feature_hdf: tk.Path, duration_hdf: tk.Path, bliss: tk.Path, segments: Optional[tk.Path] = None, mem=16):

    self.features = feature_hdf
    self.durations = duration_hdf
    self.corpus = bliss
    self.segments = segments
    self.rqmt = {"time": 4, "mem": mem}

    self.out_csv = self.output_path("stats.csv")
    self.out_variance = self.output_var("variance")
    self.out_weight_variance = self.output_var("weighted_variance")
    self.out_variance_no_sil = self.output_var("variance_no_sil")
    self.out_weight_variance_no_sil = self.output_var("weighted_variance_no_sil")

  def tasks(self):
    yield Task("run", rqmt=self.rqmt)

  def run(self):

    duration_data = h5py.File(self.durations.get_path(), "r")
    duration_inputs = duration_data["inputs"]
    duration_raw_tags = duration_data["seqTags"]
    duration_lengths = duration_data["seqLengths"]

    durations_by_tag = {}
    offset = 0
    for tag, length in zip(duration_raw_tags, duration_lengths):
      durations_by_tag[tag] = duration_inputs[offset:offset + length[0]]
      offset += length[0]
    print("Durations loaded ", len(durations_by_tag))

    feature_data = h5py.File(self.features.get_path(), "r")
    feature_inputs = feature_data["inputs"]
    feature_raw_tags = list(feature_data["seqTags"])
    feature_lengths = feature_data["seqLengths"]
    features_by_tag = {}
    offset = 0
    for tag, length in zip(feature_raw_tags, feature_lengths):
      features_by_tag[tag if isinstance(tag, str) else str(tag, "utf-8")] = feature_inputs[offset:offset + length[0]]
      offset += length[0]

    print("Features loaded ", len(features_by_tag))

    bliss = corpus.Corpus()
    bliss.load(self.corpus.get_path())
    counter = 0
    phoneme_dict = {}
    if self.segments is not None:
        with open(self.segments.get_path(), "rt") as f:
            segment_list = f.read().splitlines()
    else:
        segment_list = None
    for recording in bliss.all_recordings():
      for segment in recording.segments:
        if segment_list is not None and segment not in segment_list and segment.fullname() not in segment_list:
            continue
        text = segment.orth.split(" ")
        durations = durations_by_tag[segment.fullname()]
        features = features_by_tag[segment.fullname()]
        counter += 1
        print(counter)
        assert np.sum(durations) == len(features), (sum(durations), len(features), segment.fullname())
        assert len(text) == len(durations), (len(text), len(durations), segment.fullname(), text)
        offset = 0
        for duration, token in zip(durations, text):
          if token not in phoneme_dict:
            phoneme_dict[token] = []
          if int(duration) > 0:
            for feature in features[offset:int(duration) + offset]:
              phoneme_dict[token].append(feature)
            offset += int(duration)
        # print(list(zip(text, durations)))

    mean_ls = []
    counts = []
    no_sil_means = []
    no_sil_counts = []
    with open(self.out_csv.get_path(), "wt") as f:
      f.write("Phoneme, mean, median, min, max, std\n")
      for token in phoneme_dict:
        if token.startswith("[start") or token.startswith("[end"):
          continue
        covs = np.std(phoneme_dict[token], axis=0)
        print(token, covs)
        print("Mean:", np.mean(covs), "Median:", np.median(covs), "Min:", covs.min(), "Max:", covs.max(), "Stds:",
          np.std(covs))
        string = "%s,%.3f,%.3f,%.3f,%.3f,%.3f\n" % (
          token,
          float(np.mean(covs)),
          float(np.median(covs)),
          float(np.min(covs)),
          float(np.max(covs)),
          float(np.std(covs))
        )
        f.write(string)
        mean_ls.append(np.mean(covs))
        counts.append(len(phoneme_dict[token]))
        if not token.startswith("["):
          no_sil_means.append(np.mean(covs))
          no_sil_counts.append(len(phoneme_dict[token]))
    assert len(mean_ls) == len(no_sil_means) + 1, (len(mean_ls), len(no_sil_means))
    assert len(counts) == len(no_sil_counts) + 1, (len(counts), len(no_sil_counts))
    self.out_variance.set(np.mean(mean_ls))
    self.out_weight_variance.set(np.average(mean_ls, weights=counts))
    self.out_variance_no_sil.set(np.mean(no_sil_means))
    self.out_weight_variance_no_sil.set(np.average(no_sil_means, weights=no_sil_counts))


class CalculateVarianceFromDurations(Job):

  __sis_hash_exclude__ = {"segments": None}
  def __init__(self, duration_hdf: tk.Path, bliss: tk.Path, segments: Optional[tk.Path] = None):

    self.durations = duration_hdf
    self.bliss = bliss
    self.segments = segments

    self.out_variance = self.output_var("variance")
    self.out_weight_variance = self.output_var("weighted_variance")
    self.out_variance_no_sil = self.output_var("variance_no_sil")
    self.out_weight_variance_no_sil = self.output_var("weighted_variance_no_sil")
    self.out_csv = self.output_path("stats.csv")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):

    c = corpus.Corpus()
    c.load(self.bliss.get_path())

    input_data = h5py.File(self.durations.get_path(), 'r')

    inputs = input_data['inputs']
    seq_tags = input_data['seqTags']
    lengths = input_data['seqLengths']

    sequences = []
    tags = []
    offset = 0
    for tag, length in zip(seq_tags, lengths):
      tag = tag if isinstance(tag, str) else tag.decode()
      in_data = inputs[offset:offset + length[0]]
      sequences.append(in_data)
      offset += length[0]
      tags.append(tag)

    tagged_sequences = {tag: sequence for tag, sequence in zip(tags, sequences)}
    tagged_annotated_sequences = {}

    if self.segments is not None:
        with open(self.segments.get_path(), "rt") as f:
            segment_list = f.read().splitlines()
    else:
        segment_list = None

    for segment in c.segments():
      if segment_list is not None and segment not in segment_list and segment.fullname() not in segment_list:
          continue
      name = segment.fullname()
      tagged_annotated_sequences[name] = np.squeeze(tagged_sequences[name]), segment.orth.split(" ")
      assert len(tagged_annotated_sequences[name][0]) == len(tagged_annotated_sequences[name][1])

    statistics = {}
    print(tagged_annotated_sequences)
    for key, (seq, orth) in tagged_annotated_sequences.items():
      for dur, tok in zip(seq, orth):
        if tok not in statistics:
          statistics[tok] = []
        statistics[tok].append(dur)

    std_ls = []
    no_sil_std = []
    counts = []
    no_sil_counts = []
    with open(self.out_csv.get_path(), "wt") as f:
      f.write("Phoneme, count, mean, median, max, min, std\n")
      for token in statistics:
        if token.startswith("[start") or token.startswith("[end"):
          continue
        string = "%s,%d,%.3f,%d,%d,%d,%.3f\n" % (
          token,
          len(statistics[token]),
          float(np.mean(statistics[token])),
          int(np.median(statistics[token])),
          int(np.max(statistics[token])),
          int(np.min(statistics[token])),
          float(np.std(statistics[token]))
        )
        f.write(string)
        std_ls.append(np.std(statistics[token]))
        counts.append(len(statistics[token]))
        if not token.startswith("["):
          no_sil_std.append(np.std(statistics[token]))
          no_sil_counts.append(len(statistics[token]))

    assert len(std_ls) == len(no_sil_std) + 1, (len(std_ls), len(no_sil_std))
    assert len(counts) == len(no_sil_counts) + 1, (len(counts), len(no_sil_counts))
    self.out_variance.set(np.mean(std_ls))
    self.out_weight_variance.set(np.average(std_ls, weights=counts))
    self.out_variance_no_sil.set(np.mean(no_sil_std))
    self.out_weight_variance_no_sil.set(np.average(no_sil_std, weights=no_sil_counts))


class CalculateKLDivFromDurations(Job):

  def __init__(self, pred_durations: tk.Path, ref_durations: tk.Path, bliss: tk.Path):

    self.pred_durations = pred_durations
    self.ref_durations = ref_durations
    self.bliss = bliss

    self.out_div = self.output_var("avr_kl_div")
    self.out_stats = self.output_path("kl_divs")

  def get_hdf(self, hdf: tk.Path):

    input_data = h5py.File(hdf.get_path(), 'r')

    inputs = input_data['inputs']
    seq_tags = input_data['seqTags']
    lengths = input_data['seqLengths']

    sequences = []
    tags = []
    offset = 0
    for tag, length in zip(seq_tags, lengths):
      tag = tag if isinstance(tag, str) else tag.decode()
      in_data = inputs[offset:offset + length[0]]
      sequences.append(in_data)
      offset += length[0]
      tags.append(tag)

    return sequences, tags

  def create_annotated_sequences(self, sequences, tags, corpus):
    tagged_sequences = {tag: sequence for tag, sequence in zip(tags, sequences)}
    tagged_annotated_sequences = {}
    for segment in corpus.segments():
      name = segment.fullname()
      tagged_annotated_sequences[name] = np.squeeze(tagged_sequences[name]), segment.orth.split(" ")
      assert len(tagged_annotated_sequences[name][0]) == len(tagged_annotated_sequences[name][1])
    return tagged_annotated_sequences

  def create_statistics(self, annotated_sequences):
    statistics = {}
    for key, (seq, orth) in annotated_sequences.items():
      for dur, tok in zip(seq, orth):
        if tok not in statistics:
          statistics[tok] = []
        statistics[tok].append(dur)
    return statistics

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):

    sequences_1, tags_1 = self.get_hdf(self.pred_durations)
    sequences_2, tags_2 = self.get_hdf(self.ref_durations)
    c = corpus.Corpus()
    c.load(self.bliss.get_path())

    annotated_sequences_1 = self.create_annotated_sequences(sequences_1, tags_1, c)
    annotated_sequences_2 = self.create_annotated_sequences(sequences_2, tags_2, c)

    statistics_1 = self.create_statistics(annotated_sequences_1)
    statistics_2 = self.create_statistics(annotated_sequences_2)

    bins = np.arange(-0.5, 60.5, 1)

    distr_1 = {}
    distr_2 = {}
    lens = {}

    for x in statistics_1:
      if len(statistics_1[x]) == 0:
        continue
      lens[x] = len(statistics_1[x])
      hist = np.histogram(statistics_1[x], bins=bins)
      distr = [y / len(statistics_1[x]) for y in hist[0]]
      assert sum(distr) > 0.97, (x, sum(distr))
      distr_1[x] = distr
    for x in statistics_2:
      if len(statistics_2[x]) == 0:
        continue
      hist = np.histogram(statistics_2[x], bins=bins)
      distr = [y / len(statistics_2[x]) for y in hist[0]]
      assert sum(distr) > 0.97, (x, sum(distr))
      distr_2[x] = distr
    divs = {}
    for key in distr_1:
      q = np.array(distr_2[key])
      p = np.array(distr_1[key])
      count = 0
      diff_mass_p = 0
      diff_mass_q = 0
      zero_c = 0
      for a, b in zip(p, q):
        if a != 0 and b == 0 or a == 0 and b != 0:
          diff_mass_p += a
          diff_mass_q += b
          count += 1
        elif a == 0 and b == 0:
          zero_c += 1
      kl_div = np.sum(np.where(p != 0, np.where(q != 0, p * np.log(p / q), 0), 0))
      divs[key] = kl_div
      print(key, kl_div, count, "ungleich,", zero_c, "null von", len(p))
      # print(diff_mass_p, diff_mass_q)
    div_ls = []
    len_ls = []
    with open(self.out_stats.get_path(), "wt") as f:
      f.write("Phoneme   KL_div  Count")
      for key in divs:
        f.write(f"{key}: {str(divs[key])}  {str(lens[key])}")
    for x in divs:
      if not x.startswith("["):
        div_ls.append(divs[x])
        len_ls.append(lens[x])
    avrg = np.average(div_ls, weights=len_ls)
    print(np.average(div_ls, weights=len_ls))
    self.out_div.set(avrg)

