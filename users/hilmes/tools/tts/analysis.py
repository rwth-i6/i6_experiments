from sisyphus import *
import h5py
from i6_core.lib import corpus
import numpy as np


class CalculateVarianceFromFeaturesJob(Job):

  def __init__(self, feature_hdf: tk.Path, duration_hdf: tk.Path, bliss: tk.Path):

    self.features = feature_hdf
    self.durations = duration_hdf
    self.corpus = bliss

    self.out_csv = self.output_path("stats.csv")
    self.out_variance = self.output_var("variance")

  def tasks(self):
    yield Task("run", rqmt={"mem": 16, "time": 2})

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

    phoneme_dict = {}
    counter = 0
    for recording in bliss.all_recordings():
      for segment in recording.segments:
        text = segment.orth.split(" ")
        print(counter)
        counter += 1
        durations = durations_by_tag[segment.fullname()]
        features = features_by_tag[segment.fullname()]
        if np.sum(durations) != len(features):
          print(sum(durations), len(features), segment.fullname())
        # assert np.sum(durations) == len(features), (sum(durations), len(features), segment.fullname())
        assert len(text) == len(durations), (len(text), len(durations), segment.fullname())
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
    self.out_variance.set(np.mean(mean_ls))
