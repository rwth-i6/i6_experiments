from sisyphus import *
import soundfile
import librosa
import numpy

from i6_core.lib import corpus


class CompareF0ValuesJob(Job):
  """
  Extracts F0 for two given Corpora and calculates MAE over both
  """

  def __init__(self, ref_corpus: tk.Path, test_corpus: tk.Path, segment_list: tk.Path):

    self.ref_corpus = ref_corpus
    self.test_corpus = test_corpus
    self.segment_list = segment_list

    self.rqmt = {
      "mem": 1,
      "time": 4,
    }

    # default parameter for DTW
    self.step_len = 0.0125
    self.window_len = 0.05
    self.fmin_pyin = 60
    self.fmin_ex = 60
    self.fmax_ex = 7600
    self.fmax_pyin = 1000
    self.center = True
    self.n_mels = 80

    self.out_maes = self.output_path("maes")
    self.out_ref_means = self.output_path("ref_means")
    self.out_ref_stds = self.output_path("ref_stds")
    self.out_test_means = self.output_path("test_means")
    self.out_test_stds = self.output_path("test_stds")

  def tasks(self):
    # TODO: Caching and then with rqmt
    yield Task("run", mini_task=True)

  def run(self):

    mae_ls = []
    mean_1_ls = []
    std_1_ls = []
    mean_2_ls = []
    std_2_ls = []

    with open(self.segment_list.get_path(), "r") as f:
      segment_list = f.read().splitlines()

    bliss_1 = corpus.Corpus()
    bliss_1.load(self.ref_corpus.get_path())
    bliss_2 = corpus.Corpus()
    bliss_2.load(self.test_corpus.get_path())

    recordings_1 = {}
    for recording in bliss_1.all_recordings():
      for segment in recording.segments:
        if segment.fullname() in segment_list:
          recordings_1[segment.fullname()] = recording.audio

    recordings_2 = {}
    for recording in bliss_2.all_recordings():
      for segment in recording.segments:
        if segment.fullname() in segment_list:
          recordings_2[segment.fullname()] = recording.audio

    for segment in segment_list:
      signal_1, sr_1 = soundfile.read(recordings_1[segment])
      signal_2, sr_2 = soundfile.read(recordings_2[segment])

      assert sr_1 == sr_2, "Sample rates must match"
      sample_rate = sr_1

      mel_filterbank_1 = librosa.feature.melspectrogram(
        y=signal_1, sr=sample_rate,
        n_mels=self.n_mels,
        hop_length=int(self.step_len * sample_rate),
        n_fft=int(self.window_len * sample_rate),
        fmin=self.fmin_ex, fmax=self.fmax_ex, center=self.center
      )
      mel_filterbank_2 = librosa.feature.melspectrogram(
        y=signal_2, sr=sample_rate,
        n_mels=self.n_mels,
        hop_length=int(self.step_len * sample_rate),
        n_fft=int(self.window_len * sample_rate),
        fmin=self.fmin_ex, fmax=self.fmax_ex, center=self.center
      )

      D, wp = librosa.sequence.dtw(mel_filterbank_1, mel_filterbank_2)

      f0_1, voiced_1, _ = librosa.pyin(y=signal_1, sr=sample_rate, hop_length=int(self.step_len * sample_rate),
        frame_length=int(self.window_len * sample_rate), win_length=int(self.window_len * sample_rate) // 2,
        fmin=self.fmin_pyin, fmax=self.fmax_pyin, center=self.center, fill_na=0.0)
      f0_2, voiced_2, _ = librosa.pyin(y=signal_2, sr=sample_rate, hop_length=int(self.step_len * sample_rate),
        frame_length=int(self.window_len * sample_rate), win_length=int(self.window_len * sample_rate) // 2,
        fmin=self.fmin_pyin, fmax=self.fmax_pyin, center=self.center, fill_na=0.0)

      scale = 1200 / len(wp)
      sum = 0
      pitch_ls_1 = []
      pitch_ls_2 = []
      for t_1, t_2 in wp:
        if voiced_1[t_1] and voiced_2[t_2]:
          pitch_ls_1.append(f0_1[t_1])
          pitch_ls_2.append(f0_2[t_2])
          sum += abs(numpy.log2((f0_2[t_2] / f0_1[t_1])))

      mae = scale * sum
      print("MAE     Real Data        Synth Data ")
      print("%.2f " % mae, "%.2f+-%.2f" % (float(numpy.mean(pitch_ls_1)), float(numpy.std(pitch_ls_1))),
        "   %.2f+-%.2f " % (float(numpy.mean(pitch_ls_2)), float(numpy.std(pitch_ls_2))))
      mae_ls.append(mae)
      mean_1_ls.append(numpy.mean(pitch_ls_1))
      mean_2_ls.append(numpy.mean(pitch_ls_2))
      std_1_ls.append(numpy.std(pitch_ls_1))
      std_2_ls.append(numpy.std(pitch_ls_2))

    with open(self.out_maes.get_path(), "w") as f:
      for mae in mae_ls:
        f.write("%s\n" % mae)
    with open(self.out_ref_means.get_path(), "w") as f:
      for mean in mean_1_ls:
        f.write("%s\n" % mean)
    with open(self.out_test_means.get_path(), "w") as f:
      for mean in mean_2_ls:
        f.write("%s\n" % mean)
    with open(self.out_ref_stds.get_path(), "w") as f:
      for std in std_1_ls:
        f.write("%s\n" % std)
    with open(self.out_test_stds.get_path(), "w") as f:
      for std in std_2_ls:
        f.write("%s\n" % std)


class CompareEnergyValuesJob(Job):

  def __init__(self, ref_corpus: tk.Path, test_corpus: tk.Path, segment_list: tk.Path):
    self.ref_corpus = ref_corpus
    self.test_corpus = test_corpus
    self.segment_list = segment_list

    # default parameter for Extraction
    self.step_len = 0.0125
    self.window_len = 0.05
    self.fmin_pyin = 60
    self.fmin_ex = 60
    self.fmax_ex = 7600
    self.fmax_pyin = 1000
    self.center = True
    self.n_mels = 80

    self.out_maes = self.output_path("maes")

  def tasks(self):
    # TODO: Caching and then with rqmt
    yield Task("run", mini_task=True)

  def run(self):
    maes = []

    with open(self.segment_list.get_path(), "r") as f:
      segment_list = f.read().splitlines()

    bliss_1 = corpus.Corpus()
    bliss_1.load(self.ref_corpus.get_path())
    bliss_2 = corpus.Corpus()
    bliss_2.load(self.test_corpus.get_path())

    recordings_1 = {}
    for recording in bliss_1.all_recordings():
      for segment in recording.segments:
        if segment.fullname() in segment_list:
          recordings_1[segment.fullname()] = recording.audio

    recordings_2 = {}
    for recording in bliss_2.all_recordings():
      for segment in recording.segments:
        if segment.fullname() in segment_list:
          recordings_2[segment.fullname()] = recording.audio

    for segment in segment_list:
      signal_1, sr_1 = soundfile.read(recordings_1[segment])
      signal_2, sr_2 = soundfile.read(recordings_2[segment])

      assert sr_1 == sr_2, "Sample rates must match"
      sample_rate = sr_1

      mel_filterbank_1 = librosa.feature.melspectrogram(
        y=signal_1, sr=sample_rate,
        n_mels=self.n_mels,
        hop_length=int(self.step_len * sample_rate),
        n_fft=int(self.window_len * sample_rate),
        fmin=self.fmin_ex, fmax=self.fmax_ex, center=self.center
      )
      mel_filterbank_2 = librosa.feature.melspectrogram(
        y=signal_2, sr=sample_rate,
        n_mels=self.n_mels,
        hop_length=int(self.step_len * sample_rate),
        n_fft=int(self.window_len * sample_rate),
        fmin=self.fmin_ex, fmax=self.fmax_ex, center=self.center
      )

      D, wp = librosa.sequence.dtw(mel_filterbank_1, mel_filterbank_2)

      energy_1 = librosa.feature.rms(
        y=signal_1,
        hop_length=int(self.step_len * sample_rate), frame_length=int(self.window_len * sample_rate))
      energy_2 = librosa.feature.rms(
        y=signal_2,
        hop_length=int(self.step_len * sample_rate), frame_length=int(self.window_len * sample_rate))

      scale = 1 / len(wp)
      sum = 0
      energy_ls_1 = []
      energy_ls_2 = []
      for t_1, t_2 in wp:
        energy_ls_1.append(energy_1[t_1])
        energy_ls_2.append(energy_2[t_2])
        sum += abs(energy_1[t_1] - energy_2[t_2])
      sum = sum * scale
      maes.append(sum)

    with open(self.out_maes.get_path(), "w") as f:
      for mae in maes:
        f.write("%s\n" % mae)
