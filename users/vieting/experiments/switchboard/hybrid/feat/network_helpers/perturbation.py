class PerturbationFactor:
  """
  Class to wrap perturbation factors, e.g. for speed or tempo perturbation.
  """
  def __init__(self, prob, minimum, maximum):
    self.prob = prob
    self.min = minimum
    self.max = maximum


class WaveformPerturbation:
  """
  Helper class to perform perturbation techniques on audio waveforms.
  """
  def __init__(self, speed=None, tempo=None, sox_effects=None, codecs=None, preemphasis=None):
    """
    :param speed:
    """
    import torch
    import functools
    from functools import partial
    self._speed = PerturbationFactor(**speed) if speed else None
    self._tempo = PerturbationFactor(**tempo) if tempo else None
    self._perturbations = [functools.partial(self.sox, sox_effects=sox_effects)]
    if preemphasis:
      self._perturbations.append(functools.partial(self.preemphasis, factor=PerturbationFactor(**preemphasis)))
    if codecs:
      self._perturbations.append(functools.partial(self.apply_codecs, codecs=codecs))

  def run(self, audio, sample_rate, random_state):
    # input_shape = audio.shape
    audio = torch.from_numpy(audio).unsqueeze(0).to(torch.float32)
    for perturbation in self._perturbations:
      audio = perturbation(audio, sample_rate, random_state)
    audio = audio.numpy().squeeze()
    # output_shape = audio.shape
    # print("change shape from {} to {}".format(input_shape, output_shape))
    assert isinstance(audio, np.ndarray)
    assert len(audio.shape) == 1
    return audio

  def sox(self, audio, sample_rate, random_state, sox_effects):
    import torchaudio
    sox_effects = sox_effects or []
    speed = False
    if self._speed is not None:
      if random_state.random() < self._speed.prob:
        factor = random_state.random() * (self._speed.max - self._speed.min) + self._speed.min
        sox_effects.append(["speed", str(factor)])
        sox_effects.append(["rate", str(sample_rate)])
        speed = True
    if self._tempo is not None:
      if random_state.random() < self._tempo.prob and not speed:
        factor = random_state.random() * (self._tempo.max - self._tempo.min) + self._tempo.min
        sox_effects.append(["tempo", str(factor)])
    audio, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sample_rate, sox_effects)
    return audio

  @staticmethod
  def preemphasis(audio, sample_rate, random_state, factor):
    import torch
    if random_state.random() < factor.prob:
      preemphasis_coefficient = random_state.random() * (factor.max - factor.min) + factor.min
      # audio[i,j] -= preemphasis_coefficient * audio[i, max(0, j-1)] for all i,j
      offset_audio = torch.nn.functional.pad(audio.unsqueeze(0), (1, 0), mode="replicate").squeeze(
          0
      )  # size (m, window_size + 1)
      audio = audio - preemphasis_coefficient * offset_audio[:, :-1]
    return audio

  @staticmethod
  def apply_codecs(audio, sample_rate, random_state, codecs):
    import torchaudio
    for codec in codecs:
      prob = codec.pop("prob", 1.0)
      if random_state.random() < prob:
        audio = torchaudio.functional.apply_codec(audio, sample_rate, **codec)
    return audio


def get_classes_perturbation():
    classes = []
    for cls_name, cls in list(globals().items()):
        if isinstance(cls, type):
            classes.append(cls)
    return classes