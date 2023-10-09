"""
RETURNN Dataset compatible processing code snippets
"""

def legacy_speed_perturbation(audio, sample_rate, random_state):
  """
  Use with the old TF setups Rossenbach/Zeineldeen

  :param audio:
  :param sample_rate:
  :param random_state:
  :return:
  """
  import librosa
  new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
  if new_sample_rate != sample_rate:
    audio = librosa.core.resample(audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast")
  return audio