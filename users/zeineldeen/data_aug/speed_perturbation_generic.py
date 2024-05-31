speed_pert = """
def speed_pert(audio, sample_rate, random_state, min_factor={min_factor}, max_factor={max_factor}, step={step}):
  import librosa
  new_sample_rate = int(sample_rate * (1 + random_state.randint(min_factor, max_factor) * step))
  if new_sample_rate != sample_rate:
    audio = librosa.core.resample(audio, sample_rate, new_sample_rate, res_type="kaiser_fast")
  return audio
"""

speed_pert_v2 = """
import numpy
def speed_pert(audio, sample_rate={sample_rate}, min_factor={min_factor}, max_factor={max_factor}, step={step}, random_state=numpy.random.RandomState(1)):
  import librosa
 
  new_sample_rate = int(sample_rate * (1 + random_state.randint(min_factor, max_factor) * step))
  if new_sample_rate != sample_rate:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast", axis=0)
  return audio
"""
