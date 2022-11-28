def speed_pert(audio, sample_rate, random_state):
  import librosa
  new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
  if new_sample_rate != sample_rate:
    audio = librosa.core.resample(audio, sample_rate, new_sample_rate, res_type="kaiser_fast")
  return audio
