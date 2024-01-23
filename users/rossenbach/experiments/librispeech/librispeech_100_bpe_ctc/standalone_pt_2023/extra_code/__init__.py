def speed_perturbation(audio, sample_rate, random_state):
  import librosa
  new_sample_rate = int(sample_rate * (1 + random_state.randint(-1, 2) * 0.1))
  if new_sample_rate != sample_rate:
    audio = librosa.core.resample(audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast")
  return audio