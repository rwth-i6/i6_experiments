def speed_pert(audio, sample_rate, random_state):
    import librosa

    new_sample_rate = int(sample_rate * random_state.uniform(0.88, 1.12))
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(audio, sample_rate, new_sample_rate, res_type="kaiser_fast")
    return audio
