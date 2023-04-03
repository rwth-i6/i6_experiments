def speed_pert(audio, sample_rate, random_state):
    # speed perturbation
    import librosa
    from scipy import signal

    # Speed perturbation
    SPEED_PERTURBATION_PROB = 0.6
    SPEED_PERTURBATION_MIN_SCALE = 0.88
    SPEED_PERTURBATION_MAX_SCALE = 1.12

    # temporal perturbation
    TEMPORAL_PERTURBATION_PROB = 0.6
    TEMPORAL_PERTURBATION_MIN_SCALE = 0.83
    TEMPORAL_PERTURBATION_MAX_SCALE = 1.17

    # pre-emphasis
    PRE_EMPHASIS_PROB = 0.9
    PRE_EMPHASIS_MIN = 0.9
    PRE_EMPHASIS_MAX = 1.0

    do_speed_perturbation = random_state.random() < SPEED_PERTURBATION_PROB
    if do_speed_perturbation:
        new_sample_rate = int(
            sample_rate * random_state.uniform(SPEED_PERTURBATION_MIN_SCALE, SPEED_PERTURBATION_MAX_SCALE)
        )
        audio = librosa.core.resample(audio, sample_rate, new_sample_rate, res_type="kaiser_fast")

    do_temporal_perturbation = random_state.random() < TEMPORAL_PERTURBATION_PROB
    if do_temporal_perturbation:
        audio = librosa.effects.time_stretch(
            audio, random_state.uniform(TEMPORAL_PERTURBATION_MIN_SCALE, TEMPORAL_PERTURBATION_MAX_SCALE)
        )

    do_preemphasis = random_state.random() < PRE_EMPHASIS_PROB
    if do_preemphasis:
        pre_emphasis_value = random_state.uniform(PRE_EMPHASIS_MIN, PRE_EMPHASIS_MAX)
        audio = signal.lfilter([1, -pre_emphasis_value], [1], audio)
    return audio
