from __future__ import annotations
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    import numpy as np


def speed_pert_librosa_config(audio: np.ndarray, sample_rate: int, random_state: np.random.RandomState) -> np.ndarray:
    """
    Speed perturbation, configurable via RETURNN config.

    Options ``speed_pert_prob`` and ``speed_pert_discrete_values`` in the config.
    See code below.

    :param audio: shape (audio_len,)
    :param sample_rate: e.g. 16_000
    :param random_state:
    :return: audio
    """
    import librosa
    from returnn.config import get_global_config

    config = get_global_config()

    prob = config.float("speed_pert_prob", 1)
    assert prob is not None and 0 <= prob <= 1, f"set speed_pert_prob in [0,1] in config, got {prob}"
    if random_state.uniform(0, 1) >= prob:
        return audio

    # Mapping factor -> prob.
    # Factor is e.g. 0.9, 1.0, 1.1 or so.
    # Prob will be renormalized, so does not need to sum to 1.
    discrete_values: Dict[float, float] = config.typed_value("speed_pert_discrete_values")
    if isinstance(discrete_values, (tuple, list)):
        discrete_values = {k: 1 for k in discrete_values}  # evenly distributed, all the same prob
    assert (
        isinstance(discrete_values, dict)
        and discrete_values
        and all(
            isinstance(k, (int, float)) and isinstance(v, (int, float)) and v > 0 for (k, v) in discrete_values.items()
        )
    ), f"speed_pert_discrete_values invalid in config, got {discrete_values}"
    prob_sum = sum(v for v in discrete_values.values())
    assert prob_sum > 0
    discrete_values_ = [(k, v / prob_sum) for k, v in discrete_values.items()]
    i = random_state.choice(len(discrete_values_), p=[v for k, v in discrete_values_])

    # Note: We multiply by the sample rate.
    # E.g. a factor 2 means, we get twice as many samples per second,
    # but then we use the audio as if it would be with the original sample rate,
    # meaning that one second before corresponds to two seconds now.
    # I.e. factor 2 makes it twice as slow, or the audio twice as long.
    # Vice versa, factor 0.5 makes it twice as fast, or the audio half the length.
    new_sample_rate = int(sample_rate * discrete_values_[i][0])
    if new_sample_rate != sample_rate:
        audio = librosa.core.resample(audio, orig_sr=sample_rate, target_sr=new_sample_rate, res_type="kaiser_fast")
    return audio
