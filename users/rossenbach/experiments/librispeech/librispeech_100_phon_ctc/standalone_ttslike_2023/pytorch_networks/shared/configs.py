from dataclasses import dataclass

from typing import Optional, Tuple


@dataclass
class DbMelFeatureExtractionConfig():
    """

    :param sample_rate: audio sample rate in Hz
    :param win_size: window size in seconds
    :param hop_size: window shift in seconds
    :param f_min: minimum mel filter frequency in Hz
    :param f_max: maximum mel fitler frequency in Hz
    :param min_amp: minimum amplitude for safe log
    :param num_filters: number of mel windows
    :param center: centered STFT with automatic padding
    :param norm: tuple optional of mean & std_dev for feature normalization
    """
    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool
    norm: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, d):
        return DbMelFeatureExtractionConfig(**d)