from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union


class Config:
    def __init__(self, **kwargs):
        pass

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

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


@dataclass()
class ConvDurationSigmaPredictorConfig(Config):
    hidden_size: int
    filter_size: int
    dropout: float
    

@dataclass()
class NarEncoderConfig(Config):
    label_in_dim: int
    embedding_size: int
    conv_hidden_size: int
    filter_size: int
    dropout: float
    lstm_size: int


@dataclass()
class NarTacoDecoderConfig(Config):
    lstm_size: int
    dropout: float


@dataclass
class ModelConfig:
    speaker_embedding_size: int
    dropout: float
    encoder_config: NarEncoderConfig
    decoder_config: NarTacoDecoderConfig
    duration_predictor_config: ConvDurationSigmaPredictorConfig
    feature_extraction_config: DbMelFeatureExtractionConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["encoder_config"] = NarEncoderConfig.from_dict(d["encoder_config"])
        d["decoder_config"] = NarTacoDecoderConfig.from_dict(d["decoder_config"])
        d["duration_predictor_config"] = ConvDurationSigmaPredictorConfig.from_dict(d["duration_predictor_config"])
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig.from_dict(d["feature_extraction_config"])
        return ModelConfig(**d)

