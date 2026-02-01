from dataclasses import dataclass

from i6_core.returnn import PtCheckpoint
from sisyphus import Path


@dataclass(frozen=True)
class PretrainedConfig:
    """
    Encoder configuration base dataclass.

    Can contain default values.
    """

    pretrained_encoder: str = None
    pretrained_decoder: str = None

    pretrained_sllm: str = None

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        if self.pretrained_sllm is not None:
            assert self.pretrained_encoder is None, "If using pretrained SLLM, other encoder can't be loaded"
            assert self.pretrained_decoder is None, "If using pretrained SLLM, other decoder can't be loaded"


"""
Checkpoints
"""

_encoder_checkpoints = {
    "ctc_v1": "/u/marti.juanola/experiments/25_11_13_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.1MYTcWVoOsDz/output/models/epoch.500.pt"
    # More here
}

_decoder_checkpoints = {
    "llm_base_transcriptions": "/u/marti.juanola/experiments/25_11_10_llm/work/i6_core/returnn/training/ReturnnTrainingJob.CNLGypuo4I0A/output/models/epoch.100.pt",
    "llm_base_combined": "/u/marti.juanola/experiments/25_11_10_llm/work/i6_core/returnn/training/ReturnnTrainingJob.YRfhVjefAJao/output/models/epoch.100.pt",
    "llm_small_transcriptions": "/u/marti.juanola/experiments/25_11_10_llm/work/i6_core/returnn/training/ReturnnTrainingJob.xqzaOV0eAJSt/output/models/epoch.100.pt",
    "llm_small_combined": "/u/marti.juanola/experiments/25_11_10_llm/work/i6_core/returnn/training/ReturnnTrainingJob.erL8ScQicX6D/output/models/epoch.100.pt",
    # More here
}

_sllm_partial_trainings = {
    "SLLM_pretrained_ed_s_c_f2_oclr1": "/u/marti.juanola/experiments/25_10_17_sllm_d2/work/i6_core/returnn/training/ReturnnTrainingJob.pxoTwGri6FBD/output/models/epoch.010.pt"
}


# TODO: extract one main method
def get_encoder_checkpoint(pretrained_config: PretrainedConfig):
    model_name = pretrained_config.pretrained_encoder
    if model_name is None:
        raise ValueError("No encoder checkpoint specified.")
    if model_name not in _encoder_checkpoints:
        raise ValueError(f"Model '{model_name}' not found in encoder checkpoints.")
    return PtCheckpoint(Path(_encoder_checkpoints[model_name]))


def get_decoder_checkpoint(pretrained_config: PretrainedConfig):
    model_name = pretrained_config.pretrained_decoder
    if model_name is None:
        raise ValueError("No encoder checkpoint specified.")
    if model_name not in _decoder_checkpoints:
        raise ValueError(f"Model '{model_name}' not found in decoder checkpoints.")
    return PtCheckpoint(Path(_decoder_checkpoints[model_name]))


def get_sllm_checkpoint(pretrained_config: PretrainedConfig):
    model_name = pretrained_config.pretrained_sllm
    if model_name is None:
        raise ValueError("No encoder checkpoint specified.")
    if model_name not in _sllm_partial_trainings:
        raise ValueError(f"Model '{model_name}' not found in decoder checkpoints.")
    return PtCheckpoint(Path(_sllm_partial_trainings[model_name]))


"""
Specific configurations set below.
"""


def no_pretrained() -> PretrainedConfig:
    return PretrainedConfig()


def dec_base_transcriptions() -> PretrainedConfig:
    return PretrainedConfig(pretrained_decoder="llm_base_transcriptions")


def dec_base_combined() -> PretrainedConfig:
    return PretrainedConfig(pretrained_decoder="llm_base_combined")


def dec_small_combined() -> PretrainedConfig:
    return PretrainedConfig(pretrained_decoder="llm_small_combined")


def enc_dec_base_combined() -> PretrainedConfig:
    return PretrainedConfig(pretrained_encoder="ctc_v1", pretrained_decoder="llm_base_combined")


def enc_dec_small_combined() -> PretrainedConfig:
    return PretrainedConfig(pretrained_encoder="ctc_v1", pretrained_decoder="llm_small_combined")


def enc_dec_base_transcriptions() -> PretrainedConfig:
    return PretrainedConfig(
        pretrained_encoder="ctc_v1",
        pretrained_decoder="llm_base_transcriptions",
    )


"""
SLLM
"""


def load_SLLM_pretrained_ed_s_c_f2_oclr1() -> PretrainedConfig:
    return PretrainedConfig(
        pretrained_sllm="SLLM_pretrained_ed_s_c_f2_oclr1",
    )
