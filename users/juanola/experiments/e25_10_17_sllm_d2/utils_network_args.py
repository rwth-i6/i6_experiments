from dataclasses import asdict
from typing import Any

from .configurations.data.label_config import LabelConfig
from .configurations.network.network_config import NetworkConfig
from ...sisyphus_jobs.configs.qwen2_decoder_config_job_v2 import Qwen2DecoderConfigJobV2


def get_network_args(n_config: NetworkConfig, l_config: LabelConfig) -> dict[str, Any]:
    """
    Builds network arguments and alias for the model.

    :param config:
    :return:
    """
    label_config = asdict(l_config)
    fe_config = asdict(n_config.feature_extraction)
    encoder_config = asdict(n_config.encoder)
    if "aux_loss_scales" in encoder_config:
        encoder_config.pop("aux_loss_scales")
    assert "aux_loss_scales" not in encoder_config, "aux_loss_scales is only supposed to be used as a train_step param"
    adapter_config = asdict(n_config.adapter)
    qwen2_decoder_config_job = Qwen2DecoderConfigJobV2(
        n_config.decoder, l_config, target_filename=f"config-{n_config.decoder.name}-for-i6-spm.json"
    )
    decoder_config = {"config_path": qwen2_decoder_config_job.out_file}

    network_args = label_config | fe_config | encoder_config | adapter_config | decoder_config

    # Frozen layers
    if n_config.frozen_encoder_from_the_start():
        network_args["freeze_encoder_from_the_start"] = True
    if n_config.frozen_decoder_from_the_start():
        network_args["freeze_decoder_from_the_start"] = True

    # Lora
    if n_config.encoder_lora_opts is not None:
        network_args["encoder_lora_opts"] = asdict(n_config.encoder_lora_opts)
    if n_config.decoder_lora_opts is not None:
        network_args["decoder_lora_opts"] = asdict(n_config.decoder_lora_opts)

    return network_args