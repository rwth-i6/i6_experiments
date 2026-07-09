from i6_core.rasr.config import RasrConfig


def get_no_op_label_scorer_config(scale: float = 1.0) -> RasrConfig:
    config = RasrConfig()
    config.type = "no-op"
    if scale != 1.0:
        config.scale = scale

    return config


def get_encoder_decoder_label_scorer_config(
    encoder_config: RasrConfig,
    decoder_label_scorer_config: RasrConfig = None,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:
    rasr_config = RasrConfig()
    if decoder_label_scorer_config is None:
        rasr_config.type = "encoder-only"
    else:
        rasr_config.type = "encoder-decoder"
        rasr_config.decoder = decoder_label_scorer_config

    rasr_config.encoder = encoder_config
    if scale != 1:
        rasr_config.scale = scale

    return rasr_config
