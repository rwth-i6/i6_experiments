"""Opt-in 256-utterance clean/20/10 dB pilot; nine GPU recognitions only."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.public_pilots import register_noise_pilot

    return register_noise_pilot(gs.OUTPUT_DIR)
