"""Local serializer entrypoint for the JXU LID-aware self-conditioned CTC model."""

from i6_experiments.users.jxu.experiments.multilingual.voxpopuli.pytorch_networks.language_aware_sc_ctc import (
    Model,
    get_model_config,
    prior_finish_hook,
    prior_init_hook,
    prior_step,
    train_step,
)
