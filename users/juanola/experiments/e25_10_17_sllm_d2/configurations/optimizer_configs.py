from .optimizer import \
    conformer_aed_weight_decay_blacklist_v2

v1 = {
    "optimizer": { # TODO: adapt to new configs
        "class": "adamw",
        "epsilon": 1e-16,
        "weight_decay": 0.01,
        "weight_decay_custom_include_check": conformer_aed_weight_decay_blacklist_v2,
    },
}
