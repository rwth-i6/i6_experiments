from i6_experiments.users.juanola.experiments.e25_09_08_demo1_sllm.extra_code.optimizer import \
    conformer_aed_weight_decay_blacklist_v2

v1 = {
    "optimizer": {
        "class": "adamw",
        "epsilon": 1e-16,
        "weight_decay": 0.01,
        "weight_decay_custom_include_check": conformer_aed_weight_decay_blacklist_v2,
    },
}
