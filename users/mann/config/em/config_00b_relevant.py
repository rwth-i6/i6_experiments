from sisyphus import gs

import importlib

CONFIGS = [
    "config_06a_dense",
    "config_06b_lbs",
    "config_07_encoder_comparison",
    # "config_08a_swb_align",
    # "config_09_one_state",
    "config_10_tedlium",
]

for config_name in CONFIGS:
    print("Loading config: {}".format(config_name))
    config = importlib.import_module(".".join(("config", config_name)))
    config.all()
