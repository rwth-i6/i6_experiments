from sisyphus import gs

import importlib

CONFIGS = [
    "config_01_baseline",
    "config_02_libri_other",
    "config_03_viterbi_realign",
    "config_04_pronunciation_variants",
    "config_05_swb_baseline",
    # "config_05a_cv",
    "config_06_trainable_tdps",
    "config_06a_dense",
    "config_06b_lbs",
    "config_07_encoder_comparison",
    "config_08_shifted_align",
    "config_08a_swb_align",
]

# gs.DEFAULT_ENVIRONMENT_SET["CUDA_VISIBLE_DEVICES"] = "0"

for config_name in CONFIGS:
    print("Loading config: {}".format(config_name))
    config = importlib.import_module(".".join(("config", config_name)))
    try:
        config.clean(gpu=True)
    except AttributeError:
        print(f"Could not clean {config_name}. No clean() function found.")
