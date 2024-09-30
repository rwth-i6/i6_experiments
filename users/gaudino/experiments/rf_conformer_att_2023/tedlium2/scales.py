"""All scales for tedlium2 experiments."""

model_names = [
    "model_baseline",

    # scales models
    "model_ctc0.9_att0.1",
    "model_ctc0.8_att0.2",
    "model_ctc0.7_att0.3",
    "model_ctc0.6_att0.4",
    "model_ctc0.5_att0.5",
    "model_ctc0.4_att0.6",
    "model_ctc0.3_att0.7",
    "model_ctc0.2_att0.8",
    "model_ctc0.1_att0.9",
    "model_ctc0.001_att0.999",

    # layer models baseline
    "model_baseline_lay6",
    "model_baseline_lay8",
    "model_baseline_lay10",

    # layer models ctc 0.3 att 0.7
    "model_ctc0.3_att0.7_lay6",
    "model_ctc0.3_att0.7_lay8",
    "model_ctc0.3_att0.7_lay10",

    # standalone models
    "model_att_only_currL",
    "model_ctc_only",

    # gauss window models


]

att_model_names = [name for name in model_names if "att" in name or "baseline" in name]
ctc_model_names = [name for name in model_names if "ctc" in name or "baseline" in name]
both_model_names = [name for name in model_names if ("att" in name and "ctc" in name) or "baseline" in name]

scales_model_names = [name for name in model_names if "ctc" in name and "att" in name and not "lay" in name]
scales_att_model_names = scales_model_names + ["model_att_only_currL"]
scales_ctc_model_names = ["model_ctc_only"] + scales_att_model_names

scales_model_names_exclude_edge = [name for name in model_names if "ctc" in name and "att" in name and not "lay" in name and not "999" in name]
scales_att_model_names_exclude_edge = scales_model_names_exclude_edge + ["model_att_only_currL"]
scales_ctc_model_names_exclude_edge = ["model_ctc_only"] + scales_model_names_exclude_edge


train_scale = {
    "model_ctc0.9_att0.1": 0.1,
    "model_ctc0.8_att0.2": 0.2,
    "model_ctc0.7_att0.3": 0.3,
    "model_ctc0.6_att0.4": 0.4,
    "model_ctc0.5_att0.5": 0.5,
    "model_ctc0.4_att0.6": 0.6,
    "model_ctc0.3_att0.7": 0.7,
    "model_ctc0.2_att0.8": 0.8,
    "model_ctc0.1_att0.9": 0.9,
    "model_ctc0.001_att0.999": 0.999,

    "model_att_only_currL": 1.0,

    "model_ctc_only": 0.0,
}

### no lm

# att
scales_att = {
    "model_baseline": {
        "wer": (7.4, 6.9),
    },
    "model_ctc0.9_att0.1": {
        "wer": (8.3, 7.7),
    },
    "model_ctc0.8_att0.2": {
        "wer": (7.8, 7.2),
    },
    "model_ctc0.7_att0.3": {
        "wer": (7.7, 7.2),
    },
    "model_ctc0.6_att0.4": {
        "wer": (7.7, 7.0),
    },
    "model_ctc0.5_att0.5": {
        "wer": (7.5, 6.9),
    },
    "model_ctc0.4_att0.6": {
        "wer": (7.6, 6.7),
    },
    "model_ctc0.3_att0.7": {
        "wer": (7.7, 6.9),
    },
    "model_ctc0.2_att0.8": {
        "wer": (7.4, 6.9),
    },
    "model_ctc0.1_att0.9": {
        "wer": (7.9, 6.9),
    },
    "model_ctc0.001_att0.999": {
        "wer": (9.2, 8.6),
    },
    "model_baseline_lay6": {
        "wer": (8.1, 7.1),
    },
    "model_baseline_lay8": {
        "wer": (7.7, 7.2),
    },
    "model_baseline_lay10": {
        "wer": (7.7, 7.2),
    },
    "model_ctc0.3_att0.7_lay6": {
        "wer": (7.9, 7.2),
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (7.5, 7.1),
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (7.7, 6.9),
    },
    "model_att_only_currL": {
        "wer": (9.3, 9.1),
    },
} # done

# ctc greedy + prior
scales_ctc_prior = {
    "model_baseline": {
        "wer": (8.2, 7.9),
        "scales": [[0.15]],  # dev/test 8.39/8.01 -> 8.19/7.92
    },
    # "model_ctc0.43_att1.0": {  # dev/test 8.62/7.97 -> 8.58/7.86
    #     "prior_scale": [0.15],
    # },
    # "model_ctc0.25_att1.0": {
    #     "prior_scale": [0.22],  # dev/test 9.03/8.32 -> 8.79/8.25
    # },
    # "model_ctc0.2_att1.0": {  # dev/test 9.56/8.67 -> 9.38/8.65
    #     "prior_scale": [0.2],
    # },
    "model_ctc0.9_att0.1": {
        "wer": (8.9, 8.4),
        "scales": [[0.22]],  # bsf 10 dev/test 9.04/8.33 -> 8.85/8.44
    },
    "model_ctc0.8_att0.2": {
        "wer": (9.0, 8.2),
        "scales": [[0.2]],  # bsf 10 dev/test 9.03/8.24 -> 8.96/8.21
    },
    "model_ctc0.7_att0.3": {
        "wer": (8.6, 8.0),
        "scales": [[0.22]],  # bsf 10 dev/test 8.67/8.00 -> 8.58/7.94
    },
    "model_ctc0.6_att0.4": {
        "wer": (8.6, 8.0),
        "scales": [[0.2]],  # bsf 10 dev/test 8.65/8.04 -> 8.64/7.98
    },
    "model_ctc0.5_att0.5": {
        "wer": (8.3, 7.9),
        "scales": [[0.2]],  # bsf 10 dev/test 8.50/8.03 -> 8.31/7.92
    },
    "model_ctc0.4_att0.6": {
        "wer": (8.4, 7.9),
        "scales": [[0.2]],  # bsf 10 dev/test 8.55/7.76 -> 8.42/7.89
    },
    "model_ctc0.3_att0.7": {
        "wer": (8.5, 8.1),
        "scales": [[0.25]],  # dev/test 8.58/8.15 -> 8.46/8.11
    },
    "model_ctc0.2_att0.8": {
        "wer": (8.8, 8.3),
        "scales": [[0.22]],  # dev/test 9.05/8.35 -> 8.78/8.33
    },
    "model_ctc0.1_att0.9": {
        "wer": (9.8, 9.2),
        "scales": [[0.17]],  # dev/test 9.92/9.22 -> 9.84/9.20
    },
    "model_ctc0.001_att0.999": {
        "wer": (26.3, 24.8),
        "scales": [[0.2]],  # dev/test 27.00/25.10 -> 26.32/24.76
    },
    "model_ctc0.3_att0.7_lay6": {
        "wer": (10.7, 9.7),
        "scales": [[0.15]],  # dev/test 10.75/9.83 -> 10.70/9.74
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (9.7, 9.0),
        "scales": [[0.2]],  # dev/test 9.68/9.11 -> 9.66/9.01
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (9.0, 8.3),
        "scales": [[0.2]],  # dev/test 9.26/8.44 -> 9.01/8.32
    },
    "model_baseline_lay6": {
        "wer": (10.2, 9.6),
        "scales": [[0.2]],  # dev/test 10.34/9.51 -> 10.22/9.62
    },
    "model_baseline_lay8": {
        "wer": (9.4, 8.6),
        "scales": [[0.15]],  # dev/test 9.53/8.73 -> 9.4/8.64
    },
    "model_baseline_lay10": {
        "wer": (9.0, 8.4),
        "scales": [[0.25]],  # dev/test 9.38/8.56 -> 9.03/8.44
    },
    "model_ctc_only": {
        "wer": (9.2, 8.4),
        "scales": [[0.17]],  # dev/test 9.27/8.46 -> 9.23/8.37
    },
} # done

# opls ctc + prior
scales_ctc_prior_opls = {
    "model_baseline": {
        "wer": (8.3, 7.8),
        "scales": [[0.1]],
    },
    "model_ctc0.9_att0.1": {
        "wer": (8.8, 8.2),
        "scales": [[0.1]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (8.9, 8.1),
        "scales": [[0.1]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (8.4, 7.8),
        "scales": [[0.1]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (8.5, 7.9),
        "scales": [[0.0]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (8.4, 8.0),
        "scales": [[0.2]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (8.3, 7.8),
        "scales": [[0.1]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (8.4, 7.9),
        "scales": [[0.1]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (8.5, 8.2),
        "scales": [[0.1]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (9.6, 9.1),
        "scales": [[0.1]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (25.3, 23.3),
        "scales": [[0.0]],
    },
    "model_ctc_only": {
        "wer": (9.1, 8.5),
        "scales": [[0.1]],
    },
    "model_ctc0.3_att0.7_lay6": {
        "wer": (10.6, 9.7),
        "scales": [[0.0]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (9.4, 8.9),
        "scales": [[0.1]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (9.0, 8.4),
        "scales": [[0.2]],
    },
    "model_baseline_lay6": {
        "wer": (10.1, 9.6),
        "scales": [[0.1]],
    },
    "model_baseline_lay8": {
        "wer": (9.2, 8.6),
        "scales": [[0.2]],
    },
    "model_baseline_lay10": {
        "wer": (8.9, 8.4),
        "scales": [[0.2]],
    },
} # done

# tsbs ctc + prior
scales_ctc_prior_tsbs = {
    "model_baseline": {"scales": [[0.1]], "wer": None},
} # done for now

# optsr att + ctc
scales_att_ctc_optsr = {
    "model_baseline": {
        "wer": (7.1, 6.8),
        "scales": [[0.85, 0.15, 0.3]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (7.7, 7.2),
        "scales": [[0.75, 0.25, 0.6]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (7.5, 7.1),
        "scales": [[0.7, 0.3, 0.7]], # did not try higher prior scale
    },
    "model_ctc0.7_att0.3": {
        "wer": (7.1, 6.9),
        "scales": [[0.75, 0.25, 0.4]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (7.3, 6.7),
        "scales": [[0.75, 0.25, 0.4]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (7.2, 6.6),
        "scales": [[0.85, 0.15, 0.4]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (7.2, 6.7),
        "scales": [[0.8, 0.2, 0.5]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (7.2, 6.7),
        "scales": [[0.7, 0.3, 0.4]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (7.2, 6.9),
        "scales": [[0.7, 0.3, 0.3]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (7.7, 6.9),
        "scales": [[0.85, 0.15, 0.1]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (9.3, 9.0),
        "scales": [[0.9, 0.1, 0.0]],
    },

    "model_baseline_lay6": {
        "wer": (8.1, 7.2),
        "scales": [[0.7, 0.3, 0.0]], # broken
    },
    "model_baseline_lay8": {
        "wer": (7.6, 7.0),
        "scales": [[0.5, 0.5, 0.0]],
    },
    "model_baseline_lay10": {
        "wer": (9.2, 8.4),
        "scales": [[0.5, 0.5, 0.0]], # broken
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (8.1, 7.9),
        "scales": [[0.7, 0.3, 0.0]], # broken
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (7.7, 7.2),
        "scales": [[0.55, 0.45, 0.0]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (8.3, 7.6),
        "scales": [[0.5, 0.5, 0.0]], # broken
    },
} # done

# opls att + ctc
scales_att_ctc_opls = {
    "model_baseline": {
        "wer": (7.0, 6.6),
        "scales": [[0.8, 0.2, 0.75]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (7.7, 7.3),
        "scales": [[0.6, 0.4, 0.6]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (7.5, 6.9),
        "scales": [[0.65, 0.35, 0.6]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (7.0, 6.9),
        "scales": [[0.65, 0.35, 0.8]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (7.3, 6.7),
        "scales": [[0.75, 0.25, 0.7]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (7.0, 6.6),
        "scales": [[0.7, 0.3, 0.8]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (7.0, 6.6),
        "scales": [[0.8, 0.2, 0.8]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (7.2, 6.7),
        "scales": [[0.67, 0.33, 0.7]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (7.1, 6.7),
        "scales": [[0.8, 0.2, 0.6]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (7.5, 6.8),
        "scales": [[0.8, 0.2, 0.6]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (9.1, 8.5),
        "scales": [[0.9, 0.1, 0.9]],
    },

    "model_baseline_lay6": {
        "wer": (7.9, 6.9),
        "scales": [[0.85, 0.15, 0.8]],
    },
    "model_baseline_lay8": {
        "wer": (7.5, 6.9),
        "scales": [[0.8, 0.2, 0.9]],
    },
    "model_baseline_lay10": {
        "wer": (7.4, 7.0),
        "scales": [[0.8, 0.2, 0.9]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (7.7, 7.1),
        "scales": [[0.85, 0.15, 0.5]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (7.3, 6.9),
        "scales": [[0.85, 0.15, 0.7]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (7.5, 6.8),
        "scales": [[0.9, 0.1, 0.9]],
    },
} # done

# tsbs att + ctc
scales_att_ctc_tsbs = {
    "model_baseline": {"scales": [[0.5, 0.5, 0.4]], "wer": (7.1, 6.7)},
} # done for now

# ------------------------ with trafo lm --------------------------

# att + lm

scales_att_lm = {
    "model_baseline": {
        "scales": [[0.18]],
        "wer": (6.8, 6.4),
    },
    "model_ctc0.9_att0.1": {
        "scales": [[0.23]],
        "wer": (7.5, 6.9),
    },
    "model_ctc0.8_att0.2": {
        "scales": [[0.25]],
        "wer": (7.3, 6.5),
    },
    "model_ctc0.7_att0.3": {
        "scales": [[0.3]],
        "wer": (7.1, 6.5),
    },
    "model_ctc0.6_att0.4": {
        "scales": [[0.25]],
        "wer": (6.8, 6.3),
    },
    "model_ctc0.5_att0.5": {
        "scales": [[0.3]],
        "wer": (6.6, 6.3),
    },
    "model_ctc0.4_att0.6": {
        "scales": [[0.23]],
        "wer": (6.9, 6.3),
    },
    "model_ctc0.3_att0.7": {
        "scales": [[0.3]],
        "wer": (7.0, 6.3),
    },
    "model_ctc0.2_att0.8": {
        "scales": [[0.23]],
        "wer": (6.8, 6.5),
    },
    "model_ctc0.1_att0.9": {
        "scales": [[0.27]],
        "wer": (7.3, 6.3),
    },
    "model_ctc0.001_att0.999": {
        "scales": [[0.21]],
        "wer": (8.1, 7.6),
    },
    "model_baseline_lay6": {
        "scales": [[0.2]],
        "wer": (7.3, 6.6),
    },
    "model_baseline_lay8": {
        "scales": [[0.25]],
        "wer": (7.0, 6.5),
    },
    "model_baseline_lay10": {
        "scales": [[0.29]],
        "wer": (6.9, 6.5),
    },
    "model_ctc0.3_att0.7_lay6": {
        "scales": [[0.25]],
        "wer": (7.2, 6.6),
    },
    "model_ctc0.3_att0.7_lay8": {
        "scales": [[0.23]],
        "wer": (6.7, 6.4),
    },
    "model_ctc0.3_att0.7_lay10": {
        "scales": [[0.25]],
        "wer": (6.9, 6.2),
    },
    "model_att_only_currL": {
        "scales": [[0.3]],
        "wer": (8.2, 8.6), # wierd
    },
} # done

# ctc + lm optsr

scales_ctc_lm_optsr = {
    "model_baseline": {
        "wer": (7.1, 6.7),
        "scales": [scales_ctc_prior["model_baseline"]["scales"][0] + [0.4]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (7.2, 6.7),
        "scales": [scales_ctc_prior["model_ctc0.9_att0.1"]["scales"][0] + [0.5]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (7.3, 6.7),
        "scales": [scales_ctc_prior["model_ctc0.8_att0.2"]["scales"][0] + [0.45]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (6.9, 6.6),
        "scales": [scales_ctc_prior["model_ctc0.7_att0.3"]["scales"][0] + [0.5]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (7.1, 6.5),
        "scales": [scales_ctc_prior["model_ctc0.6_att0.4"]["scales"][0] + [0.5]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (7.1, 6.5),
        "scales": [scales_ctc_prior["model_ctc0.5_att0.5"]["scales"][0] + [0.45]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (6.9, 6.5),
        "scales": [scales_ctc_prior["model_ctc0.4_att0.6"]["scales"][0] + [0.45]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (6.9, 6.5),
        "scales": [scales_ctc_prior["model_ctc0.3_att0.7"]["scales"][0] + [0.5]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (7.0, 6.6),
        "scales": [scales_ctc_prior["model_ctc0.2_att0.8"]["scales"][0] + [0.5]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (7.6, 7.0),
        "scales": [scales_ctc_prior["model_ctc0.1_att0.9"]["scales"][0] + [0.45]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (12.8, 11.8),
        "scales": [scales_ctc_prior["model_ctc0.001_att0.999"]["scales"][0] + [0.5]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (98, 98),
        "scales": [scales_ctc_prior["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.1]], # very broken
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (19.4, 20.4),
        "scales": [scales_ctc_prior["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.1]], # broken
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (10, 10.3),
        "scales": [scales_ctc_prior["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.3]],
    },
    "model_baseline_lay6": {
        "wer": (91, 90),
        "scales": [scales_ctc_prior["model_baseline_lay6"]["scales"][0] + [0.1]],
    },
    "model_baseline_lay8": {
        "wer": (29, 29),
        "scales": [scales_ctc_prior["model_baseline_lay8"]["scales"][0] + [0.1]],
    },
    "model_baseline_lay10": {
        "wer": (7.9, 7.8),
        "scales": [scales_ctc_prior["model_baseline_lay10"]["scales"][0] + [0.3]],
    },

    "model_ctc_only": {
        "wer": (7.6, 7.0),
        "scales": [scales_ctc_prior["model_ctc_only"]["scales"][0] + [0.4]],
    },
} # why do models break?

# ctc + lm opls

scales_ctc_lm_opls = {
    "model_baseline": {
        "wer": (6.4, 6.4),
        "scales": [[0.2, 0.7]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (7.0, 6.6),
        "scales": [scales_ctc_prior_opls["model_ctc0.9_att0.1"]["scales"][0] + [0.6]], # did not try higher lm scale
    },
    "model_ctc0.8_att0.2": {
        "wer": (6.9, 6.5),
        "scales": [scales_ctc_prior_opls["model_ctc0.8_att0.2"]["scales"][0] + [0.6]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (6.6, 6.5),
        "scales": [scales_ctc_prior_opls["model_ctc0.7_att0.3"]["scales"][0]+ [0.5]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (6.8, 6.5),
        "scales": [scales_ctc_prior_opls["model_ctc0.6_att0.4"]["scales"][0] + [0.7]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (6.6, 6.4),
        "scales": [scales_ctc_prior_opls["model_ctc0.5_att0.5"]["scales"][0] + [0.55]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (6.7, 6.4),
        "scales": [scales_ctc_prior_opls["model_ctc0.4_att0.6"]["scales"][0] + [0.55]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (6.7, 6.2),
        "scales": [scales_ctc_prior_opls["model_ctc0.3_att0.7"]["scales"][0]+ [0.55]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (6.6, 6.4),
        "scales": [scales_ctc_prior_opls["model_ctc0.2_att0.8"]["scales"][0] + [0.65]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (6.9, 6.6),
        "scales": [scales_ctc_prior_opls["model_ctc0.1_att0.9"]["scales"][0] + [0.7]], # did not try higher
    },
    "model_ctc0.001_att0.999": {
        "wer": (10.1, 9.4),
        "scales": [scales_ctc_prior_opls["model_ctc0.001_att0.999"]["scales"][0] + [0.7]], # did not try highger
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (7.1, 6.7),
        "scales": [scales_ctc_prior_opls["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.6]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (7.0, 6.4),
        "scales": [scales_ctc_prior_opls["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.55]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (6.9, 6.5),
        "scales": [scales_ctc_prior_opls["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.65]],
    },
    "model_baseline_lay6": {
        "wer": (7.2, 6.6),
        "scales": [scales_ctc_prior_opls["model_baseline_lay6"]["scales"][0] + [0.65]],
    },
    "model_baseline_lay8": {
        "wer": (7.0, 6.4),
        "scales": [scales_ctc_prior_opls["model_baseline_lay8"]["scales"][0] + [0.5]],
    },
    "model_baseline_lay10": {
        "wer": (6.9, 6.4),
        "scales": [scales_ctc_prior_opls["model_baseline_lay10"]["scales"][0]+ [0.65]],
    },

    "model_ctc_only": {
        "wer": (7.1, 6.6),
        "scales": [scales_ctc_prior_opls["model_ctc_only"]["scales"][0] + [0.6]],
    },
} # done

# att + ctc + lm optsr

scales_att_ctc_lm_optsr = {
    "model_baseline": {
        "wer": (6.4, 6.2),
        "scales": [scales_att_ctc_optsr["model_baseline"]["scales"][0] + [0.55]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (6.7, 6.5),
        "scales": [scales_att_ctc_optsr["model_ctc0.9_att0.1"]["scales"][0] + [0.4]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (6.4, 6.2),
        "scales": [scales_att_ctc_optsr["model_ctc0.8_att0.2"]["scales"][0] + [0.55]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (6.5, 6.1),
        "scales": [scales_att_ctc_optsr["model_ctc0.7_att0.3"]["scales"][0] + [0.5]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (6.6, 6.1),
        "scales": [scales_att_ctc_optsr["model_ctc0.6_att0.4"]["scales"][0] + [0.4]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (6.4, 6.0),
        "scales": [scales_att_ctc_optsr["model_ctc0.5_att0.5"]["scales"][0] + [0.5]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (6.3, 6.1),
        "scales": [scales_att_ctc_optsr["model_ctc0.4_att0.6"]["scales"][0] + [0.55]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (6.4, 6.0),
        "scales": [scales_att_ctc_optsr["model_ctc0.3_att0.7"]["scales"][0] + [0.45]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_optsr["model_ctc0.2_att0.8"]["scales"][0] + [0.4]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (6.9, 6.3),
        "scales": [scales_att_ctc_optsr["model_ctc0.1_att0.9"]["scales"][0] + [0.4]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (8.3, 7.6),
        "scales": [scales_att_ctc_optsr["model_ctc0.001_att0.999"]["scales"][0] + [0.3]], # did not try lower
    },

    "model_baseline_lay6": {
        "wer": (89, 88),
        "scales": [scales_att_ctc_optsr["model_baseline_lay6"]["scales"][0] + [0.3]], # broken
    },
    "model_baseline_lay8": {
        "wer": (22, 17),
        "scales": [scales_att_ctc_optsr["model_baseline_lay8"]["scales"][0] + [0.4]], # broken
    },
    "model_baseline_lay10": {
        "wer": (6.6, 6.1),
        "scales": [scales_att_ctc_optsr["model_baseline_lay10"]["scales"][0] + [0.3]], # did not try lower
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (44, 37),
        "scales": [scales_att_ctc_optsr["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.3]], # broken
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (10.1, 7.5),
        "scales": [scales_att_ctc_optsr["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.3]], # broken
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (6.4, 6.2),
        "scales": [scales_att_ctc_optsr["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.3]], # did not try lower
    },
} # done

# att + ctc + lm opls

scales_att_ctc_lm_opls = {
    "model_baseline": {
        "wer": (6.3, 6.2),
        "scales": [scales_att_ctc_opls["model_baseline"]["scales"][0] + [0.4]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (6.7, 6.5),
        "scales": [scales_att_ctc_opls["model_ctc0.9_att0.1"]["scales"][0] + [0.4]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (6.6, 6.3),
        "scales": [scales_att_ctc_opls["model_ctc0.8_att0.2"]["scales"][0] + [0.55]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_opls["model_ctc0.7_att0.3"]["scales"][0] + [0.4]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_opls["model_ctc0.6_att0.4"]["scales"][0] + [0.5]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (6.3, 6.1),
        "scales": [scales_att_ctc_opls["model_ctc0.5_att0.5"]["scales"][0] + [0.45]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_opls["model_ctc0.4_att0.6"]["scales"][0] + [0.35]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (6.3, 5.9),
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7"]["scales"][0] + [0.5]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_opls["model_ctc0.2_att0.8"]["scales"][0] + [0.45]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (6.5, 6.0),
        "scales": [scales_att_ctc_opls["model_ctc0.1_att0.9"]["scales"][0] + [0.4]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (7.5, 7.2),
        "scales": [scales_att_ctc_opls["model_ctc0.001_att0.999"]["scales"][0] + [0.4]],
    },

    "model_baseline_lay6": {
        "wer": (6.6, 6.2),
        "scales": [scales_att_ctc_opls["model_baseline_lay6"]["scales"][0] + [0.35]],
    },
    "model_baseline_lay8": {
        "wer": (6.4, 6.0),
        "scales": [scales_att_ctc_opls["model_baseline_lay8"]["scales"][0] + [0.45]],
    },
    "model_baseline_lay10": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_opls["model_baseline_lay10"]["scales"][0] + [0.45]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (6.7, 6.2),
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.45]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (6.3, 6.2),
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.5]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (6.6, 6.1),
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.45]],
    },
} # done


# ----------------- system combination --------------------------

# att + ctc only model optsr

scales_att_ctc_only_optsr = {
    "model_baseline": {
        "wer": (7.1, 6.7),
        "scales": [[0.85, 0.15, 0.1]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (7.4, 6.9),
        "scales": [[0.7, 0.3, 0.6]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (7.2, 6.7),
        "scales": [[0.8, 0.2, 0.5]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (7.1, 6.7),
        "scales": [[0.85, 0.15, 0.4]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (7.2, 6.7),
        "scales": [[0.8, 0.2, 0.4]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (6.9, 6.5),
        "scales": [[0.85, 0.15, 0.3]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (7.1, 6.5),
        "scales": [[0.85, 0.15, 0.2]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (7.0, 6.5),
        "scales": [[0.85, 0.15, 0.2]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (7.0, 6.5),
        "scales": [[0.85, 0.15, 0.3]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (7.2, 6.4),
        "scales": [[0.85, 0.15, 0.2]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (7.3, 6.8),
        "scales": [[0.75, 0.25, 0.3]],
    },

    "model_baseline_lay6": {
        "wer": (7.2, 6.5),
        "scales": [[0.8, 0.2, 0.4]],
    },
    "model_baseline_lay8": {
        "wer": (7.1, 6.6),
        "scales": [[0.8, 0.2, 0.2]],
    },
    "model_baseline_lay10": {
        "wer": (7.2, 6.7),
        "scales": [[0.85, 0.15, 0.4]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (7.4, 6.8),
        "scales": [[0.85, 0.15, 0.2]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (7.0, 6.6),
        "scales": [[0.85, 0.15, 0.2]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (7.0, 6.6),
        "scales": [[0.85, 0.15, 0.5]],
    },

    "model_att_only_currL": {
        "wer": (7.3, 6.7),
        "scales": [[0.7, 0.3, 0.2]],
    },
} # done

# att + ctc only model opls

scales_att_ctc_only_opls = {
    "model_baseline": {
        "wer": (7.0, 6.6),
        "scales": [[0.8, 0.2, 0.5]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (7.4, 6.9),
        "scales": [[0.65, 0.35, 0.7]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (7.3, 6.6),
        "scales": [[0.75, 0.25, 0.5]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (7.1, 6.7),
        "scales": [[0.85, 0.15, 0.8]], # did not try higher
    },
    "model_ctc0.6_att0.4": {
        "wer": (7.2, 6.7),
        "scales": [[0.8, 0.2, 0.8]], # did not try higher
    },
    "model_ctc0.5_att0.5": {
        "wer": (7.1, 6.5),
        "scales": [[0.85, 0.15, 0.7]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (7.0, 6.4),
        "scales": [[0.8, 0.2, 0.7]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (7.0, 6.5),
        "scales": [[0.75, 0.25, 0.8]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (7.0, 6.5),
        "scales": [[0.8, 0.2, 0.4]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (7.2, 6.5),
        "scales": [[0.65, 0.35, 0.7]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (7.4, 6.7),
        "scales": [[0.75, 0.25, 0.7]],
    },

    "model_baseline_lay6": {
        "wer": (7.3, 6.5),
        "scales": [[0.65, 0.35, 0.7]],
    },
    "model_baseline_lay8": {
        "wer": (7.2, 6.6),
        "scales": [[0.75, 0.25, 0.4]],
    },
    "model_baseline_lay10": {
        "wer": (7.1, 6.6),
        "scales": [[0.75, 0.25, 0.7]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (7.4, 6.7),
        "scales": [[0.85, 0.15, 0.7]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (7.1, 6.6),
        "scales": [[0.9, 0.1, 0.8]], # did not try higher both
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (7.1, 6.5),
        "scales": [[0.85, 0.15, 0.8]],
    },

    "model_att_only_currL": {
        "wer": (7.4, 6.7),
        "scales": [[0.65, 0.35, 0.6]],
    },
} # done

# att + ctc only model + lm optsr

scales_att_ctc_only_lm_optsr = {
    "model_baseline": {
        "wer": (6.6, 6.2),
        "scales": [scales_att_ctc_only_optsr["model_baseline"]["scales"][0] + [0.3]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (6.5, 6.1),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.9_att0.1"]["scales"][0] + [0.6]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.8_att0.2"]["scales"][0] + [0.55]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.7_att0.3"]["scales"][0] + [0.5]],
    },
    "model_ctc0.6_att0.4": {
        "wer": (6.4, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.6_att0.4"]["scales"][0] + [0.55]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (6.3, 6.1),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.5_att0.5"]["scales"][0] + [0.6]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (6.6, 6.1),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.4_att0.6"]["scales"][0] + [0.3]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (6.5 ,6.1),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.3_att0.7"]["scales"][0] + [0.4]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (6.4, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.2_att0.8"]["scales"][0] + [0.55]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (6.6, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.1_att0.9"]["scales"][0] + [0.4]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (6.7, 6.1),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.001_att0.999"]["scales"][0] + [0.55]],
    },

    "model_baseline_lay6": {
        "wer": (6.5, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_baseline_lay6"]["scales"][0] + [0.45]],
    },
    "model_baseline_lay8": {
        "wer": (6.5, 6.2),
        "scales": [scales_att_ctc_only_optsr["model_baseline_lay8"]["scales"][0] + [0.45]],
    },
    "model_baseline_lay10": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_baseline_lay10"]["scales"][0] + [0.6]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (6.5, 6.1),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.5]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.55]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (6.2, 5.9),
        "scales": [scales_att_ctc_only_optsr["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.5]],
    },

    "model_att_only_currL": {
        "wer": (6.8, 6.3),
        "scales": [scales_att_ctc_only_optsr["model_att_only_currL"]["scales"][0] + [0.4]],
    },
} # done

# att + ctc only model + lm opls

scales_att_ctc_only_lm_opls = {
    "model_baseline": {
        "wer": (6.3, 5.9),
        "scales": [scales_att_ctc_only_opls["model_baseline"]["scales"][0] + [0.43]],
    },

    "model_ctc0.9_att0.1": {
        "wer": (6.5, 6.3),
        "scales": [scales_att_ctc_only_opls["model_ctc0.9_att0.1"]["scales"][0] + [0.4]],
    },
    "model_ctc0.8_att0.2": {
        "wer": (6.4, 6.0),
        "scales": [scales_att_ctc_only_opls["model_ctc0.8_att0.2"]["scales"][0] + [0.4]],
    },
    "model_ctc0.7_att0.3": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_only_opls["model_ctc0.7_att0.3"]["scales"][0] + [0.3]], # did not try lower
    },
    "model_ctc0.6_att0.4": {
        "wer": (6.3, 6.2),
        "scales": [scales_att_ctc_only_opls["model_ctc0.6_att0.4"]["scales"][0] + [0.45]],
    },
    "model_ctc0.5_att0.5": {
        "wer": (6.3, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.5_att0.5"]["scales"][0] + [0.45]],
    },
    "model_ctc0.4_att0.6": {
        "wer": (6.3, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.4_att0.6"]["scales"][0] + [0.4]],
    },
    "model_ctc0.3_att0.7": {
        "wer": (6.2, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.3_att0.7"]["scales"][0] + [0.4]],
    },
    "model_ctc0.2_att0.8": {
        "wer": (6.2, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.2_att0.8"]["scales"][0] + [0.45]],
    },
    "model_ctc0.1_att0.9": {
        "wer": (6.3, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.1_att0.9"]["scales"][0] + [0.4]],
    },
    "model_ctc0.001_att0.999": {
        "wer": (6.6, 6.1),
        "scales": [scales_att_ctc_only_opls["model_ctc0.001_att0.999"]["scales"][0] + [0.45]],
    },

    "model_baseline_lay6": {
        "wer": (6.4, 6.1),
        "scales": [scales_att_ctc_only_opls["model_baseline_lay6"]["scales"][0] + [0.4]],
    },
    "model_baseline_lay8": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_only_opls["model_baseline_lay8"]["scales"][0] + [0.5]],
    },
    "model_baseline_lay10": {
        "wer": (6.4, 5.9),
        "scales": [scales_att_ctc_only_opls["model_baseline_lay10"]["scales"][0] + [0.45]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "wer": (6.3, 6.0),
        "scales": [scales_att_ctc_only_opls["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.4]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": (6.3, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.45]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": (6.2, 5.9),
        "scales": [scales_att_ctc_only_opls["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.45]],
    },

    "model_att_only_currL": {
        "wer": (6.5, 6.2),
        "scales": [scales_att_ctc_only_opls["model_att_only_currL"]["scales"][0] + [0.45]],
    },
} # done