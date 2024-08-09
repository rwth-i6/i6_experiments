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

### no lm

# att
scales_att = {
    "model_baseline": {
        "wer": (7.4, 6.9),
    },
    "model_ctc0.9_att0.1": {
        "wer": None,
    },
    "model_ctc0.8_att0.2": {
        "wer": None,
    },
    "model_ctc0.7_att0.3": {
        "wer": None,
    },
    "model_ctc0.6_att0.4": {
        "wer": None,
    },
    "model_ctc0.5_att0.5": {
        "wer": None,
    },
    "model_ctc0.4_att0.6": {
        "wer": None,
    },
    "model_ctc0.3_att0.7": {
        "wer": None,
    },
    "model_ctc0.2_att0.8": {
        "wer": None,
    },
    "model_ctc0.1_att0.9": {
        "wer": None,
    },
    "model_ctc0.001_att0.999": {
        "wer": None,
    },
    "model_baseline_lay6": {
        "wer": None,
    },
    "model_baseline_lay8": {
        "wer": None,
    },
    "model_baseline_lay10": {
        "wer": None,
    },
    "model_ctc0.3_att0.7_lay6": {
        "wer": None,
    },
    "model_ctc0.3_att0.7_lay8": {
        "wer": None,
    },
    "model_ctc0.3_att0.7_lay10": {
        "wer": None,
    },
    "model_att_only_currL": {
        "wer": None,
    },
}

# ctc greedy + prior
scales_ctc_prior = {}

# opls ctc + prior
scales_ctc_prior_opls = {
    "model_baseline": {
        "scales": [[0.1]],
    },
    "model_ctc0.9_att0.1": {
        "scales": [[0.1]],
    },
    "model_ctc0.8_att0.2": {
        "scales": [[0.1]],
    },
    "model_ctc0.7_att0.3": {
        "scales": [[0.1]],
    },
    "model_ctc0.6_att0.4": {
        "scales": [[0.0]],
    },
    "model_ctc0.5_att0.5": {
        "scales": [[0.2]],
    },
    "model_ctc0.4_att0.6": {
        "scales": [[0.1]],
    },
    "model_ctc0.3_att0.7": {
        "scales": [[0.1]],
    },
    "model_ctc0.2_att0.8": {
        "scales": [[0.1]],
    },
    "model_ctc0.1_att0.9": {
        "scales": [[0.1]],
    },
    "model_ctc0.001_att0.999": {
        "scales": [[0.0]],
    },
    "model_ctc_only": {
        "scales": [[0.1]],
    },
    "model_ctc0.3_att0.7_lay6": {
        "scales": [],
    },
    "model_ctc0.3_att0.7_lay8": {
        "scales": [],
    },
    "model_ctc0.3_att0.7_lay10": {
        "scales": [],
    },
    "model_ctc1.0_att1.0_lay6": {
        "scales": [],
    },
    "model_ctc1.0_att1.0_lay8": {
        "scales": [],
    },
}

# tsbs ctc + prior
scales_ctc_prior_tsbs = {
    "model_baseline": {"scales": [[]], "wer": None},
}

# optsr att + ctc
scales_att_ctc_optsr = {}

# opls att + ctc
scales_att_ctc_opls = {
    "model_baseline": {
        "wer": (7.0, 6.6),
        "scales": [[0.8, 0.2, 0.75]],
    },

    "model_ctc0.9_att0.1": {
        "scales": [[0.6, 0.4, 0.6]],
    },
    "model_ctc0.8_att0.2": {
        "scales": [[0.65, 0.35, 0.6]],
    },
    "model_ctc0.7_att0.3": {
        "scales": [[0.65, 0.35, 0.8]],
    },
    "model_ctc0.6_att0.4": {
        "scales": [[0.75, 0.25, 0.7]],
    },
    "model_ctc0.5_att0.5": {
        "scales": [[0.7, 0.3, 0.8]],
    },
    "model_ctc0.4_att0.6": {
        "scales": [[0.8, 0.2, 0.8]],
    },
    "model_ctc0.3_att0.7": {
        "scales": [[0.67, 0.33, 0.7]],
    },
    "model_ctc0.2_att0.8": {
        "scales": [[0.8, 0.2, 0.6]],
    },
    "model_ctc0.1_att0.9": {
        "scales": [[0.8, 0.2, 0.6]],
    },
    "model_ctc0.001_att0.999": {
        "scales": [[0.9, 0.1, 0.9]],
    },

    "model_baseline_lay6": {
        "scales": [[0.85, 0.15, 0.8]],
    },
    "model_baseline_lay8": {
        "scales": [[0.8, 0.2, 0.9]],
    },
    "model_baseline_lay10": {
        "scales": [[0.8, 0.2, 0.9]],
    },

    "model_ctc0.3_att0.7_lay6": {
        "scales": [[0.85, 0.15, 0.5]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "scales": [[0.85, 0.15, 0.7]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "scales": [[0.9, 0.1, 0.9]],
    },
    "model_att_only_currL": {
        "scales": [],
    }
}

# tsbs att + ctc
scales_att_ctc_tsbs = {
    "model_baseline": {"scales": [[0.5, 0.5, 0.4]], "wer": (7.1, 6.7)},
}

### with trafo lm

# att + lm

scales_att_lm = {}

# ctc + lm optsr

scales_ctc_lm_optsr = {}

# ctc + lm opls

scales_ctc_lm_opls = {

}

# att + ctc + lm optsr

scales_att_ctc_lm_optsr = {}

# att + ctc + lm opls

scales_att_ctc_lm_opls = {
    "model_baseline": {
        "wer": (6.3, 6.2),
        "scales": [scales_att_ctc_opls["model_baseline"]["scales"][0] + [0.4]],
    },

    "model_ctc0.9_att0.1": {
        "scales": [],
    },
    "model_ctc0.8_att0.2": {
        "scales": [],
    },
    "model_ctc0.7_att0.3": {
        "scales": [],
    },
    "model_ctc0.6_att0.4": {
        "scales": [],
    },
    "model_ctc0.5_att0.5": {
        "scales": [scales_att_ctc_opls["model_ctc0.5_att0.5"]["scales"][0] + [0.45]],
    },
    "model_ctc0.4_att0.6": {
        "scales": [],
    },
    "model_ctc0.3_att0.7": {
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7"]["scales"][0] + [0.45]],
    },
    "model_ctc0.2_att0.8": {
        "scales": [scales_att_ctc_opls["model_ctc0.2_att0.8"]["scales"][0] + [0.45]],
    },
    "model_ctc0.1_att0.9": {
        "scales": [scales_att_ctc_opls["model_ctc0.1_att0.9"]["scales"][0] + [0.4]],
    },
    "model_ctc0.001_att0.999": {
        "scales": [scales_att_ctc_opls["model_ctc0.001_att0.999"]["scales"][0] + [0.45]],
    },

    "model_baseline_lay6": {
        "scales": [],
    },
    "model_baseline_lay8": {
        "scales": [],
    },
    "model_baseline_lay10": {
        "scales": [],
    },

    "model_ctc0.3_att0.7_lay6": {
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7_lay6"]["scales"][0] + [0.45]],
    },
    "model_ctc0.3_att0.7_lay8": {
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7_lay8"]["scales"][0] + [0.45]],
    },
    "model_ctc0.3_att0.7_lay10": {
        "scales": [scales_att_ctc_opls["model_ctc0.3_att0.7_lay10"]["scales"][0] + [0.45]],
    },

    "model_att_only_currL": {
        "scales": [],
    }
}

# att + ctc only model optsr

scales_att_ctc_only_optsr = {}

# att + ctc only model optls

scales_att_ctc_only_optls = {}

# att + ctc only model + lm optsr

scales_att_ctc_only_lm_optsr = {}

# att + ctc only model + lm opls

scales_att_ctc_only_lm_opls = {}