import numpy as np

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.tedlium2.scales import *

model_names = scales_model_names_exclude_edge
train_scales = [train_scale[model_name] for model_name in model_names]

test_scores = {
    "mc": {
        "no_lm": {
            "opls": np.asarray([scales_att_ctc_opls[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_optsr[model_name]["wer"][1] for model_name in model_names]),
            "att": np.asarray([scales_att[model_name]["wer"][1] for model_name in model_names]),
        },
        "with_lm": {
            "opls": np.asarray([scales_att_ctc_lm_opls[model_name]["wer"][1] for model_name in model_names]),
            "att": np.asarray([scales_att_lm[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_lm_optsr[model_name]["wer"][1] for model_name in model_names]),
        }
    },
    "sc": {
        "no_lm": {
            "opls": np.asarray([scales_att_ctc_only_opls[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_only_optsr[model_name]["wer"][1] for model_name in model_names]),
        },
        "with_lm": {
            "opls": np.asarray([scales_att_ctc_only_lm_opls[model_name]["wer"][1] for model_name in model_names]),
            "optsr": np.asarray([scales_att_ctc_only_lm_optsr[model_name]["wer"][1] for model_name in model_names]),
        },
    }
}

def get_avg_imp(data1, data2, rel=True):
    N = len(data1)
    imp = 0.0
    for i in range(N):
        if rel:
            imp += ((data1[i] - data2[i]) / data1[i]) * 100
        else:
            imp += data1[i] - data2[i]
    return imp / N


def print_imp_method(method, rel=True):
    avg_imp_str = "Avg " + ("rel" if rel else "abs") + f" improvement {method} "

    imp_mc = get_avg_imp(test_scores["mc"]["no_lm"]["att"], test_scores["mc"]["no_lm"][method], rel=rel)
    print(avg_imp_str, "mc:", imp_mc, "%")

    imp_mc_lm = get_avg_imp(test_scores["mc"]["with_lm"]["att"], test_scores["mc"]["with_lm"][method], rel=rel)
    print("with lm: ", imp_mc_lm, "%")

    imp_sc = get_avg_imp(test_scores["mc"]["no_lm"]["att"], test_scores["sc"]["no_lm"][method], rel=rel)
    print(avg_imp_str, "sc:", imp_sc, "%")

    imp_sc_lm = get_avg_imp(test_scores["mc"]["with_lm"]["att"], test_scores["sc"]["with_lm"][method], rel=rel)
    print("with lm: ", imp_sc_lm, "%")

print("Relative improvements:")
print_imp_method("opls")
print()
print_imp_method("optsr")
print()
print("Absolute improvements:")
print_imp_method("opls", rel=False)
print()
print_imp_method("optsr", rel=False)


# Improvement on best systems

ls960_opls = {
    "mc": {
        "no_lm": [(2.4, 5.6), (2.4, 5.6)],
        "with_lm": [(2.2, 4.6), (2.0, 4.4)],
        "with_lm_ilm": [(2.0, 4.2), (1.9, 4.1)]
    },
    "sc": {
        "no_lm": [(2.4, 5.6), (2.3, 5.2)],
        "with_lm": [(2.2, 4.6), (2.0, 4.3)],
        "with_lm_ilm": [(2.0, 4.2), (1.9, 3.9)]
    }
}

ls960_optsr = {
    "mc": {
        "no_lm": [(2.4, 5.6), (2.4, 5.6)],
        "with_lm": [(2.2, 4.6), (2.1, 4.5)],
        "with_lm_ilm": [(2.0, 4.2), (2.0, 4.3)]
    },
    "sc": {
        "no_lm": [(2.4, 5.6), (2.3, 5.2)],
        "with_lm": [(2.2, 4.6), (2.0, 4.3)],
        "with_lm_ilm": [(2.0, 4.2), (1.9, 4.0)]
    }
}

def print_rel(data, name="mc"):
    mc_rel = 0
    count = 0
    for lm_cond in ["no_lm", "with_lm", "with_lm_ilm"]:
        # clean
        mc_rel += (data[name][lm_cond][0][0] - data[name][lm_cond][1][0]) / data[name][lm_cond][0][0]
        # other
        mc_rel += (data[name][lm_cond][0][1] - data[name][lm_cond][1][1]) / data[name][lm_cond][0][1]
        count += 2

    print(f"Relative improvement {name}: {100*mc_rel / count}%")

    return mc_rel / count

print()
print("label-sync.")
rel_mc_ls = print_rel(ls960_opls, "mc")
rel_sc_ls = print_rel(ls960_opls, "sc")
print()
print("time-sync.")
rel_mc_ts = print_rel(ls960_optsr, "mc")
rel_sc_ts = print_rel(ls960_optsr, "sc")
print()
print("Average relative improvement mc: ", 100 * (rel_mc_ls + rel_mc_ts) / 2)
print("Average relative improvement sc: ", 100 * (rel_sc_ls + rel_sc_ts) / 2)
