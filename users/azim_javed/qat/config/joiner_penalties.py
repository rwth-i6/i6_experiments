from recipe.experiments.librispeech import run_all


def py():
    models = {
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_l1_scale_0.1",
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_outliers_scale_1.0_thresholds_0.5_0.5",
    }
    run_all(
        filename="/u/azim.javed/experiments/training/qat/report_joiner_penalties_digital.txt",
        selection=models.__contains__,
    )
