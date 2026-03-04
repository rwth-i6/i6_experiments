
"""
NEW V2 LLMs (SLLM vocabs)
"""
from dataclasses import replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.experiment_config import \
    exp_v14_3ctc_b_pre_ed, exp_v14_3ctc_s_pre_ed, exp_v14_3ctc_b_pre_ed_f20, exp_v14_3ctc_s_pre_ed_f10, ExperimentConfig
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pretrained_models import \
    enc_dec_small_combined_v2, enc_dec_base_combined_v2


def exp_v14_3ctc_b_pre_ed_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_b_pre_ed(), pretrained=enc_dec_base_combined_v2())


def exp_v14_3ctc_s_pre_ed_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_s_pre_ed(), pretrained=enc_dec_small_combined_v2())

def exp_v14_3ctc_b_pre_ed_f20_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_b_pre_ed_f20(), pretrained=enc_dec_base_combined_v2())


def exp_v14_3ctc_s_pre_ed_f10_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_s_pre_ed_f10(), pretrained=enc_dec_small_combined_v2())

