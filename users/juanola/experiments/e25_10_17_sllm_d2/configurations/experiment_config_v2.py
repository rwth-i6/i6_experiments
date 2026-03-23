"""
NEW V2 LLMs (SLLM vocabs)
"""
from dataclasses import replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.experiment_config import (
    exp_v14_3ctc_b_pre_ed,
    exp_v14_3ctc_s_pre_ed,
    exp_v14_3ctc_b_pre_ed_f20,
    exp_v14_3ctc_s_pre_ed_f10,
    ExperimentConfig,
    bv2_pre_ed_s_c_4gpu,
    bv2_ds_pre_ed_b_fe,
    bv2_pre_ed_s_fe,
    exp_v7,
    bv2_pre_ed_s_c,
    bv2_pre_ed_s_lm,
    exp_v15_small_12ep_lr4,
    exp_v15_small_12ep_lr5,
    bv2_ds_pre_ed_b_c,
    bv2_ds_pre_ed_b_lm,
    bv2_pre_d_s_c,
    bv2_pre_d_b_c,
    bv3_ds_pre_ed_b_c_lora,
    bv3_pre_ed_s_c_lora_small,
    bv3_pre_ed_s_c_lora,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.network_config import (
    network_small_v2_3ctc,
    network_base_v2_3ctc,
    network_with_frozen_layers,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.search_config import (
    base_searches,
    search_baseline_ctc_greedy_decoding,
    search_baseline_v2, V4_autoscaling_good_combs_1_reduced, PretrainedExternalModules,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.training_config import (
    i6_4gpu_setup_v4,
    itc_v2_80k,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pretrained_models import (
    enc_dec_small_combined_v2,
    enc_dec_base_combined_v2,
    enc_dec_small_lm_v2,
    enc_dec_base_lm_v2,
    dec_small_combined_v2,
    dec_base_combined_v2,
)


def exp_v14_3ctc_b_pre_ed_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_b_pre_ed(), pretrained=enc_dec_base_combined_v2())


def exp_v14_3ctc_s_pre_ed_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_s_pre_ed(), pretrained=enc_dec_small_combined_v2(),
                   search=[
                       *base_searches(),
                       *V4_autoscaling_good_combs_1_reduced(
                           ext_encoder=PretrainedExternalModules.CTC_STANDALONE_3_LAYERS.value,
                           ext_decoder=PretrainedExternalModules.LLM_BASE_COMBINED_V2.value
                       )
                   ])


def bv2_pre_ed_s_c_4gpu_v2() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c_4gpu(),
        pretrained=enc_dec_small_combined_v2(),
    )


def exp_v14_3ctc_b_pre_ed_f20_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_b_pre_ed_f20(),
                   pretrained=enc_dec_base_combined_v2(),
                   search=[
                       *base_searches(),
                       *V4_autoscaling_good_combs_1_reduced(
                           ext_encoder=PretrainedExternalModules.CTC_STANDALONE_3_LAYERS.value,
                           ext_decoder=PretrainedExternalModules.LLM_BASE_COMBINED_V2.value
                       )
                   ])


def exp_v14_3ctc_b_pre_ed_f40_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_b_pre_ed_f20(),
                   pretrained=enc_dec_base_combined_v2(),
                   network=network_with_frozen_layers(
                       network_base_v2_3ctc(), encoder_partial_epochs=40, decoder_partial_epochs=40
                   ),
                   search=[
                       *base_searches(),
                       *V4_autoscaling_good_combs_1_reduced(
                           ext_encoder=PretrainedExternalModules.CTC_STANDALONE_3_LAYERS.value,
                           ext_decoder=PretrainedExternalModules.LLM_BASE_COMBINED_V2.value
                       )
                   ])


def exp_v14_3ctc_s_pre_ed_f10_v2() -> ExperimentConfig:
    return replace(exp_v14_3ctc_s_pre_ed_f10(), pretrained=enc_dec_small_combined_v2())


def bv2_ds_pre_ed_b_fe_v2() -> ExperimentConfig:
    return replace(bv2_ds_pre_ed_b_fe(), pretrained=enc_dec_base_combined_v2())


def bv2_pre_ed_s_fe_v2() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_fe(), pretrained=enc_dec_small_combined_v2())


def exp_v15_3ctc_small() -> ExperimentConfig:
    return replace(exp_v7(), network=network_small_v2_3ctc(), training=i6_4gpu_setup_v4(),
                   search=[
                       *base_searches(),
                   ])


def exp_v15_3ctc_small_hpc() -> ExperimentConfig:
    return replace(exp_v15_3ctc_small(), training=itc_v2_80k())


def bv2_pre_ed_s_c_v2() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_c(), pretrained=enc_dec_small_combined_v2())


def bv2_pre_ed_s_lm_v2() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_lm(), pretrained=enc_dec_small_lm_v2(),
                   search=[
                       *base_searches(),
                       *V4_autoscaling_good_combs_1_reduced(
                           ext_encoder=PretrainedExternalModules.CTC_STANDALONE_3_LAYERS.value,
                           ext_decoder=PretrainedExternalModules.LLM_BASE_COMBINED_V2.value
                       )
                   ])


def exp_v15_small_12ep_lr4_v2() -> ExperimentConfig:
    return replace(exp_v15_small_12ep_lr4(), pretrained=enc_dec_small_combined_v2())


def exp_v15_small_12ep_lr5_v2() -> ExperimentConfig:
    return replace(exp_v15_small_12ep_lr5(), pretrained=enc_dec_small_combined_v2())


def bv2_ds_pre_ed_b_c_v2() -> ExperimentConfig:
    return replace(bv2_ds_pre_ed_b_c(), pretrained=enc_dec_base_combined_v2())


def bv2_ds_pre_ed_b_lm_v2() -> ExperimentConfig:
    return replace(bv2_ds_pre_ed_b_lm(), pretrained=enc_dec_base_lm_v2())


def bv2_pre_d_s_c_v2() -> ExperimentConfig:
    return replace(bv2_pre_d_s_c(), pretrained=dec_small_combined_v2(), training=itc_v2_80k())


def bv2_pre_d_s_c_v2_i6() -> ExperimentConfig:
    return replace(bv2_pre_d_s_c_v2(), training=i6_4gpu_setup_v4())


def bv2_pre_d_b_c_v2() -> ExperimentConfig:
    return replace(bv2_pre_d_b_c(), pretrained=dec_base_combined_v2(), training=itc_v2_80k())


def bv3_ds_pre_ed_b_c_lora_v2() -> ExperimentConfig:
    return replace(
        bv3_ds_pre_ed_b_c_lora(),
        pretrained=enc_dec_base_combined_v2(),
        training=itc_v2_80k(),
        search=[
            # *base_searches()
            search_baseline_ctc_greedy_decoding(),
            search_baseline_v2(),
        ],
    )


def bv3_pre_ed_s_c_lora_v2() -> ExperimentConfig:
    return replace(
        bv3_pre_ed_s_c_lora(),
        pretrained=enc_dec_small_combined_v2(),
        training=itc_v2_80k(),
        search=[
            # *base_searches()
            search_baseline_ctc_greedy_decoding(),
            search_baseline_v2(),
        ],
    )


def bv3_pre_ed_s_c_lora_small_v2() -> ExperimentConfig:
    return replace(
        bv3_pre_ed_s_c_lora_small(),
        pretrained=enc_dec_small_combined_v2(),
        training=itc_v2_80k(),
        search=[
            # *base_searches()
            search_baseline_ctc_greedy_decoding(),
            search_baseline_v2(),
        ],
    )


def base_3ctc_pre_ed_c_f20() -> ExperimentConfig:
    return replace(
        exp_v7(),
        pretrained=enc_dec_base_combined_v2(),
        network=network_with_frozen_layers(
            network_base_v2_3ctc(), encoder_partial_epochs=20, decoder_partial_epochs=20
        ),
        training=itc_v2_80k(),
    )


def small_3ctc_pre_ed_lm() -> ExperimentConfig:
    return replace(
        exp_v7(),
        pretrained=enc_dec_small_lm_v2(),
        network=network_small_v2_3ctc(),
        training=itc_v2_80k(),
    )
