import warnings
from dataclasses import dataclass, replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.data.dataset_config import (
    DatasetConfig,
    dataset_baseline,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.data.label_config import (
    LabelConfig,
    label_baseline,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.lora_config import (
    decoder_lora_v1,
    decoder_small_lora_v1,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.network_config import (
    NetworkConfig,
    network_baseline,
    network_SLLM_dropout,
    network_SLLM_tuned_dropout,
    network_SLLM_small_decoder_td,
    network_small_linear_adapter,
    network_SLLM_tuned_dropout_v2,
    network_linear_adapter,
    network_SLLM_tuned,
    network_baseline_v2,
    network_baseline_v2_td,
    network_baseline_v2_td_linear,
    network_baseline_v2_td_linear_small,
    network_with_frozen_layers,
    network_baseline_v2_td_small,
    network_with_dec_lora,
    network_base_baseline_v3,
    network_small_baseline_v3,
    network_base_v2_3ctc,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.search_config import (
    SearchConfig,
    search_baseline,
    greedy_search,
    greedy_search_v2,
    search_baseline_with_ctc_gd,
    search_baseline_v2,
    search_baseline_ctc_decoding_11gb,
    search_baseline_v2_multiple_beams,
    search_baseline_ctc_greedy_decoding,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.training_config import (
    TrainingConfig,
    training_baseline,
    itc_batch_size_80k,
    itc_batch_size_150k,
    bsv2_lrv2,
    bsv2_lrv3,
    itc_batch_size_250k,
    i6_4gpu_setup_v1,
    i6_4gpu_setup_v2,
    bsv2_lrv4,
    bsv2_lrv5,
    itc_batch_size_80k_150_epochs,
    itc_batch_size_80k_200_epochs,
    i6_4gpu_setup_v3,
    training_n2_test,
    itc_v2_80k_300_epochs,
    itc_v2_80k,
    i6_4gpu_setup_v4,
    i6_4gpu_setup_v4_for_n_epochs,
    itc_v2_80k_200_epochs,
    finetuning_v1_lr4,
    finetuning_v2_lr5,
    finetuning_v3_ca_lr4,
    finetuning_v4_ca_lr5, finetuning_v3_ca_lr4_i6, finetuning_v4_ca_lr5_i6,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pretrained_models import (
    PretrainedConfig,
    no_pretrained,
    dec_base_transcriptions,
    enc_dec_base_transcriptions,
    dec_small_combined,
    dec_base_combined,
    enc_dec_small_combined,
    enc_dec_base_combined,
    load_SLLM_pretrained_ed_s_c_f2_oclr1,
    enc_dec_base_lm,
    enc_dec_small_lm,
)


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Experiment configuration base dataclass. Contains all parameters needed for the experiment.

    Can contain default values.
    """

    # general: GeneralParams # add if needed
    dataset: DatasetConfig
    labels: LabelConfig

    network: NetworkConfig

    training: TrainingConfig
    search: list[SearchConfig]

    # Perhaps should be in network. but for now here
    pretrained: PretrainedConfig = no_pretrained()


"""
Specific configurations set below.
"""


def exp_baseline() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_baseline(),
        training=training_baseline(),
        search=[search_baseline(), search_baseline_v2()],
    )


def exp_v2() -> ExperimentConfig:
    return replace(exp_baseline(), network=network_SLLM_dropout())


def exp_v3() -> ExperimentConfig:
    warnings.warn(
        "[BUG] Doesn't use DROPOUT DROPOUT + intermediate_size too large",
        DeprecationWarning,
        stacklevel=2,
    )
    return replace(exp_baseline(), network=network_SLLM_tuned_dropout())


def exp_v3_2() -> ExperimentConfig:
    return replace(exp_baseline(), network=network_SLLM_tuned_dropout_v2())


def exp_v4() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_SLLM_small_decoder_td(),  # !!
        training=itc_batch_size_80k(),  # !!
        search=[search_baseline(), search_baseline_v2()],
    )


def exp_v5() -> ExperimentConfig:
    return replace(exp_v4(), network=network_small_linear_adapter())


def exp_v6() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_batch_size_150k())


def exp_v7() -> ExperimentConfig:
    """
    V4 but with proper decoder parameters
    :return:
    """
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_SLLM_tuned_dropout_v2(),  # !!
        training=itc_batch_size_80k(),
        search=[search_baseline_v2()],  # [search_baseline(), search_baseline_v2()], # OLD search removed from v7
    )


def exp_v7_with_ctc_gd() -> ExperimentConfig:
    return replace(
        exp_v7(),
        # search=[search_baseline(), search_baseline_ctc_greedy_decoding(), search_baseline_ctc_decoding_11gb()])
        search=[search_baseline_ctc_decoding_11gb()],
    )


def exp_v7_with_beam() -> ExperimentConfig:
    return replace(exp_v7(), search=[search_baseline_v2_multiple_beams()])


def exp_v8_1() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv2())


def exp_v8_2() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv3())


def exp_v9() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_batch_size_250k())


def exp_v10() -> ExperimentConfig:
    return replace(exp_v4(), training=i6_4gpu_setup_v1())


def exp_v10_2() -> ExperimentConfig:
    return replace(exp_v4(), training=i6_4gpu_setup_v2())


def exp_v10_3() -> ExperimentConfig:
    return replace(exp_v4(), training=i6_4gpu_setup_v3())


def exp_v10_3_s2() -> ExperimentConfig:
    return replace(
        exp_v4(),
        network=network_baseline_v2_td_small(),  # TO use new v2 model
        training=replace(i6_4gpu_setup_v3(), random_seed=1234),
    )


def exp_v11() -> ExperimentConfig:
    """
    SLLM-td-80k but without tuned params, only dropout
    :return:
    """
    return replace(exp_v7(), network=network_SLLM_dropout())


def exp_v12() -> ExperimentConfig:
    """
    SLLM-td-80k but without dropout, only tuned params
    :return:
    """
    return replace(exp_v7(), network=network_SLLM_tuned())


def exp_v13() -> ExperimentConfig:
    """
    SLLM-td-80k with linear adapter
    :return:
    """
    return replace(exp_v7(), network=network_linear_adapter())


def exp_v2_s2() -> ExperimentConfig:
    return replace(exp_v2(), training=training_baseline(seed=1234))


def exp_v8_3() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv4())


def exp_v8_4() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv5())


def exp_v7_150() -> ExperimentConfig:
    return replace(
        exp_v7(),
        training=itc_batch_size_80k_150_epochs()
        # , search=[search_baseline_v2(), search_baseline_ctc_greedy_decoding()])
        # , search=[search_baseline_ctc_greedy_decoding()])
        ,
        search=[search_baseline_ctc_decoding_11gb()],
    )


def exp_v7_200() -> ExperimentConfig:
    return replace(
        exp_v7(),
        training=itc_batch_size_80k_200_epochs()
        # , search=[search_baseline_v2(), search_baseline_ctc_greedy_decoding()])
        # , search=[search_baseline_ctc_greedy_decoding(), search_baseline_ctc_decoding_11gb()])
        ,
        search=[search_baseline_ctc_decoding_11gb()],
    )


"""
MODEL V2
"""


def model_v2_baseline() -> ExperimentConfig:
    """
    <ul>
        <li>Model V2</li>
        <li>Forward step 2</li>
        <li>Dropout + v2 params</li>
        <li>no downsampling / linear adapter</li>
    </ul>
    """
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_baseline_v2_td_linear(),  # !!
        training=itc_batch_size_80k(),
        search=[search_baseline_v2()],  # !!
    )


def model_v2_baseline_with_ds() -> ExperimentConfig:
    """
    exp 7 with new v2 model
    :return:
    """
    return replace(model_v2_baseline(), network=network_baseline_v2_td())


def model_v2_small_baseline() -> ExperimentConfig:
    return replace(model_v2_baseline(), network=network_baseline_v2_td_linear_small())


def exp_v7_300() -> ExperimentConfig:
    return replace(
        model_v2_baseline_with_ds(),
        training=itc_v2_80k_300_epochs(),
        search=[search_baseline_v2(), search_baseline_ctc_greedy_decoding(), search_baseline_ctc_decoding_11gb()],
    )


"""
Pretrained models
"""


def bv2_pre_d_b_t() -> ExperimentConfig:
    return replace(model_v2_baseline(), pretrained=dec_base_transcriptions())


def bv2_pre_ed_b_t() -> ExperimentConfig:
    return replace(model_v2_baseline(), pretrained=enc_dec_base_transcriptions())


def bv2_pre_d_s_c() -> ExperimentConfig:
    return replace(model_v2_small_baseline(), pretrained=dec_small_combined())


def bv2_pre_d_b_c() -> ExperimentConfig:
    return replace(model_v2_baseline(), pretrained=dec_base_combined())


def bv2_pre_ed_s_c() -> ExperimentConfig:
    return replace(
        model_v2_small_baseline(),
        pretrained=enc_dec_small_combined(),
        training=itc_v2_80k(),
        search=[search_baseline_v2(), search_baseline_ctc_greedy_decoding(), search_baseline_ctc_decoding_11gb()],
    )


def bv2_pre_ed_b_c() -> ExperimentConfig:
    return replace(
        model_v2_baseline(),
        pretrained=enc_dec_base_combined(),
        training=itc_v2_80k(),
        search=[search_baseline_v2(), search_baseline_ctc_greedy_decoding(), search_baseline_ctc_decoding_11gb()],
    )


# +++


def bv2_ds_pre_d_b_c() -> ExperimentConfig:
    return replace(model_v2_baseline_with_ds(), pretrained=dec_base_combined(), training=itc_v2_80k())


def bv2_ds_pre_ed_b_c() -> ExperimentConfig:
    return replace(model_v2_baseline_with_ds(), pretrained=enc_dec_base_combined(), training=itc_v2_80k())


def bv2_ds_pre_d_b_c_f2() -> ExperimentConfig:
    return replace(bv2_ds_pre_d_b_c(), network=network_with_frozen_layers(network_baseline_v2_td(), decoder_epochs=2))


def bv2_ds_pre_ed_b_c_f1() -> ExperimentConfig:
    return replace(
        bv2_ds_pre_ed_b_c(),
        network=network_with_frozen_layers(network_baseline_v2_td(), encoder_epochs=1, decoder_epochs=1),
    )


# +++


def bv2_ds_pre_ed_b_lm() -> ExperimentConfig:
    return replace(model_v2_baseline_with_ds(), pretrained=enc_dec_base_lm(), training=itc_v2_80k())


def bv2_pre_ed_s_lm() -> ExperimentConfig:
    return replace(model_v2_small_baseline(), pretrained=enc_dec_small_lm(), training=itc_v2_80k())


"""
Pretrained frozen
"""


def bv2_pre_d_s_c_f1() -> ExperimentConfig:
    return replace(
        bv2_pre_d_s_c(),
        training=i6_4gpu_setup_v4(),
        network=network_with_frozen_layers(network_baseline_v2_td_linear_small(), decoder_epochs=1),
    )


def bv2_pre_d_s_c_f2() -> ExperimentConfig:
    return replace(
        bv2_pre_d_s_c(),
        training=i6_4gpu_setup_v4(),
        network=network_with_frozen_layers(network_baseline_v2_td_linear_small(), decoder_epochs=2),
    )


def bv2_pre_d_s_c_f5() -> ExperimentConfig:
    return replace(
        bv2_pre_d_s_c(),
        training=i6_4gpu_setup_v4(),
        network=network_with_frozen_layers(network_baseline_v2_td_linear_small(), decoder_epochs=5),
    )


def bv2_pre_ed_s_c_f1() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c(),
        training=i6_4gpu_setup_v4(),
        network=network_with_frozen_layers(network_baseline_v2_td_linear_small(), encoder_epochs=1, decoder_epochs=1),
    )


def bv2_pre_ed_s_c_f2() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c(),
        training=i6_4gpu_setup_v4(),
        network=network_with_frozen_layers(network_baseline_v2_td_linear_small(), encoder_epochs=2, decoder_epochs=2),
    )


def bv2_pre_ed_s_c_f2_oclr1() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c(),
        training=i6_4gpu_setup_v4_for_n_epochs(2),
        network=network_with_frozen_layers(network_baseline_v2_td_linear_small(), encoder_epochs=2, decoder_epochs=2),
    )


def bv2_pre_ed_s_c_f2_oclr2() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c(),
        pretrained=load_SLLM_pretrained_ed_s_c_f2_oclr1(),
        training=i6_4gpu_setup_v4_for_n_epochs(98),
        network=network_baseline_v2_td_linear_small(),
    )


def bv2_with_ds_pre_d_s_c_f1() -> ExperimentConfig:
    return replace(
        bv2_pre_d_s_c(),
        training=i6_4gpu_setup_v3(),
        network=network_with_frozen_layers(network_baseline_v2_td_small(), decoder_epochs=1),
    )


def SLLM_small_linear_4gpu_10k() -> ExperimentConfig:
    return replace(
        model_v2_small_baseline(),
        training=i6_4gpu_setup_v4(),
    )


def SLLM_small_linear_4gpu_10k_pre_d() -> ExperimentConfig:
    return replace(model_v2_small_baseline(), training=i6_4gpu_setup_v4(), pretrained=dec_small_combined())


def exp_v13_200() -> ExperimentConfig:
    """
    SLLM-td-80k with linear adapter
    :return:
    """
    return replace(exp_v7(), network=network_linear_adapter(), training=itc_v2_80k_200_epochs())


def exp_v14_3ctc() -> ExperimentConfig:
    """
    SLLM-td-80k with linear adapter
    :return:
    """
    return replace(exp_v7(), network=network_base_v2_3ctc(), training=itc_v2_80k())


def exp_v15_small_12ep_lr4() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_c(), training=finetuning_v1_lr4(), search=[search_baseline_v2()])


def exp_v15_small_12ep_lr5() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_c(), training=finetuning_v2_lr5(), search=[search_baseline_v2()])


def exp_v15_small_12ep_ca_lr4() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_c(), training=finetuning_v3_ca_lr4(), search=[search_baseline_v2()])


def exp_v15_small_12ep_ca_lr5() -> ExperimentConfig:
    return replace(bv2_pre_ed_s_c(), training=finetuning_v4_ca_lr5(), search=[search_baseline_v2()])


def exp_v15_small_12ep_ca_lr4_i6() -> ExperimentConfig:
    return replace(exp_v15_small_12ep_ca_lr4(), training=finetuning_v3_ca_lr4_i6())


def exp_v15_small_12ep_ca_lr5_i6() -> ExperimentConfig:
    return replace(exp_v15_small_12ep_ca_lr5(), training=finetuning_v4_ca_lr5_i6())


def exp_v7_s2() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_baseline_v2_td(),
        training=replace(itc_v2_80k(), random_seed=1234),
        search=[search_baseline_v2()],
    )


def exp_v5_s2() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_baseline_v2_td_linear_small(),
        training=replace(itc_v2_80k(), random_seed=1234),
        search=[search_baseline_v2()],
    )


"""
Pretrained LORA
"""


def bv3_ds_pre_ed_b_c_lora() -> ExperimentConfig:
    return replace(bv2_ds_pre_ed_b_c(), network=network_with_dec_lora(network_base_baseline_v3(), decoder_lora_v1()))


def bv3_pre_ed_s_c_lora() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c(),
        network=network_with_dec_lora(network_small_baseline_v3(), decoder_lora_v1()),
        search=[search_baseline_v2()],
    )


def bv3_pre_ed_s_c_lora_small() -> ExperimentConfig:
    return replace(
        bv2_pre_ed_s_c(),
        network=network_with_dec_lora(network_small_baseline_v3(), decoder_small_lora_v1()),
        search=[search_baseline_v2()],
    )


"""
Tests
"""


def t_v1() -> ExperimentConfig:
    return replace(exp_v8_2(), search=[greedy_search()])


def t_v1_2() -> ExperimentConfig:
    return replace(exp_v8_2(), search=[greedy_search_v2()])


def n2_test() -> ExperimentConfig:
    return replace(
        exp_v7(),
        training=training_n2_test(),
        network=network_baseline_v2(),
        search=[search_baseline_ctc_decoding_11gb()],
    )


def n2_test_sv2() -> ExperimentConfig:
    return replace(exp_v7(), training=training_n2_test(), network=network_baseline_v2(), search=[search_baseline_v2()])


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
