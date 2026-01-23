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
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.training_config import (
    TrainingConfig,
    training_baseline,
    itc_batch_size_80k,
    itc_batch_size_150k,
    bsv2_lrv2,
    bsv2_lrv3,
    itc_batch_size_250k,
    itc_4gpu_setup_v1,
    itc_4gpu_setup_v2,
    bsv2_lrv4,
    bsv2_lrv5,
    itc_batch_size_80k_150_epochs,
    itc_batch_size_80k_200_epochs,
    itc_4gpu_setup_v3,
    training_n2_test,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pretrained_models import (
    PretrainedConfig,
    no_pretrained,
    dec_base_transcriptions,
    enc_dec_base_transcriptions,
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
        search=[search_baseline_v2()], #[search_baseline(), search_baseline_v2()], # OLD search removed from v7
    )


def exp_v7_with_ctc_gd() -> ExperimentConfig:
    return replace(exp_v7(), search=[search_baseline_with_ctc_gd(), search_baseline_v2()])


def exp_v7_with_beam() -> ExperimentConfig:
    return replace(exp_v7(), search=[search_baseline_v2_multiple_beams()])


def exp_v8_1() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv2())


def exp_v8_2() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv3())


def exp_v9() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_batch_size_250k())


def exp_v10() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_4gpu_setup_v1())


def exp_v10_2() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_4gpu_setup_v2())


def exp_v10_3() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_4gpu_setup_v3())


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
    return replace(exp_v7(), training=itc_batch_size_80k_150_epochs())


def exp_v7_200() -> ExperimentConfig:
    return replace(exp_v7(), training=itc_batch_size_80k_200_epochs())


def pre_d_t_linear_adapter() -> ExperimentConfig:
    return replace(exp_v13(), pretrained=dec_base_transcriptions())


def pre_ed_t_linear_adapter() -> ExperimentConfig:
    return replace(exp_v13(), pretrained=enc_dec_base_transcriptions())


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
