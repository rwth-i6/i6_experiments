import dataclasses
import dataclasses
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, List

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.data.label_config import label_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.network_config import (
    NetworkConfig,
    network_baseline_v2_td,
    network_base_v2_3ctc,
    network_baseline_v2_td_linear_small,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.beam_search_config import (
    BeamSearchConfig,
    beam_search_baseline,
    greedy,
    beam_search_multiple_beams,
    single_beam, beam_search_multiple_beams_v2, beam_search_multiple_beams_v3, beam_search_multiple_beams_v4,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.prior_config import (
    prior_v1,
    PriorConfig,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.experiments_core.tuning.scales import Scales


@dataclass(frozen=True)
class SearchConfig:
    """
    Search (inference) configuration base dataclass.

    Can contain default values.
    """

    batch_size: int
    batch_size_factor: int
    use_gpu: bool
    gpu_memory: int  # Avoid using bigger that 11Gb
    avg_best_loss_name: str
    max_seqs: int

    cpu_memory: int

    prior: Optional[PriorConfig]

    # Tunable Parameters # TODO: this could be grouped...
    beam_search: BeamSearchConfig

    lm_scales: list[float]
    prior_scales: list[float]
    ctc_scales: list[float]
    sllm_scales: list[float]

    length_norm_exponent: float = None # backwards compatibility (in reality it is 1.0), # Only for V2 recog for now

    # Other
    forward_method: str = None

    # What to run
    auto_scaling: bool = False
    sllm_as_llm: bool = False  # SLLM LM is used as the external LM (no ext_decoder needed)
    auto_scaling_use_ctc_sum_scores: bool = False
    prior_relative_to: Optional[str] = None
    run_ctc_greedy_decoding_last_epoch: bool = False

    debug_returnn_param: bool = True

    # External modules
    ext_encoder: dict[str, Any] = None
    ext_decoder: dict[str, Any] = None

    ext_decoder_no_preloading: bool = False  # TODO: remove, only for test

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        if self.use_gpu:
            assert self.gpu_memory is not None, "if use_gpu is set please set gpu_memory variable."


"""
PRETIRAINED PAIRS
"""


class PretrainedExternalModules(Enum):

    # Encoders
    CTC_STANDALONE_2_LAYERS = {
        "checkpoint_key": "ctc_v1",
        "network_config": network_baseline_v2_td(),
        "label_config": label_baseline(),
    }
    CTC_STANDALONE_3_LAYERS = {
        "checkpoint_key": "ctc_v1-3",
        "network_config": network_base_v2_3ctc(),
        "label_config": label_baseline(),
    }

    # Decoders
    LLM_BASE_TRANSCRIPTIONS = {
        "checkpoint_key": "llm_base_transcriptions",
        "network_config": network_baseline_v2_td(),
        "label_config": label_baseline(),
    }
    LLM_SMALL_TRANSCRIPTIONS = {
        "checkpoint_key": "llm_small_transcriptions",
        "network_config": network_baseline_v2_td(),
        "label_config": label_baseline(),
    }
    LLM_BASE_COMBINED_V2 = {
        "checkpoint_key": "llm_base_combined_v2",
        "network_config": network_baseline_v2_td(),
        "label_config": label_baseline(),
    }
    LLM_SMALL_COMBINED_V2 = {
        "checkpoint_key": "llm_small_combined_v2",
        "network_config": network_baseline_v2_td_linear_small(),
        "label_config": label_baseline(),
    }

    LLM_BASE_COMBINED = {  # TODO: broken - different vocab!!
        "checkpoint_key": "llm_base_combined",
        "network_config": network_baseline_v2_td(),
        "label_config": label_baseline(),
    }
    LLM_SMALL_COMBINED = {  # TODO: broken - different vocab!!
        "checkpoint_key": "llm_small_combined",
        "network_config": network_baseline_v2_td_linear_small(),
        "label_config": label_baseline(),
    }


"""
parameter sets
"""

_LM_PRIOR_SCALES = dict(
    lm_scales=[2.0, 2.2, 2.4, 2.6, 2.8],
    prior_scales=[0.7, 0.9],
)

"""
Specific configurations set below.
"""


def base_searches():
    return search_baseline_ctc_greedy_decoding(), search_baseline_v2(), V4_CTC_SLLM(), V4_CTC_SLLM_autoscaling()


"""
V1
"""


def search_baseline() -> SearchConfig:
    """
    V2 should be used!
    :return:
    """
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return SearchConfig(
        batch_size=15_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        cpu_memory=12,
        beam_search=beam_search_baseline(),
        prior=None,
        lm_scales=[0.0],
        prior_scales=[0.0],
        ctc_scales=[0.0],
        sllm_scales=[None],  # Not used!
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
    )


def search_baseline_with_ctc_gd() -> SearchConfig:
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataclasses.replace(search_baseline(), run_ctc_greedy_decoding_last_epoch=True)


def greedy_search() -> SearchConfig:
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataclasses.replace(search_baseline(), beam_search=greedy())


def greedy_search_v2() -> SearchConfig:
    warnings.warn(
        "[BUG] Doesn't use beam search correctly -> essentially greedy",
        DeprecationWarning,
        stacklevel=2,
    )
    return dataclasses.replace(search_baseline(), batch_size=13_000, beam_search=greedy())


"""
V2
"""


def search_baseline_v2() -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_v2",
        batch_size=15_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        cpu_memory=12,
        beam_search=beam_search_baseline(),
        prior=None,
        lm_scales=[None],  # Not used!
        prior_scales=[None],  # Not used!
        ctc_scales=[None],  # Not used!
        sllm_scales=[None],  # Not used!
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
    )


def search_baseline_v2_multiple_beams() -> SearchConfig:
    return dataclasses.replace(search_baseline_v2(), beam_search=beam_search_multiple_beams())

def search_baseline_v2_multiple_beams_v2() -> SearchConfig:
    return dataclasses.replace(search_baseline_v2(), beam_search=beam_search_multiple_beams_v2())

def search_baseline_v2_multiple_beams_v3() -> SearchConfig:
    return dataclasses.replace(search_baseline_v2(), beam_search=beam_search_multiple_beams_v3())

def search_baseline_v2_multiple_beams_v4() -> SearchConfig:
    return dataclasses.replace(search_baseline_v2(), beam_search=beam_search_multiple_beams_v4(), batch_size=5000)

def search_v4_ctc_sllm_multiple_beams() -> SearchConfig:
    return dataclasses.replace(V4_CTC_SLLM(), beam_search=beam_search_multiple_beams())

def search_v4_ctc_sllm_multiple_beams_autoscale() -> SearchConfig:
    return dataclasses.replace(V4_CTC_SLLM_autoscaling(), beam_search=beam_search_multiple_beams())

def search_v4_ctc_sllm_multiple_beams_autoscale_v3() -> SearchConfig:
    return dataclasses.replace(V4_CTC_SLLM_autoscaling(), beam_search=beam_search_multiple_beams_v3())

def search_v4_ctc_sllm_multiple_beams_autoscale_v4() -> SearchConfig:
    return dataclasses.replace(V4_CTC_SLLM_autoscaling(), beam_search=BeamSearchConfig(beam_sizes=[64, 128]), batch_size=5000)

def search_v4_ctc_sllm_multiple_beams_autoscale_v5() -> SearchConfig:
    return dataclasses.replace(V4_CTC_SLLM_autoscaling(), beam_search=BeamSearchConfig(beam_sizes=[64]), batch_size=2000)

"""
ctc decoding
"""


def V3_search_baseline_ctc_decoding_11gb() -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_ctc_decoding",
        batch_size=5_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        cpu_memory=12,
        beam_search=beam_search_baseline(),
        prior=None,
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
        lm_scales=[1.0],
        ctc_scales=[1.0],
        prior_scales=[0.0],
        sllm_scales=[None],  # Not used!
    )


def V3_search_baseline_ctc_decoding_24gb() -> SearchConfig:
    return dataclasses.replace(V3_search_baseline_ctc_decoding_11gb(), batch_size=10_000, gpu_memory=24)


"""
ctc decoding v2 (with external modules)
"""


def V4_baseline(
    ext_encoder: Optional[tuple[str, NetworkConfig]] = None, ext_decoder: Optional[tuple[str, NetworkConfig]] = None
) -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_ctc_decoding_v2",
        batch_size=5_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,  # TODO: perhaps increase this
        cpu_memory=12,
        beam_search=beam_search_baseline(),
        prior=prior_v1(),
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
        lm_scales=[1.0],
        sllm_scales=[1.0],
        ctc_scales=[1.0],
        prior_scales=[0.0],
        ext_encoder=ext_encoder,
        ext_decoder=ext_decoder,
    )


def V4_CTC_SLLM() -> SearchConfig:
    return dataclasses.replace(
        V4_baseline(),
        lm_scales=[0.0],
        sllm_scales=[1.0],
        ctc_scales=[1.0],
        prior_scales=[0.0],
    )


def V4_ctc_sllm_lm_combinations(
    ext_encoder: Optional[tuple[str, NetworkConfig]] = None, ext_decoder: Optional[tuple[str, NetworkConfig]] = None
) -> SearchConfig:
    return dataclasses.replace(
        V4_baseline(),
        lm_scales=[0.0, 1.0],
        sllm_scales=[0.0, 1.0],
        ctc_scales=[1.0],
        ext_encoder=ext_encoder,
        ext_decoder=ext_decoder,
    )


TO_TUNE_SCALE_FOR_AUTOSCALING = 0.0

def V4_autoscaling_64_ctc_prior_sllm_lm(
    ext_encoder: Optional[tuple[str, NetworkConfig]] = None,
    ext_decoder: Optional[tuple[str, NetworkConfig]] = None,
    use_ctc: bool = True,
    use_sllm: bool = True,
    use_llm: bool = True,
    use_prior: bool = True,
    auto_scaling_use_ctc_sum_scores: bool = False,
    prior_relative_to: Optional[str] = None,
) -> SearchConfig:
    """
    For autoscaling:
    - None -> don't use
    - zero -> to tune
    -> positive number -> fix scale
    """

    ctc_scales = [TO_TUNE_SCALE_FOR_AUTOSCALING] if use_ctc else [None]
    sllm_scales = [TO_TUNE_SCALE_FOR_AUTOSCALING] if use_sllm else [None]
    llm_scales = [TO_TUNE_SCALE_FOR_AUTOSCALING] if use_llm else [None]
    prior_scales = [TO_TUNE_SCALE_FOR_AUTOSCALING] if use_prior else [None]

    # Frozen scales
    if use_ctc:
        ctc_scales = [1.0]
    elif use_sllm:
        sllm_scales = [1.0]

    sllm_as_llm = use_llm and (ext_decoder is None)

    return dataclasses.replace(
        V4_baseline(),
        batch_size=5_000,  # TODO_??? # Tested: 3000, 5000(failed last recog)
        ext_encoder=ext_encoder,
        ext_decoder=ext_decoder,
        lm_scales=llm_scales,
        sllm_scales=sllm_scales,
        ctc_scales=ctc_scales,
        prior_scales=prior_scales,
        auto_scaling=True,
        beam_search=single_beam(12),
        sllm_as_llm=sllm_as_llm,
        auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
        prior_relative_to=prior_relative_to,
    )

def V4_CTC_SLLM_autoscaling():
    return V4_autoscaling_64_ctc_prior_sllm_lm(
        use_ctc = True,
        use_sllm = True,
        use_llm = False,
        use_prior = False,
        auto_scaling_use_ctc_sum_scores = True,
    )


def V4_autoscaling_64_all_combs(
    ext_encoder: Optional[tuple[str, NetworkConfig]] = None,
    ext_decoder: Optional[tuple[str, NetworkConfig]] = None,
    auto_scaling_use_ctc_sum_scores: bool = False,
    force_ext_llm: bool = False,
    force_prior:bool = False,
    prior_relative_to: Optional[str] = None,
) -> List[SearchConfig]:
    searches = []

    opts = [True, False]
    opts_llm = [True] if force_ext_llm else opts
    opts_prior = [True] if force_prior else opts
    for (ctc, sllm, llm, prior) in [(a, b, c, d) for a in opts for b in opts for c in opts_llm for d in opts_prior]:
        if ctc + sllm + llm + prior <= 1:
            continue  # At least 2 models/components
        if not ctc and prior:
            continue  # Prior only for CTC
        if prior and not (llm or sllm):
            continue  # Prior should be used if an LM is also used

        prior_relative_to_aux = prior_relative_to
        if prior_relative_to == Scales.LLM.value and not llm:
            if sllm:
                prior_relative_to_aux = Scales.SLLM.value
            else:
                prior_relative_to_aux = None
        elif prior_relative_to == Scales.SLLM.value and not sllm:
            if llm:
                prior_relative_to_aux = Scales.LLM.value
            else:
                prior_relative_to_aux = None

        searches.append(
            V4_autoscaling_64_ctc_prior_sllm_lm(
                ext_encoder=ext_encoder,
                ext_decoder=ext_decoder,
                use_ctc=ctc,
                use_sllm=sllm,
                use_llm=llm,
                use_prior=prior,
                auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
                prior_relative_to=prior_relative_to_aux,
            )
        )
    return searches


def V4_autoscaling_64_ext_llm_combs(
    ext_encoder: Optional[tuple[str, NetworkConfig]] = None,
    ext_decoder: Optional[tuple[str, NetworkConfig]] = None,
    auto_scaling_use_ctc_sum_scores: bool = False,
    prior_relative_to: Optional[str] = None,
) -> List[SearchConfig]:
    """
    Forces ext LLM to be used
    :param ext_encoder:
    :param ext_decoder:
    :param auto_scaling_use_ctc_sum_scores:
    :return:
    """
    return V4_autoscaling_64_all_combs(
        ext_encoder=ext_encoder,
        ext_decoder=ext_decoder,
        auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
        force_ext_llm=True,
        prior_relative_to=prior_relative_to,
    )



def V4_autoscaling_good_combs_1(
        ext_encoder: Optional[tuple[str, NetworkConfig]] = None,
        ext_decoder: Optional[tuple[str, NetworkConfig]] = None,
        auto_scaling_use_ctc_sum_scores: bool = True,
        prior_relative_to: Optional[str] = Scales.LLM.value,
        force_prior: bool = False,
) -> List[SearchConfig]:
    configs = []

    # Base combs + SLLM as LM
    configs.extend(V4_autoscaling_64_all_combs(
        # Finetuned CTC
        # SLLM as LLM
        auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
        prior_relative_to=prior_relative_to,
        force_prior=force_prior,
    ))

    if ext_decoder is None:
        return configs

    # Base combs + ext LM
    configs.extend(V4_autoscaling_64_all_combs(
        # Finetuned CTC
        ext_decoder=ext_decoder,
        auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
        prior_relative_to=prior_relative_to,
        force_prior=force_prior,
    ))

    if ext_encoder is None:
        return configs

    # ext CTC + ext LM
    configs.extend(V4_autoscaling_64_all_combs(
        ext_encoder=ext_encoder,
        ext_decoder=ext_decoder,
        auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
        prior_relative_to=prior_relative_to,
        force_prior=force_prior,
    ))

    # Opt... ext CTC, SLLM as LM -> not tested

    return configs


def V4_autoscaling_good_combs_1_reduced(
        ext_encoder: Optional[tuple[str, NetworkConfig]] = None,
        ext_decoder: Optional[tuple[str, NetworkConfig]] = None,
        auto_scaling_use_ctc_sum_scores: bool = True,
        prior_relative_to: Optional[str] = Scales.LLM.value,
) -> List[SearchConfig]:
    """
    Forces prior to reduce in half approx the combinations
    """
    return V4_autoscaling_good_combs_1(
        ext_encoder=ext_encoder,
        ext_decoder=ext_decoder,
        auto_scaling_use_ctc_sum_scores=auto_scaling_use_ctc_sum_scores,
        prior_relative_to=prior_relative_to,
        force_prior=True,
    )




"""
ctc greedy
"""


def search_baseline_ctc_greedy_decoding() -> SearchConfig:
    return SearchConfig(
        forward_method="forward_step_greedy_ctc",
        batch_size=15_000,
        batch_size_factor=160,
        use_gpu=True,
        gpu_memory=11,
        cpu_memory=12,
        beam_search=beam_search_baseline(),
        prior=None,
        lm_scales=[None],  # Not used!
        prior_scales=[None],  # Not used!
        ctc_scales=[None],  # Not used!
        sllm_scales=[None],  # Not used!
        avg_best_loss_name="dev_loss_ce",
        max_seqs=200,
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
