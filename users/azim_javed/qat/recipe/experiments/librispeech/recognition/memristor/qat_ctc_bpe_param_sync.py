from dataclasses import dataclass, fields, replace
from typing import List, Optional, Tuple

from i6_core.rasr import RasrConfig
from i6_experiments.common.setups.serialization import Collection, Call, NonhashedCode, ExternalImport

from .....data.librispeech import datasets as librispeech_datasets
from .....data.librispeech import lm as librispeech_lm
from .....data.librispeech.bpe import vocab_to_bpe_size
from .....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from .....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from .....model_pipelines.common.recog import (
    post_recog_memristor_offline,
    OfflineRecogParameters,
    RecogResult,
    MemristorRecogResult,
    StreamingRecogParameters,
)
from .....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from .....model_pipelines.common.serializers import get_model_serializers
from .....model_pipelines.common.train import TrainedModel
from .....model_pipelines.qat_ctc.mem_inited.prior import compute_priors as compute_priors_memristor
from .....model_pipelines.qat_ctc.prior import compute_priors
from .....model_pipelines.qat_ctc.mem_inited.pytorch_modules import (
    QATConformerCTCModel,
    QATConformerCTCConfig,
    QATConformerCTCRecogConfig,
    QATConformerCTCRecogModel,
)

from .....model_pipelines.qat_ctc.pytorch_modules import (
    QATConformerCTCModel as BaseQATConformerCTCModel,
    QATConformerCTCConfig as BaseQATConformerCTCConfig,
)
from ..common import BaseRecogVariant, run_single_bpe_variant

from .....model_pipelines.common.memrecog import convert_model_for_memristor
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

from .....tools import synaptogen_ml_root


@dataclass
class CTCRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    bpe_trafo_lm_scale: float = 0.0
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model: TrainedModel[QATConformerCTCConfig],
    variants: Optional[List[CTCRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    correction_settings: Optional[CycleCorrectionSettings] = None,
    max_runs: Optional[int] = 5,
    memristor_prior: bool = False,
    batched_decoder: bool = False,
) -> Tuple[List[MemristorRecogResult], List[RecogResult]]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = librispeech_datasets.EVAL_SETS

    if converter_hardware_settings is None:
        converter_hardware_settings = DacAdcHardwareSettings(
            input_bits=0,
            output_precision_bits=0,
            output_range_bits=0,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )

    if pos_enc_converter_hardware_settings is None:
        pos_enc_converter_hardware_settings = DacAdcHardwareSettings(
            input_bits=0,
            output_precision_bits=0,
            output_range_bits=0,
            hardware_input_vmax=0.6,
            hardware_output_current_scaling=8020.0,
        )

    memristor_results = []
    results = []
    for variant in variants:
        cycles_variant_results = []
        for num_cycles in range(1, max_runs + 1):
            _, variant_result = _run_single_variant(
                model=model,
                variant=variant,
                corpora=corpora,
                converter_hardware_settings=converter_hardware_settings,
                pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
                correction_settings=correction_settings,
                num_cycles=num_cycles,
                memristor_prior=memristor_prior,
                batched_decoder=batched_decoder,
            )
            # variant result is a list
            cycles_variant_results.extend(variant_result)

        results.extend(cycles_variant_results)
        for corpus in corpora:
            score_corpus = librispeech_datasets.get_default_score_corpus(corpus)
            corpus_results = [
                result for result in cycles_variant_results if result.corpus_name == score_corpus.corpus_name
            ]
            memristor_results.append(
                post_recog_memristor_offline(
                    descriptor=f"{model.descriptor}_memristor_{variant.descriptor}",
                    recog_corpus=score_corpus,
                    recog_results=corpus_results,
                )
            )
    return memristor_results, results


def default_recog_variants() -> List[CTCRecogVariant]:
    return [
        default_offline_lexfree_recog_variant(),
        # memristor_eq_base_tree_recog_variant(),
        # default_offline_lexfree_lstm_recog_variant(),
        # default_offline_lexfree_trafo_recog_variant(),
        # default_offline_tree_recog_variant(),
        # memristor_eq_base_tree_recog_variant(),
        # default_offline_tree_4gram_recog_variant(),
        # default_offline_tree_lstm_recog_variant(),
        # default_offline_tree_lstm_4gram_recog_variant(),
        # default_offline_tree_trafo_recog_variant(),
        # default_offline_tree_trafo_recog_variant_gpu(),
        # default_streaming_lexfree_recog_variant(),
        # default_streaming_tree_4gram_recog_variant(),
    ] + param_sweep_tree_4gram_recog()


def param_sweep_tree_4gram_recog() -> List[CTCRecogVariant]:
    variants = []
    for lm_scale, prior_scale in [(0.7, 0.3), (0.8, 0.2), (0.8, 0.3), (0.8, 0.4), (0.8, 0.5), (0.9, 0.3)]:
        variants.append(
            CTCRecogVariant(
                descriptor=f"tree_4gram_lm{lm_scale}_p{prior_scale}",
                search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
                search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
                    collapse_repeated_labels=True,
                    word_lm_params=librispeech_lm.ArpaLmParams(scale=lm_scale),
                    score_thresholds=[18.0],
                    max_beam_sizes=[2048],
                    word_end_score_threshold=None,
                    max_word_end_beam_size=None,
                ),
                prior_scale=prior_scale,
            )
        )
    return variants


def memristor_eq_base_tree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_memristor_eq",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[18.0],
            max_beam_sizes=[1024],
            # no word_lm_params → no LM
        ),
        # word_end_score_threshold and max_word_end_beam_size default to None
    )


def default_offline_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
    )


def default_offline_lexfree_lstm_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree_bpe-LSTM",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0, 12.0],
            max_beam_sizes=[2048, 256],
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_lexfree_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree_bpe-TrafoLM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[10.0, 8.0],
            max_beam_sizes=[64, 32],
        ),
        prior_scale=0.2,
        bpe_trafo_lm_scale=0.8,
    )


def default_offline_tree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[6.0],
            max_beam_sizes=[8],
            word_end_score_threshold=0.0,
            max_word_end_beam_size=1,
        ),
    )


def default_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),  # lm scales 0.6 to 1.2
            score_thresholds=[12.0],  # TODO: Try 14, 18 # first try the thresholds
            max_beam_sizes=[256],  # Try 1024, 512, 2048
            word_end_score_threshold=0.5,  # TODO: find out, should it be None?
            max_word_end_beam_size=16,  # TODO: ^^
        ),
        prior_scale=0.2,  # 0.2 to 0.5
    )


def default_offline_tree_lstm_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_bpe-LSTM",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[12.0, 10.0],
            max_beam_sizes=[128, 64],
            word_end_score_threshold=0.6,
            max_word_end_beam_size=16,
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_lstm_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_bpe-LSTM",
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[10.0, 10.0],
            max_beam_sizes=[128, 64],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=32,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.2),
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.6,
    )


def default_offline_tree_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8),
        ),
        search_mode_params=OfflineRecogParameters(),
        prior_scale=0.2,
    )


def default_offline_tree_trafo_recog_variant_gpu() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_trafoLM_gpu",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8, use_gpu=True, use_kv_cache=False),
        ),
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
        prior_scale=0.2,
    )


def default_streaming_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_streaming_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="streaming_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0],
            max_beam_sizes=[1024],
            word_end_score_threshold=0.5,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def _get_model_serializers(
    model: TrainedModel[QATConformerCTCConfig],
    base_model: TrainedModel[BaseQATConformerCTCConfig],
    variant: CTCRecogVariant,
    memristor_prior: bool,
) -> Collection:
    checkpoint = model.get_checkpoint(variant.epoch)
    base_checkpoint = base_model.get_checkpoint(variant.epoch)
    if variant.prior_scale != 0.0:
        if memristor_prior:
            prior_file = compute_priors_memristor(
                prior_data_config=librispeech_datasets.get_default_prior_data(),
                model_config=model.model_config,
                checkpoint=checkpoint,
            )
        else:
            prior_file = compute_priors(
                prior_data_config=librispeech_datasets.get_default_prior_data(),
                model_config=base_model.model_config,
                checkpoint=base_checkpoint,
            )
    if "tree_4gram_lm0.9_p0.3" in variant.descriptor and variant.prior_scale != 0.0:
        compute_priors_memristor(
            prior_data_config=librispeech_datasets.get_default_prior_data(),
            model_config=model.model_config,
            checkpoint=checkpoint,
        )
    serializers = get_model_serializers(
        QATConformerCTCRecogModel,
        QATConformerCTCRecogConfig(
            **{f.name: getattr(model.model_config, f.name) for f in fields(model.model_config)},
            prior_file=prior_file if variant.prior_scale != 0.0 else None,
            prior_scale=variant.prior_scale,
            blank_penalty=variant.blank_penalty,
        ),
    )
    serializers.serializer_objects.insert(0, ExternalImport(synaptogen_ml_root))
    serializers.serializer_objects.insert(1, NonhashedCode("import synaptogen_ml\n"))
    serializers.serializer_objects.insert(
        2, Call(callable_name="synaptogen_ml.set_fast_inference", kwargs=[("enabled", True)])
    )
    return serializers


def _get_label_scorer_configs(model: TrainedModel[QATConformerCTCConfig], variant: CTCRecogVariant) -> List[RasrConfig]:
    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)

    label_scorer_configs = [get_no_op_label_scorer_config()]
    if variant.bpe_lstm_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_lstm_label_scorer_config(
                bpe_size=bpe_size,
                scale=variant.bpe_lstm_lm_scale,
                use_gpu=variant.search_mode_params.gpu_mem_rqmt > 0,
            )
        )
    if variant.bpe_trafo_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_transformer_label_scorer_config(
                bpe_size=bpe_size,
                use_gpu=variant.search_mode_params.gpu_mem_rqmt > 0,
                scale=variant.bpe_trafo_lm_scale,
            )
        )

    return label_scorer_configs


def _convert_model_for_memristor(
    model: TrainedModel[BaseQATConformerCTCConfig],
    variant: CTCRecogVariant,
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
) -> TrainedModel:
    memristor_model, memristor_config = convert_model_for_memristor(
        checkpoint=model.get_checkpoint(variant.epoch),
        config=model.model_config,
        model_class=BaseQATConformerCTCModel,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        num_cycles=num_cycles,
    )
    epoch = variant.epoch if variant.epoch is not None else 0
    return TrainedModel(
        descriptor=f"memristor_{model.descriptor}",
        model_config=memristor_config,
        checkpoints={epoch: memristor_model},
    )


def _run_single_variant(
    model: TrainedModel[BaseQATConformerCTCConfig],
    variant: CTCRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
    memristor_prior: bool,
    batched_decoder: bool = False,
) -> Tuple[int, List[RecogResult]]:

    memristor_model = _convert_model_for_memristor(
        model=model,
        variant=variant,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        num_cycles=num_cycles,
    )
    variant_prefix = f"num_cycle_{num_cycles}_"
    if memristor_prior and variant.prior_scale != 0.0:
        variant_prefix = f"{variant_prefix}memristor_prior_"

    if batched_decoder:
        variant_prefix = f"{variant_prefix}batched_decoder_v2_"
        variant = replace(variant, search_mode_params=replace(variant.search_mode_params, batched_decoder=True))

    variant = replace(variant, descriptor=f"{variant_prefix}{variant.descriptor}", num_cycles=num_cycles)

    return run_single_bpe_variant(
        model_descriptor=memristor_model.descriptor,
        checkpoint=memristor_model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(
            model=memristor_model, base_model=model, variant=variant, memristor_prior=memristor_prior
        ),
        label_scorer_configs=_get_label_scorer_configs(model=memristor_model, variant=variant),
        bpe_size=vocab_to_bpe_size(memristor_model.model_config.target_size - 1),
        blank_index=memristor_model.model_config.target_size - 1,
        sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 or variant.bpe_trafo_lm_scale != 0 else None,
        variant=variant,
        corpora=corpora,
    )
