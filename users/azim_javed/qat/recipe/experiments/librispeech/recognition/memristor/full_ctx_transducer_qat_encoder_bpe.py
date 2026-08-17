from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Type

from i6_core.rasr import RasrConfig

from i6_experiments.common.setups.serialization import Collection, Call, NonhashedCode, ExternalImport

from .....data.librispeech import datasets as librispeech_datasets
from .....data.librispeech import lm as librispeech_lm
from .....data.librispeech.bpe import vocab_to_bpe_size
from .....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from .....model_pipelines.common.recog import OfflineRecogParameters, RecogResult, StreamingRecogParameters
from .....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from .....model_pipelines.common.serializers import get_model_serializers
from .....model_pipelines.common.train import TrainedModel
from .....model_pipelines.full_ctx_transducer_qat_encoder.mem_inited.label_scorer_config import (
    get_lstm_transducer_label_scorer_config,
)
from .....model_pipelines.full_ctx_transducer_qat_encoder.mem_inited.pytorch_modules import (
    LstmTransducerQATEncoderConfig,
    LstmTransducerQATEncoderEncoder,
)

from .....model_pipelines.full_ctx_transducer_qat_encoder.pytorch_modules import (
    LstmTransducerQATEncoderConfig as BaseLstmTransducerQATEncoderConfig,
    LstmTransducerQATEncoderEncoder as BaseLstmTransducerQATEncoderEncoder,
    LstmTransducerQATEncoderModel as BaseLstmTransducerQATEncoderModel,
)
from ..common import BaseRecogVariant, post_recog_memristor_results, run_single_bpe_variant

from i6_models.config import ModelConfiguration, ModuleType

from .....model_pipelines.common.memrecog import convert_model_for_memristor
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

from .....tools import synaptogen_ml_root


@dataclass
class TransducerRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    ilm_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model: TrainedModel[LstmTransducerQATEncoderConfig],
    variants: Optional[List[TransducerRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    correction_settings: Optional[CycleCorrectionSettings] = None,
    max_runs: Optional[int] = 5,
    batched_decoder: bool = False,
) -> List[RecogResult]:
    # max_runs = 1  # TODO: debug
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
                batched_decoder=batched_decoder,
            )
            # variant result is a list
            cycles_variant_results.extend(variant_result)

        results.extend(cycles_variant_results)
        memristor_results.extend(
            post_recog_memristor_results(
                descriptor=f"{model.descriptor}_memristor_{variant.descriptor}",
                corpora=corpora,
                recog_results=cycles_variant_results,
            )
        )
    return memristor_results, results


def default_recog_variants() -> List[TransducerRecogVariant]:
    return (
        [
            default_offline_lexfree_recog_variant(),
            # default_offline_lexfree_lstm_recog_variant(),
            # default_offline_tree_recog_variant(),
            # default_offline_tree_4gram_recog_variant(),
            # default_offline_tree_lstm_recog_variant(),
            mbs1024_offline_tree_recog_variant(),
            # mbs2048_offline_tree_recog_variant(),
            # default_offline_tree_lstm_4gram_recog_variant(),
            # default_offline_tree_trafo_recog_variant(),
            # default_offline_tree_trafo_recog_variant_gpu(),
            # default_streaming_lexfree_recog_variant(),
            # default_streaming_tree_4gram_recog_variant(),
            # ]
        ]
        + param_sweep_tree_4gram_recog_variants()
        # + param_sweep_tree_lstm_recog_variants()
        # + param_sweep_lexfree_lstm_recog_variants()
    )


def param_sweep_tree_4gram_recog_variants() -> List[TransducerRecogVariant]:
    variants = []
    params = [
        (0.1, 0.6),
        # (0.1, 0.8),
        # (0.1, 0.9),
        (0.2, 0.6),
        (0.2, 0.7),
        (0.2, 0.8),
        (0.2, 0.8),
        (0.3, 0.5),
        (0.3, 0.6),
        (0.3, 0.7),
        (0.4, 0.6),
    ]
    for ilm_scale, ext_lm_scale in params:
        variants.append(
            TransducerRecogVariant(
                descriptor=f"tree_4gram_ilm{ilm_scale}_elm{ext_lm_scale}",
                search_mode_params=OfflineRecogParameters(
                    gpu_mem_rqmt=11, mem_rqmt=32, dataloader_num_workers=0, batch_size_seconds=360
                ),
                search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
                    collapse_repeated_labels=False,
                    word_lm_params=librispeech_lm.ArpaLmParams(scale=ext_lm_scale),
                    max_beam_sizes=[2048],
                    score_thresholds=[18.0],
                    word_end_score_threshold=None,
                    max_word_end_beam_size=None,
                ),
                ilm_scale=ilm_scale,
            )
        )
        # break
    return variants


def param_sweep_lexfree_lstm_recog_variants() -> List[TransducerRecogVariant]:
    variants = []
    for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for ext_lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            variants.append(
                TransducerRecogVariant(
                    descriptor=f"lexfree_bpe-LSTM_ilm{ilm_scale}_elm{ext_lm_scale}",
                    search_algorithm_params=LexiconfreeTimesyncRecogParams(
                        collapse_repeated_labels=False,
                        max_beam_sizes=[2048, 512],
                        score_thresholds=[18.0, 12.0],
                    ),
                    search_mode_params=OfflineRecogParameters(mem_rqmt=24),
                    ilm_scale=ilm_scale,
                    bpe_lstm_lm_scale=ext_lm_scale,
                )
            )
    return variants


def param_sweep_tree_lstm_recog_variants() -> List[TransducerRecogVariant]:
    variants = []
    for ilm_scale in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for ext_lm_scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            variants.append(
                TransducerRecogVariant(
                    descriptor=f"tree_bpe-LSTM_ilm{ilm_scale}_elm{ext_lm_scale}",
                    search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
                        collapse_repeated_labels=False,
                        max_beam_sizes=[2048, 512],
                        score_thresholds=[18.0, 14.0],
                    ),
                    search_mode_params=OfflineRecogParameters(mem_rqmt=24),
                    ilm_scale=ilm_scale,
                    bpe_lstm_lm_scale=ext_lm_scale,
                )
            )
    return variants


def default_offline_lexfree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_lexfree",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=11, mem_rqmt=32, dataloader_num_workers=0, batch_size_seconds=360
        ),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
    )


def default_offline_lexfree_lstm_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_lexfree_bpe-LSTM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[512, 256],
            score_thresholds=[12.0, 8.0],
        ),
        ilm_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[8],
            score_thresholds=[6.0],
        ),
    )


def mbs1024_offline_tree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="tree_mbs1024",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=11, mem_rqmt=32, dataloader_num_workers=0, batch_size_seconds=360
        ),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[18.0],
            max_beam_sizes=[1024],
        ),
    )


def mbs2048_offline_tree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="tree_mbs2048",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[18.0],
            max_beam_sizes=[2048],
        ),
    )


def default_offline_tree_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
            max_beam_sizes=[128],
            score_thresholds=[12.0],
            word_end_score_threshold=0.4,
            max_word_end_beam_size=4,
        ),
        ilm_scale=0.2,
    )


def default_offline_tree_lstm_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[40, 20],
            score_thresholds=[8.0, 8.0],
        ),
        ilm_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_lstm_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_4gram_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[16, 16],
            score_thresholds=[8.0, 8.0],
            word_end_score_threshold=0.7,
            max_word_end_beam_size=8,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.2),
        ),
        ilm_scale=0.2,
        bpe_lstm_lm_scale=0.3,
    )


def default_offline_tree_trafo_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[16.0],
            max_beam_sizes=[1024],
            word_end_score_threshold=0.6,
            max_word_end_beam_size=64,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8),
        ),
        search_mode_params=OfflineRecogParameters(),
        ilm_scale=0.2,
    )


def default_offline_tree_trafo_recog_variant_gpu() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_trafoLM_gpu",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[16.0],
            max_beam_sizes=[1024],
            word_end_score_threshold=0.6,
            max_word_end_beam_size=64,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8, use_kv_cache=False, use_gpu=True),
        ),
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=24),
        ilm_scale=0.2,
    )


def default_streaming_lexfree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256],
            score_thresholds=[14.0],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_streaming_tree_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_streaming_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
            max_word_end_beam_size=16,
            word_end_score_threshold=0.5,
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
        ilm_scale=0.2,
    )


def _get_label_scorer_configs(
    model: TrainedModel[LstmTransducerQATEncoderConfig], variant: TransducerRecogVariant
) -> List[RasrConfig]:
    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0

    label_scorer_configs = [
        get_lstm_transducer_label_scorer_config(
            model_config=model.model_config,
            checkpoint=model.get_checkpoint(variant.epoch),
            ilm_scale=variant.ilm_scale,
            blank_penalty=variant.blank_penalty,
            use_gpu=use_gpu,
        )
    ]
    if variant.bpe_lstm_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_lstm_label_scorer_config(
                bpe_size=bpe_size,
                scale=variant.bpe_lstm_lm_scale,
                use_gpu=use_gpu,
            )
        )

    return label_scorer_configs


def _convert_model_for_memristor(
    model: TrainedModel[BaseLstmTransducerQATEncoderConfig],
    variant: TransducerRecogVariant,
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
) -> TrainedModel:
    memristor_model, memristor_config = convert_model_for_memristor(
        checkpoint=model.get_checkpoint(variant.epoch),
        config=model.model_config,
        model_class=BaseLstmTransducerQATEncoderModel,
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


def _get_model_serializers(
    model_class: Type[ModuleType],
    model_config: ModelConfiguration,
) -> Collection:
    # TODO: move to common
    serializers = get_model_serializers(
        model_class,
        model_config,
    )
    serializers.serializer_objects.insert(0, ExternalImport(synaptogen_ml_root))
    serializers.serializer_objects.insert(1, NonhashedCode("import synaptogen_ml\n"))
    serializers.serializer_objects.insert(
        2, Call(callable_name="synaptogen_ml.set_fast_inference", kwargs=[("enabled", True)])
    )
    return serializers


def _run_single_variant(
    model: TrainedModel[BaseLstmTransducerQATEncoderConfig],
    variant: TransducerRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
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

    if batched_decoder:
        variant_prefix = f"{variant_prefix}batched_decoder_"
        variant = replace(variant, search_mode_params=replace(variant.search_mode_params, batched_decoder=True))

    variant = replace(variant, descriptor=f"{variant_prefix}{variant.descriptor}", num_cycles=num_cycles)

    label_scorer_configs = _get_label_scorer_configs(model=memristor_model, variant=variant)
    return run_single_bpe_variant(
        model_descriptor=memristor_model.descriptor,
        checkpoint=memristor_model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(LstmTransducerQATEncoderEncoder, memristor_model.model_config),
        label_scorer_configs=label_scorer_configs,
        bpe_size=vocab_to_bpe_size(memristor_model.model_config.target_size - 1),
        blank_index=memristor_model.model_config.target_size - 1,
        sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 else None,
        variant=variant,
        corpora=corpora,
    )


# def _run_single_variant(
#     model: TrainedModel[LstmTransducerQATEncoderConfig],
#     variant: TransducerRecogVariant,
#     corpora: List[librispeech_datasets.EvalSet],
# ) -> List[RecogResult]:
#     return run_single_bpe_variant(
#         model_descriptor=model.descriptor,
#         checkpoint=model.get_checkpoint(variant.epoch),
#         encoder_serializers=get_model_serializers(LstmTransducerQATEncoderEncoder, model.model_config),
#         label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
#         bpe_size=vocab_to_bpe_size(model.model_config.target_size - 1),
#         blank_index=model.model_config.target_size - 1,
#         sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 else None,
#         variant=variant,
#         corpora=corpora,
#     )
