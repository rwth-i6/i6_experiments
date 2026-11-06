from dataclasses import dataclass, fields, is_dataclass, replace
from inspect import signature
from typing import List, Optional, Tuple, Type
from urllib.parse import quote

from i6_core.rasr import RasrConfig
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.serialization import Collection, Call, NonhashedCode, ExternalImport

from .....data.librispeech import datasets as librispeech_datasets
from .....data.librispeech import lm as librispeech_lm
from .....data.librispeech.bpe import vocab_to_bpe_size
from .....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from .....model_pipelines.common.recog import (
    MemristorRecogResult,
    OfflineRecogParameters,
    RecogResult,
    StreamingRecogParameters,
)
from .....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from .....model_pipelines.common.serializers import get_model_serializers
from .....model_pipelines.common.train import TrainedModel
from .....model_pipelines.qat_ffnn_transducer.adaptation.checkpoints import ExtractAdapterCheckpointJob
from .....model_pipelines.qat_ffnn_transducer.adaptation.config import MemristorAdapterConfig
from .....model_pipelines.qat_ffnn_transducer.adaptation.pytorch_modules import MemristorAdapterScorer
from .....model_pipelines.qat_ffnn_transducer.mem_inited.label_scorer_config import (
    get_ffnn_transducer_label_scorer_config,
)
from .....model_pipelines.qat_ffnn_transducer.pytorch_modules import (
    QATFFNNTransducerConfig as BaseQATFFNNTransducerConfig,
    QATFFNNTransducerModel as BaseQATFFNNTransducerModel,
    QATFFNNTransducerModelNoOutQuant,
    QATFFNNTransducerModelNoJointOffload,
    QATFFNNTransducerModelNoJointNet7Offload,
    QATFFNNTransducerModelNoPredictionOffload,
    QATFFNNTransducerModelNoPredictionJointOffload,
)

from .....model_pipelines.qat_ffnn_transducer.mem_inited.pytorch_modules import (
    QATFFNNTransducerConfig,
    QATFFNNTransducerEncoder,
    QATFFNNTransducerModel,
    QATFFNNTransducerScorer,
    QATFFNNTransducerScorerNoOutQuant,
    QATFFNNTransducerScorerNoJointOffload,
    QATFFNNTransducerScorerNoJointNet7Offload,
    QATFFNNTransducerScorerNoPredictionOffload,
    QATFFNNTransducerScorerNoPredictionJointOffload,
)

from i6_models.config import ModelConfiguration, ModuleType

from .....model_pipelines.common.memrecog import convert_model_for_memristor
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

from ..common import BaseRecogVariant, post_recog_memristor_results, run_single_bpe_variant

from .....tools import synaptogen_ml_root


@dataclass
class TransducerRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    ilm_scale: float = 0.0
    blank_penalty: float = 0.0
    presoftmax_blank_penalty: bool = False
    cycles: Optional[int] = None


def _recognition_settings_suffix(settings: dict) -> str:
    labels = {
        "converter_hardware_settings": "adc",
        "pos_enc_converter_hardware_settings": "pos-adc",
        "joint_l1_converter_hardware_settings": "j1-adc",
        "joint_l2_converter_hardware_settings": "j2-adc",
        "correction_settings": "corr",
        "joint_correction_settings": "joint-corr",
        "joint_correction_layer": "joint-layer",
        "batched_decoder": "batched",
        "batch_size_seconds": "batch",
        "use_cache": "cache",
        "offload_prediction_network": "pred-offload",
        "offload_joint_network": "joint-offload",
        "offload_joint_net_7": "joint7-offload",
        "output_quantizer": "out-quant",
        "regrid_embedding": "embedding-regrid",
        "regrid_joint_net_7": "j2-regrid",
        "output_precision_bits": "p",
        "output_range_bits": "r",
        "hardware_input_vmax": "v",
        "hardware_output_current_scaling": "s",
        "num_cycles": "c",
        "test_input_value": "i",
        "relative_deviation": "dev",
        "ideal_programming": "ideal",
    }
    excluded = {"model", "variants", "corpora", "max_runs", "suffix", "input_bits", "adapter_model", "adapter_epoch"}

    def format_value(value):
        if is_dataclass(value):
            implicit = {"hardware_input_vmax": 0.6, "hardware_output_current_scaling": 8020.0}
            parts = []
            for field in fields(value):
                if field.name in excluded:
                    continue
                item = getattr(value, field.name)
                if item is None or (field.name in implicit and item == implicit[field.name]):
                    continue
                label = labels.get(field.name, field.name.replace("_", "-"))
                if isinstance(item, bool):
                    if item:
                        parts.append(label)
                else:
                    parts.append(f"{label}{format_value(item)}")
            return "-".join(parts)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return quote(str(value), safe="-_.")

    parts = []
    for name, parameter in signature(run).parameters.items():
        if name in excluded:
            continue
        value = settings[name]
        if value is None or (not is_dataclass(value) and value == parameter.default):
            continue
        if name.startswith("joint_l") and value == settings["converter_hardware_settings"]:
            continue
        label = labels.get(name, name.replace("_", "-"))
        if isinstance(value, bool):
            parts.append(label if value else f"no-{label}")
        else:
            parts.append(f"{label}-{format_value(value)}")
    if settings["separated_joint_adc"]:
        parts.append("split-adc")
    return "".join(f"__{part}" for part in parts)


def run(
    model: TrainedModel[QATFFNNTransducerConfig],
    variants: Optional[List[TransducerRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l1_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l2_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    correction_settings: Optional[CycleCorrectionSettings] = None,
    max_runs: Optional[int] = 5,
    batched_decoder: bool = False,
    batch_size_seconds: Optional[int] = None,
    use_cache: bool = False,
    offload_prediction_network: bool = True,
    offload_joint_network: bool = True,
    offload_joint_net_7: bool = True,
    output_quantizer: bool = True,
    suffix: str = "",
    regrid_embedding: bool = True,
    joint_correction_settings: Optional[CycleCorrectionSettings] = None,
    joint_correction_layer: int = 3,
    regrid_joint_net_7: bool = False,
    adapter_model: Optional[TrainedModel[MemristorAdapterConfig]] = None,
    adapter_epoch: Optional[int] = None,
) -> Tuple[List[MemristorRecogResult], List[RecogResult]]:

    adapter_checkpoint = None
    if adapter_model is not None:
        if not (offload_prediction_network and offload_joint_network and offload_joint_net_7 and output_quantizer):
            raise NotImplementedError("The adapter scorer currently supports fully offloaded models with output quantization")
        if adapter_epoch is None:
            adapter_epoch = max(adapter_model.checkpoints)
        adapter_checkpoint = ExtractAdapterCheckpointJob(adapter_model.get_checkpoint(adapter_epoch)).out_checkpoint
        suffix += f"_adapter_{adapter_model.descriptor}_epoch_{adapter_epoch}"
        adapter_config = adapter_model.model_config.base_config
        if joint_l1_converter_hardware_settings is None:
            joint_l1_converter_hardware_settings = getattr(adapter_config, "joint_l1_converter_hardware_settings", None)
        if joint_l2_converter_hardware_settings is None:
            joint_l2_converter_hardware_settings = getattr(adapter_config, "joint_l2_converter_hardware_settings", None)

    if joint_correction_layer not in (1, 2, 3):
        raise ValueError("joint_correction_layer must be 1, 2, or 3")
    if (joint_correction_settings is not None or joint_correction_layer != 3) and not (
        offload_prediction_network and offload_joint_network and offload_joint_net_7
    ):
        raise ValueError("Joint programming overrides without joint offload")
    separated_joint_adc = (
        joint_l1_converter_hardware_settings is not None or joint_l2_converter_hardware_settings is not None
    )
    if variants is None:
        variants = default_recog_variants()
    if not offload_prediction_network or not offload_joint_network or not offload_joint_net_7:
        # variants = [variant for variant in variants if variant.descriptor in ("lexfree", "tree_mbs1024")]
        if not offload_prediction_network and not offload_joint_network:
            variants = [
                (
                    replace(variant, search_mode_params=replace(variant.search_mode_params, gpu_mem_rqmt=24))
                    if variant.descriptor == "lexfree"
                    else variant
                )
                for variant in variants
            ]

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
    if joint_l1_converter_hardware_settings is None:
        joint_l1_converter_hardware_settings = converter_hardware_settings

    if joint_l2_converter_hardware_settings is None:
        joint_l2_converter_hardware_settings = converter_hardware_settings

    memristor_results = []
    results = []
    run_settings = dict(locals())
    for variant in variants:
        if (
            batch_size_seconds is not None
            and variant.search_mode_params.batch_size_seconds == OfflineRecogParameters.batch_size_seconds
        ):
            variant = replace(
                variant,
                search_mode_params=replace(variant.search_mode_params, batch_size_seconds=batch_size_seconds),
            )
        alias_settings = dict(run_settings, batched_decoder=batched_decoder or variant.search_mode_params.batched_decoder)
        if isinstance(variant.search_mode_params, OfflineRecogParameters):
            effective_batch_size = variant.search_mode_params.batch_size_seconds
            alias_settings["batch_size_seconds"] = (
                effective_batch_size if effective_batch_size != OfflineRecogParameters.batch_size_seconds else None
            )
        descriptor_suffix = suffix + _recognition_settings_suffix(alias_settings)
        cycles_variant_results = []
        for num_cycles in range(1, max_runs + 1):

            if variant.cycles is not None and num_cycles > variant.cycles:
                break

            _, variant_result = _run_single_variant(
                model=model,
                variant=variant,
                corpora=corpora,
                converter_hardware_settings=converter_hardware_settings,
                pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
                joint_l1_converter_hardware_settings=joint_l1_converter_hardware_settings,
                joint_l2_converter_hardware_settings=joint_l2_converter_hardware_settings,
                correction_settings=correction_settings,
                joint_correction_settings=joint_correction_settings,
                joint_correction_layer=joint_correction_layer,
                num_cycles=num_cycles,
                batched_decoder=batched_decoder,
                use_cache=use_cache,
                offload_prediction_network=offload_prediction_network,
                offload_joint_network=offload_joint_network,
                offload_joint_net_7=offload_joint_net_7,
                output_quantizer=output_quantizer,
                separated_joint_adc=separated_joint_adc,
                suffix=descriptor_suffix,
                regrid_embedding=regrid_embedding,
                regrid_joint_net_7=regrid_joint_net_7,
                adapter_checkpoint=adapter_checkpoint,
            )
            # variant result is a list
            cycles_variant_results.extend(variant_result)

        results.extend(cycles_variant_results)
        memristor_results.extend(
            post_recog_memristor_results(
                descriptor=f"{model.descriptor}{descriptor_suffix}_memristor_{variant.descriptor}",
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



def ilm04_elm05_tree_4gram_recog_variant(num_shards: int =1, cycles: int = 3, resumable: bool = False) -> List[TransducerRecogVariant]:

    return TransducerRecogVariant(
        descriptor="tree_4gram_ilm0.4_elm0.5",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=24, mem_rqmt=32, use_legacy_dataloader_num_workers_0_hash=True, num_shards=num_shards, resumable=resumable
        ),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.5),
            max_beam_sizes=[2048],
            score_thresholds=[18.0],
            word_end_score_threshold=None,
            max_word_end_beam_size=None,
        ),
        ilm_scale=0.4,
        cycles=cycles
    )



def blank_penalty_ilm04_elm05_tree_4gram_recog_variant(num_shards: int =1, cycles: int = 3, resumable: bool = False, blank_penalty: float = 0) -> List[TransducerRecogVariant]:

    return TransducerRecogVariant(
        descriptor=f"tree_4gram_ilm0.4_elm0.5_bp{blank_penalty}",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=24, mem_rqmt=32, use_legacy_dataloader_num_workers_0_hash=True, num_shards=num_shards, resumable=resumable
        ),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.5),
            max_beam_sizes=[2048],
            score_thresholds=[18.0],
            word_end_score_threshold=None,
            max_word_end_beam_size=None,
        ),
        ilm_scale=0.4,
        cycles=cycles,
        blank_penalty=blank_penalty
    )


def param_sweep_tree_4gram_recog_variants(
    num_shards: int = 1, cycles: Optional[int] = None, resumable=False
) -> List[TransducerRecogVariant]:
    variants = []
    params = sorted(
        [
            (0.3, 0.5),
            (0.3, 0.6),
            (0.4, 0.5),
            (0.4, 0.6),
        ]
    )
    for ilm_scale, ext_lm_scale in params:
        variants.append(
            TransducerRecogVariant(
                descriptor=f"tree_4gram_ilm{ilm_scale}_elm{ext_lm_scale}",
                search_mode_params=OfflineRecogParameters(
                    gpu_mem_rqmt=11,
                    mem_rqmt=32,
                    use_legacy_dataloader_num_workers_0_hash=True,
                    num_shards=num_shards,
                    resumable=resumable,
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
                cycles=cycles,
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
                    search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=24),
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
                    search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=11),
                    ilm_scale=ilm_scale,
                    bpe_lstm_lm_scale=ext_lm_scale,
                )
            )
    return variants


def default_offline_lexfree_recog_variant(
    num_shards: int = 1, resumable: bool = False, cycles: Optional[int] = None
) -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="lexfree",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=32,
            mem_rqmt=32,
            use_legacy_dataloader_num_workers_0_hash=True,
            num_shards=num_shards,
            resumable=resumable,
        ),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
        cycles=cycles,
    )



def mbs1024_offline_lexfree_recog_variant(
    num_shards: int = 1, resumable: bool = False, cycles: Optional[int] = None
) -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="lexfree_mbs1024",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=24,
            mem_rqmt=32,
            use_legacy_dataloader_num_workers_0_hash=True,
            num_shards=num_shards,
            resumable=resumable,
        ),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1024],
            score_thresholds=[18.0],
        ),
        cycles=cycles,
    )


def joint_debug_offline_lexfree_recog_variant(
    num_shards: int = 1, hash_salt: Optional[str] = None
) -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="lexfree_joint_debug" + (f"_{hash_salt}" if hash_salt is not None else ""),
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=11, mem_rqmt=32, debug_joint=True, num_shards=num_shards, hash_salt=hash_salt
        ),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
        cycles=1,
    )


def blank_penalty_offline_lexfree_recog_variant(
    blank_penalty: int = 0, cycles: Optional[int] = None, num_shards: int = 1, resumable: bool = False
) -> TransducerRecogVariant:
    descriptor = "lexfree"
    if blank_penalty != 0:
        descriptor = f"lexfree_bp{blank_penalty}"
    return TransducerRecogVariant(
        descriptor=descriptor,
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=48,
            mem_rqmt=32,
            use_legacy_dataloader_num_workers_0_hash=True,
            num_shards=num_shards,
            resumable=resumable,
        ),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
        blank_penalty=blank_penalty,
        cycles=cycles,
    )

def pre_softmax_blank_penalty_offline_lexfree_recog_variant(
    blank_penalty: int = 0, cycles: Optional[int] = None, num_shards: int = 1, resumable: bool = False
) -> TransducerRecogVariant:
    descriptor = f"lexfree_presoftmax_bp{blank_penalty}"
    return TransducerRecogVariant(
        descriptor=descriptor,
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=11,
            mem_rqmt=32,
            use_legacy_dataloader_num_workers_0_hash=True,
            num_shards=num_shards,
            resumable=resumable,
        ),
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
        blank_penalty=blank_penalty,
        cycles=cycles,
        presoftmax_blank_penalty=True
    )


def default_offline_lexfree_lstm_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="lexfree_bpe-LSTM",
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
        descriptor="tree",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[8],
            score_thresholds=[6.0],
        ),
    )


def mbs1024_offline_tree_recog_variant(
    num_shards: int = 1, cycles: Optional[int] = None, resumable=False,
) -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="tree_mbs1024",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=24, mem_rqmt=32, dataloader_num_workers=0, num_shards=num_shards, resumable=resumable
        ),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[18.0],
            max_beam_sizes=[1024],
        ),
        cycles=cycles,
    )

def nbs_mbs1024_offline_tree_recog_variant(
    num_shards: int = 1, cycles: Optional[int] = None, resumable=False, nbest_size: int = 0, blank_penalty: float = 0.0
) -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor=f"tree_mbs1024_nbs{nbest_size}_bp{blank_penalty}",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=24, mem_rqmt=32, dataloader_num_workers=0, num_shards=num_shards, resumable=resumable, nbest_size=nbest_size
        ),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[18.0],
            max_beam_sizes=[1024],
        ),
        cycles=cycles,
        blank_penalty=blank_penalty
    )


def blank_penalty_mbs1024_offline_tree_recog_variant(
    blank_penalty: float = 2.5, num_shards: int = 1, cycles: Optional[int] = None, resumable=False
) -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor=f"tree_mbs1024_bp{blank_penalty}",
        search_mode_params=OfflineRecogParameters(
            gpu_mem_rqmt=11, mem_rqmt=32, dataloader_num_workers=0, num_shards=num_shards, resumable=resumable
        ),
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            score_thresholds=[18.0],
            max_beam_sizes=[1024],
        ),
        cycles=cycles,
        blank_penalty=blank_penalty,
    )


def pre_softmax_blank_penalty_mbs1024_offline_tree_recog_variant(
    blank_penalty: float = 2.5, num_shards: int = 1, cycles: Optional[int] = None, resumable=False
) -> TransducerRecogVariant:
    return replace(
        blank_penalty_mbs1024_offline_tree_recog_variant(
            blank_penalty=blank_penalty, num_shards=num_shards, cycles=cycles, resumable=resumable
        ),
        descriptor=f"tree_mbs1024_presoftmax_bp{blank_penalty}",
        presoftmax_blank_penalty=True,
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
        descriptor="tree_4gram",
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
        descriptor="tree_bpe-LSTM",
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
        descriptor="tree_4gram_bpe-LSTM",
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
        descriptor="tree_trafoLM",
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
        descriptor="tree_trafoLM_gpu",
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
        descriptor="streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256],
            score_thresholds=[14.0],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_streaming_tree_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="streaming_tree_4gram",
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
    model: TrainedModel[QATFFNNTransducerConfig],
    variant: TransducerRecogVariant,
    num_cycles: int,
    use_cache: bool = False,
    converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l1_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    joint_l2_converter_hardware_settings: Optional[DacAdcHardwareSettings] = None,
    separated_joint_adc: bool = False,
    scorer_model_class=QATFFNNTransducerScorer,
    adapter_checkpoint=None,
) -> List[RasrConfig]:
    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0

    label_scorer_configs = [
        (
            get_ffnn_transducer_label_scorer_config(
                model_config=model.model_config,
                checkpoint=model.get_checkpoint(variant.epoch),
                ilm_scale=variant.ilm_scale,
                blank_penalty=variant.blank_penalty,
                debug_joint=variant.search_mode_params.debug_joint,
                max_batch_size=2048,
                use_gpu=use_gpu,
                use_cache=use_cache,
                scorer_model_class=scorer_model_class,
                adapter_checkpoint=adapter_checkpoint,
                num_cycles=num_cycles,
                converter_hardware_settings=converter_hardware_settings,
                pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
                joint_l1_converter_hardware_settings=joint_l1_converter_hardware_settings,
                joint_l2_converter_hardware_settings=joint_l2_converter_hardware_settings,
                separated_joint_adc=True,
                presoftmax_blank_penalty=variant.presoftmax_blank_penalty,
            )
            if separated_joint_adc
            else get_ffnn_transducer_label_scorer_config(
                model_config=model.model_config,
                checkpoint=model.get_checkpoint(variant.epoch),
                ilm_scale=variant.ilm_scale,
                blank_penalty=variant.blank_penalty,
                debug_joint=variant.search_mode_params.debug_joint,
                max_batch_size=2048,
                use_gpu=use_gpu,
                use_cache=use_cache,
                scorer_model_class=scorer_model_class,
                adapter_checkpoint=adapter_checkpoint,
                num_cycles=num_cycles,
                converter_hardware_settings=converter_hardware_settings,
                pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
                presoftmax_blank_penalty=variant.presoftmax_blank_penalty
            )
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
    model: TrainedModel[QATFFNNTransducerConfig],
    variant: TransducerRecogVariant,
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
    model_class: Type[ModuleType] = BaseQATFFNNTransducerModel,
    descriptor_suffix: str = "",
    regrid_embedding: bool = True,
    joint_correction_settings: Optional[CycleCorrectionSettings] = None,
    joint_correction_layer: int = 3,
    regrid_joint_net_7: bool = False,
) -> TrainedModel:
    memristor_model, memristor_config = convert_model_for_memristor(
        checkpoint=model.get_checkpoint(variant.epoch),
        config=model.model_config,
        model_class=model_class,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        joint_correction_settings=joint_correction_settings,
        joint_correction_layer=joint_correction_layer,
        num_cycles=num_cycles,
        alias=f"{model.descriptor}{descriptor_suffix}_memristor_num_cycles_{num_cycles}",
        regrid_embedding=regrid_embedding,
        regrid_joint_net_7=regrid_joint_net_7,
    )
    epoch = variant.epoch if variant.epoch is not None else 0

    return TrainedModel(
        descriptor=f"memristor_{model.descriptor}{descriptor_suffix}",
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
    model: TrainedModel[BaseQATFFNNTransducerConfig],
    variant: TransducerRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    joint_l1_converter_hardware_settings: DacAdcHardwareSettings,
    joint_l2_converter_hardware_settings: DacAdcHardwareSettings,
    correction_settings: Optional[CycleCorrectionSettings],
    num_cycles: int,
    batched_decoder: bool = False,
    use_cache: bool = False,
    offload_prediction_network: bool = True,
    offload_joint_network: bool = True,
    offload_joint_net_7: bool = True,
    output_quantizer: bool = True,
    separated_joint_adc: bool = False,
    suffix: str = "",
    regrid_embedding: bool = True,
    joint_correction_settings: Optional[CycleCorrectionSettings] = None,
    joint_correction_layer: int = 3,
    regrid_joint_net_7: bool = False,
    adapter_checkpoint: Optional[PtCheckpoint] = None,
) -> Tuple[int, List[RecogResult]]:

    descriptor_suffix = suffix
    model_class = BaseQATFFNNTransducerModel
    scorer_model_class = QATFFNNTransducerScorer
    if not output_quantizer:
        model_class = QATFFNNTransducerModelNoOutQuant
        scorer_model_class = QATFFNNTransducerScorerNoOutQuant
    if not offload_prediction_network and not offload_joint_network:
        if not offload_joint_net_7:
            raise ValueError("conflicting offload settings: check joint 7")
        model_class = QATFFNNTransducerModelNoPredictionJointOffload
        scorer_model_class = QATFFNNTransducerScorerNoPredictionJointOffload
    elif not offload_prediction_network:
        if not offload_joint_net_7:
            raise ValueError("offload config not supported: check joint 7")
        model_class = QATFFNNTransducerModelNoPredictionOffload
        scorer_model_class = QATFFNNTransducerScorerNoPredictionOffload
    elif not offload_joint_network:
        if not offload_joint_net_7:
            raise ValueError("conflicting offload settings: check joint 7")
        model_class = QATFFNNTransducerModelNoJointOffload
        scorer_model_class = QATFFNNTransducerScorerNoJointOffload
    elif not offload_joint_net_7:
        model_class = QATFFNNTransducerModelNoJointNet7Offload
        scorer_model_class = QATFFNNTransducerScorerNoJointNet7Offload

    if adapter_checkpoint is not None:
        scorer_model_class = MemristorAdapterScorer

    memristor_model = _convert_model_for_memristor(
        model=model,
        variant=variant,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        correction_settings=correction_settings,
        joint_correction_settings=joint_correction_settings,
        joint_correction_layer=joint_correction_layer,
        num_cycles=num_cycles,
        model_class=model_class,
        descriptor_suffix=descriptor_suffix,
        regrid_embedding=regrid_embedding,
        regrid_joint_net_7=regrid_joint_net_7,
    )
    variant_prefix = f"num_cycle_{num_cycles}_"

    if batched_decoder:
        variant = replace(variant, search_mode_params=replace(variant.search_mode_params, batched_decoder=True))

    variant = replace(variant, descriptor=f"{variant_prefix}{variant.descriptor}", num_cycles=num_cycles)

    label_scorer_configs = _get_label_scorer_configs(
        model=memristor_model,
        variant=variant,
        num_cycles=num_cycles,
        use_cache=use_cache,
        scorer_model_class=scorer_model_class,
        adapter_checkpoint=adapter_checkpoint,
        converter_hardware_settings=converter_hardware_settings,
        pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        joint_l1_converter_hardware_settings=joint_l1_converter_hardware_settings,
        joint_l2_converter_hardware_settings=joint_l2_converter_hardware_settings,
        separated_joint_adc=separated_joint_adc,
    )

    return run_single_bpe_variant(
        model_descriptor=memristor_model.descriptor,
        checkpoint=memristor_model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(QATFFNNTransducerEncoder, memristor_model.model_config),
        label_scorer_configs=label_scorer_configs,
        bpe_size=vocab_to_bpe_size(memristor_model.model_config.target_size - 1),
        blank_index=memristor_model.model_config.target_size - 1,
        sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 else None,
        variant=variant,
        corpora=corpora,
    )
