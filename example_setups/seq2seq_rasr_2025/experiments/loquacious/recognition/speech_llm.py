from dataclasses import dataclass
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import tk

from ....data.loquacious import datasets as loquacious_datasets
from ....data.loquacious.recog import LoquaciousTreeTimesyncRecogParams
from ....model_pipelines.common.recog import OfflineRecogParameters, RecogResult
from ....model_pipelines.common.recog_rasr_config import LexiconfreeLabelsyncRecogParams, LexiconfreeTimesyncRecogParams
from ....model_pipelines.speech_llm.label_scorer_config import (
    get_ctc_label_scorer_config,
    get_ctc_prefix_label_scorer_config,
    get_speech_lm_label_scorer_config,
)
from ....model_pipelines.speech_llm.pytorch_modules import SpeechLmEncoder
from .common import BaseRecogVariant, run_single_hf_token_variant


@dataclass
class SLMRecogVariant(BaseRecogVariant):
    ctc_score_scale: float = 0.0


def run(
    model_descriptor: str,
    model_kwargs: dict,
    checkpoint: PtCheckpoint,
    huggingface_repo_dir: tk.Path,
    variants: Optional[List[SLMRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = ["dev.all"]

    results = []

    for variant in variants:
        results.extend(
            _run_single_variant(
                model_descriptor=model_descriptor,
                model_kwargs=model_kwargs,
                checkpoint=checkpoint,
                huggingface_repo_dir=huggingface_repo_dir,
                variant=variant,
                corpora=corpora,
            )
        )
    return results


def default_recog_variants() -> List[SLMRecogVariant]:
    return [
        default_lexfree_recog_variant(),
        default_lexfree_slm_ctc_recog_variant(),
        # default_lexfree_slm_ctc_timesync_recog_variant(),
    ]


def default_lexfree_recog_variant() -> SLMRecogVariant:
    return SLMRecogVariant(
        descriptor="recog_lexfree_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            score_thresholds=[12.0],
            max_beam_sizes=[4],
            length_norm_scale=1.0,
            recombination_mode="off",
        ),
        search_mode_params=OfflineRecogParameters(mem_rqmt=24),
    )


def default_lexfree_slm_ctc_recog_variant() -> SLMRecogVariant:
    return SLMRecogVariant(
        descriptor="recog_lexfree_aed+ctc_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            score_thresholds=[12.0, 12.0],
            max_beam_sizes=[8, 4],
            length_norm_scale=0.0,
            recombination_mode="off",
        ),
        search_mode_params=OfflineRecogParameters(mem_rqmt=24),
        ctc_score_scale=0.3,
    )


def default_lexfree_slm_ctc_timesync_recog_variant() -> SLMRecogVariant:
    return SLMRecogVariant(
        descriptor="recog_lexfree_aed+ctc_timesync",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            score_thresholds=[8.0, 8.0],
            max_beam_sizes=[32, 16],
            collapse_repeated_labels=True,
            recombination_mode="on",
        ),
        search_mode_params=OfflineRecogParameters(mem_rqmt=24),
        ctc_score_scale=0.3,
    )


def _get_label_scorer_configs(
    model_kwargs: dict, checkpoint: PtCheckpoint, variant: SLMRecogVariant
) -> List[RasrConfig]:
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0
    labelsync = isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)

    aed_label_scorer_config = get_speech_lm_label_scorer_config(
        model_kwargs=model_kwargs,
        checkpoint=checkpoint,
        use_gpu=use_gpu,
        scale=1.0 - variant.ctc_score_scale,
    )

    ctc_label_scorer_config = None
    if variant.ctc_score_scale != 0.0:
        if labelsync:
            ctc_label_scorer_config = get_ctc_prefix_label_scorer_config(
                model_kwargs=model_kwargs,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                use_gpu=use_gpu,
            )
        else:
            ctc_label_scorer_config = get_ctc_label_scorer_config(
                model_kwargs=model_kwargs,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                use_gpu=use_gpu,
            )

    if labelsync:
        return list(filter(None, [aed_label_scorer_config, ctc_label_scorer_config]))
    else:
        assert ctc_label_scorer_config is not None
        return list(filter(None, [ctc_label_scorer_config, aed_label_scorer_config]))


def _run_single_variant(
    model_descriptor: str,
    checkpoint: PtCheckpoint,
    model_kwargs: dict,
    huggingface_repo_dir: tk.Path,
    variant: SLMRecogVariant,
    corpora: List[loquacious_datasets.EvalSet],
) -> List[RecogResult]:
    if isinstance(variant.search_algorithm_params, (LexiconfreeTimesyncRecogParams, LoquaciousTreeTimesyncRecogParams)):
        blank_index = 151936  # TODO: set dynamically
    else:
        blank_index = None

    encoder_serializers = Collection(
        [
            Import(f"{SpeechLmEncoder.__module__}.{SpeechLmEncoder.__name__}"),
            PyTorchModel(
                model_class_name=SpeechLmEncoder.__name__,
                model_kwargs=model_kwargs,
            ),
        ]
    )

    return run_single_hf_token_variant(
        model_descriptor=model_descriptor,
        huggingface_repo_dir=huggingface_repo_dir,
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=_get_label_scorer_configs(
            model_kwargs=model_kwargs, checkpoint=checkpoint, variant=variant
        ),
        blank_index=blank_index,
        sentence_end_index=151643,
        variant=variant,
        corpora=corpora,
        recog_data_config_fn=(lambda _corpus: loquacious_datasets.get_hf_recog_data(_corpus)),
    )
