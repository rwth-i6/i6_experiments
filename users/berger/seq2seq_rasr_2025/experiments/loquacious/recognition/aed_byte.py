from dataclasses import dataclass
from typing import List, Optional

from i6_core.rasr import RasrConfig

from ....data.loquacious import datasets as loquacious_datasets
from ....model_pipelines.aed.label_scorer_config import (
    get_aed_label_scorer_config,
    get_ctc_label_scorer_config,
    get_ctc_prefix_label_scorer_config,
)
from ....model_pipelines.aed.pytorch_modules import AEDConfig, AEDEncoder
from ....model_pipelines.common.recog import RecogResult
from ....model_pipelines.common.recog_rasr_config import LexiconfreeLabelsyncRecogParams, LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.common.train import TrainedModel
from .common import BaseRecogVariant, run_single_byte_variant


@dataclass
class AEDRecogVariant(BaseRecogVariant):
    ctc_score_scale: float = 0.0


def run(
    model: TrainedModel[AEDConfig],
    variants: Optional[List[AEDRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = loquacious_datasets.EVAL_SETS

    results = []
    for variant in variants:
        results.extend(_run_single_variant(model=model, variant=variant, corpora=corpora))
    return results


def default_recog_variants() -> List[AEDRecogVariant]:
    return [
        default_lexfree_recog_variant(),
        default_lexfree_aed_ctc_recog_variant(),
        default_lexfree_aed_ctc_timesync_recog_variant(),
    ]


def default_lexfree_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            max_beam_sizes=[8],
            length_norm_scale=1.2,
        ),
    )


def default_lexfree_aed_ctc_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree_aed+ctc_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            max_beam_sizes=[16, 8],
            length_norm_scale=1.2,
        ),
        ctc_score_scale=0.3,
    )


def default_lexfree_aed_ctc_timesync_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree_aed+ctc_timesync",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            max_beam_sizes=[16, 8],
            collapse_repeated_labels=True,
        ),
        ctc_score_scale=0.3,
    )


def _get_label_scorer_configs(model: TrainedModel[AEDConfig], variant: AEDRecogVariant) -> List[RasrConfig]:
    checkpoint = model.get_checkpoint(variant.epoch)
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0
    labelsync = isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)

    aed_label_scorer_config = get_aed_label_scorer_config(
        model_config=model.model_config,
        checkpoint=checkpoint,
        use_gpu=use_gpu,
        scale=1.0 - variant.ctc_score_scale,
    )

    ctc_label_scorer_config = None
    if variant.ctc_score_scale != 0.0:
        if labelsync:
            ctc_label_scorer_config = get_ctc_prefix_label_scorer_config(
                model_config=model.model_config,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                use_gpu=use_gpu,
            )
        else:
            ctc_label_scorer_config = get_ctc_label_scorer_config(
                model_config=model.model_config,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                use_gpu=use_gpu,
            )

    if labelsync:
        return list(filter(None, [aed_label_scorer_config, ctc_label_scorer_config]))
    assert ctc_label_scorer_config is not None
    return list(filter(None, [ctc_label_scorer_config, aed_label_scorer_config]))


def _run_single_variant(
    model: TrainedModel[AEDConfig], variant: AEDRecogVariant, corpora: List[loquacious_datasets.EvalSet]
) -> List[RecogResult]:
    if isinstance(variant.search_algorithm_params, LexiconfreeTimesyncRecogParams):
        blank_index = model.model_config.label_target_size
    else:
        blank_index = None

    return run_single_byte_variant(
        model_descriptor=model.descriptor,
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=get_model_serializers(AEDEncoder, model.model_config),
        label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
        blank_index=blank_index,
        sentence_end_index=0,
        variant=variant,
        corpora=corpora,
    )
