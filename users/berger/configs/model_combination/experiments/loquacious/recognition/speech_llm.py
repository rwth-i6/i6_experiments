from dataclasses import dataclass, replace
from typing import Callable, List, Literal, Optional, Sequence

from i6_core.rasr import RasrConfig
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import tk

from ....data.base import DataConfig, OggZipDataConfig
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
    # Scale of the speech LM scorer. When None, the legacy coupling `1 - ctc_score_scale` is used,
    # which cannot express a search where CTC drives at full scale and the speech LM is weighted
    # independently on top of it.
    speech_lm_score_scale: Optional[float] = None
    # Constant score added on every transition that emits a new label. Negative values reward label
    # emissions, positive ones penalize them. Only used by the time-synchronous variants.
    label_penalty: Optional[float] = None


def _get_oggzip_recog_data(corpus: loquacious_datasets.EvalSet) -> OggZipDataConfig:
    return loquacious_datasets.get_fixed_recog_data(corpus)


def run(
    model_descriptor: str,
    model_kwargs: dict,
    checkpoint: PtCheckpoint,
    huggingface_repo_dir: tk.Path,
    variants: Optional[List[SLMRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
    recog_data_source: Literal["huggingface", "oggzip"] = "huggingface",
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = ["dev.all"]

    recog_data_config_fn: Callable[[loquacious_datasets.EvalSet], DataConfig]
    if recog_data_source == "huggingface":
        recog_data_config_fn = loquacious_datasets.get_hf_recog_data
    elif recog_data_source == "oggzip":
        recog_data_config_fn = _get_oggzip_recog_data
        variants = [replace(variant, descriptor=f"{variant.descriptor}_oggzip") for variant in variants]
    else:
        raise ValueError(f"Invalid recog_data_source {recog_data_source!r}")

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
                recog_data_config_fn=recog_data_config_fn,
            )
        )
    return results


def default_recog_variants() -> List[SLMRecogVariant]:
    # The time-synchronous variants are not part of this list: they are run from the experiment entry
    # point on the oggzip-v2 data, whereas this list is also used with the HuggingFace recog data.
    return [
        default_lexfree_recog_variant(),
        default_lexfree_slm_ctc_recog_variant(),
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


def default_lexfree_ctc_slm_timesync_recog_variant(
    speech_lm_score_scale: float = 0.6,
    label_penalty: Optional[float] = None,
    ctc_beam_size: int = 64,
    speech_lm_beam_size: int = 8,
    score_threshold: float = 12.0,
) -> SLMRecogVariant:
    """
    Time-synchronous CTC + speech LM search.

    The CTC head of the speech LM is the primary scorer and drives the search: it generates the
    candidates for every encoder frame over the full vocabulary including blank. The autoregressive
    speech LM only rescores the surviving candidates, and its `LM` transition preset restricts it to
    transitions that actually emit a label, so it neither scores nor advances its state on blank and
    loop transitions. This is the counterpart to the label-synchronous variants, where the speech LM
    drives the search and CTC enters as a prefix scorer.

    Consequently the two scales are independent here rather than summing to one: CTC keeps its full
    scale and `speech_lm_score_scale` weights the speech LM like a language model.

    Pruning is staged with one entry per label scorer, in scorer order. Entry `i` is the beam after
    scorer `i` has contributed, and the last entry is applied to the final beam after recombination.
    The CTC stage is therefore kept wide: it is cheap, and only what survives it is handed to the
    expensive speech LM, whose stage size determines how many autoregressive steps are run per frame.
    """
    num_scorers = 2 if label_penalty is None else 3
    return SLMRecogVariant(
        descriptor="recog_lexfree_ctc+slm_timesync",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            # Intermediate stages keep the wide CTC beam, the final entry is the actual search beam.
            max_beam_sizes=[ctc_beam_size] * (num_scorers - 1) + [speech_lm_beam_size],
            score_thresholds=[score_threshold] * num_scorers,
            collapse_repeated_labels=True,
            recombination_mode="on",
        ),
        search_mode_params=OfflineRecogParameters(mem_rqmt=24),
        ctc_score_scale=1.0,
        speech_lm_score_scale=speech_lm_score_scale,
        label_penalty=label_penalty,
    )


def timesync_tuning_variants(
    speech_lm_score_scales: Sequence[float] = (0.3, 0.6, 0.9),
    label_penalties: Sequence[Optional[float]] = (None, -1.0, -2.0),
    speech_lm_beam_size: int = 8,
    score_threshold: float = 12.0,
) -> List[SLMRecogVariant]:
    """
    Light tuning grid for the time-synchronous CTC + speech LM search.

    The two axes that decide the operating point are the speech LM scale and the label penalty: every
    emitted label costs an additional speech LM score, which biases the search towards blank, and the
    penalty is the counterweight to that. Pruning is left fixed here and probed separately.
    """
    variants = []
    for speech_lm_score_scale in speech_lm_score_scales:
        for label_penalty in label_penalties:
            variant = default_lexfree_ctc_slm_timesync_recog_variant(
                speech_lm_score_scale=speech_lm_score_scale,
                label_penalty=label_penalty,
                speech_lm_beam_size=speech_lm_beam_size,
                score_threshold=score_threshold,
            )
            variant.descriptor += f"_slm-{speech_lm_score_scale}"
            if label_penalty is not None:
                variant.descriptor += f"_labelpen-{label_penalty}"
            variant.descriptor += f"_beam-{speech_lm_beam_size}_score-{score_threshold}"
            variants.append(variant)
    return variants


def _get_label_penalty_scorer_config(label_penalty: float) -> RasrConfig:
    """
    Constant score on every transition that emits a new label.

    `word-exit` is not available here: the lexicon-free search never produces exit transitions, so
    the penalty is attached to the label emissions themselves. `label-loop` is left at zero because
    with `collapse_repeated_labels` a repeated label produces no additional output.
    """
    rasr_config = RasrConfig()
    rasr_config.type = "transition"
    rasr_config.transition_preset = "none"
    rasr_config.extra_transition_types = "label-to-label,blank-to-label,initial-label"
    rasr_config.label_to_label_score = label_penalty
    rasr_config.blank_to_label_score = label_penalty
    rasr_config.initial_label_score = label_penalty

    return rasr_config


def _get_label_scorer_configs(
    model_kwargs: dict, checkpoint: PtCheckpoint, variant: SLMRecogVariant
) -> List[RasrConfig]:
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0
    labelsync = isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)

    speech_lm_score_scale = variant.speech_lm_score_scale
    if speech_lm_score_scale is None:
        speech_lm_score_scale = 1.0 - variant.ctc_score_scale

    aed_label_scorer_config = get_speech_lm_label_scorer_config(
        model_kwargs=model_kwargs,
        checkpoint=checkpoint,
        use_gpu=use_gpu,
        scale=speech_lm_score_scale,
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
        assert variant.label_penalty is None, "Label penalty is only implemented for time-synchronous search"
        return list(filter(None, [aed_label_scorer_config, ctc_label_scorer_config]))
    else:
        assert ctc_label_scorer_config is not None
        # CTC first: it drives the search and generates the candidates. The label penalty is a
        # constant and goes before the speech LM, so that the blank/label balance is already
        # corrected when the candidates for the expensive speech LM stage are selected.
        label_scorer_configs = [ctc_label_scorer_config]
        if variant.label_penalty is not None:
            label_scorer_configs.append(_get_label_penalty_scorer_config(variant.label_penalty))
        label_scorer_configs.append(aed_label_scorer_config)
        return label_scorer_configs


def _run_single_variant(
    model_descriptor: str,
    checkpoint: PtCheckpoint,
    model_kwargs: dict,
    huggingface_repo_dir: tk.Path,
    variant: SLMRecogVariant,
    corpora: List[loquacious_datasets.EvalSet],
    recog_data_config_fn: Callable[[loquacious_datasets.EvalSet], DataConfig],
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
        recog_data_config_fn=recog_data_config_fn,
    )
