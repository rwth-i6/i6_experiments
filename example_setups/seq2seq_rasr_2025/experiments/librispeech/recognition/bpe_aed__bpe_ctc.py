from dataclasses import dataclass, fields
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_experiments.common.setups.serialization import Collection

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.aed.export import export_encoder as export_aed_encoder
from ....model_pipelines.aed.label_scorer_config import get_aed_label_scorer_config
from ....model_pipelines.aed.train import TrainedAEDModel
from ....model_pipelines.common.label_scorer_config import get_encoder_decoder_label_scorer_config
from ....model_pipelines.common.pytorch_modules import LogMelFeatureExtractionV1Model
from ....model_pipelines.common.recog import RecogResult
from ....model_pipelines.common.recog_rasr_config import LexiconfreeLabelsyncRecogParams, LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.ctc.export import export_model as export_ctc_model
from ....model_pipelines.ctc.label_scorer_config import get_ctc_prefix_label_scorer_config
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCRecogConfig
from ....model_pipelines.ctc.train import TrainedCTCModel
from .common import BaseRecogVariant, run_single_bpe_variant


@dataclass
class AEDCTCRecogVariant(BaseRecogVariant):
    aed_epoch: Optional[int] = None
    ctc_epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    ctc_score_scale: float = 0.0
    ctc_prior_scale: float = 0.0
    ctc_blank_penalty: float = 0.0


def default_lexfree_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="recog_lexfree_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            max_beam_sizes=[16, 8],
            length_norm_scale=1.2,
        ),
        ctc_score_scale=0.3,
    )


def default_lexfree_timesync_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="recog_lexfree_timesync",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            max_beam_sizes=[16, 8],
            collapse_repeated_labels=True,
        ),
        ctc_score_scale=0.3,
    )


def default_tree_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="recog_tree_timesync",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16, 8],
        ),
        ctc_score_scale=0.3,
    )


def default_tree_4gram_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="recog_tree_timesync_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16, 8],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        ctc_prior_scale=0.2,
        ctc_score_scale=0.3,
    )


def default_recog_variants() -> List[AEDCTCRecogVariant]:
    return [
        default_lexfree_recog_variant(),
        default_lexfree_timesync_recog_variant(),
        default_tree_recog_variant(),
        default_tree_4gram_recog_variant(),
    ]


def _get_label_scorer_configs(
    aed_model: TrainedAEDModel, ctc_model: TrainedCTCModel, variant: AEDCTCRecogVariant
) -> List[RasrConfig]:
    assert variant.ctc_score_scale != 0 and variant.ctc_prior_scale != 1

    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0

    aed_checkpoint = aed_model.get_checkpoint(variant.aed_epoch)
    aed_onnx_encoder = export_aed_encoder(model_config=aed_model.model_config, checkpoint=aed_checkpoint)
    aed_decoder_label_scorer_config = get_aed_label_scorer_config(
        model_config=aed_model.model_config,
        checkpoint=aed_checkpoint,
        use_gpu=use_gpu,
        scale=1.0 - variant.ctc_score_scale,
    )
    aed_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_onnx_model=aed_onnx_encoder, decoder_label_scorer_config=aed_decoder_label_scorer_config
    )

    ctc_checkpoint = ctc_model.get_checkpoint(variant.ctc_epoch)
    if variant.ctc_prior_scale != 0.0:
        prior_file = compute_priors(
            prior_data_config=librispeech_datasets.get_default_prior_data(),
            model_config=ctc_model.model_config,
            checkpoint=ctc_checkpoint,
        )
    else:
        prior_file = None
    ctc_onnx_model = export_ctc_model(
        model_config=ConformerCTCRecogConfig(
            **{f.name: getattr(ctc_model.model_config, f.name) for f in fields(ctc_model.model_config)},
            prior_file=prior_file,
            prior_scale=variant.ctc_prior_scale,
            blank_penalty=variant.ctc_blank_penalty,
        ),
        checkpoint=ctc_checkpoint,
    )

    if isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams):
        ctc_decoder_label_scorer_config = get_ctc_prefix_label_scorer_config(
            model_config=ctc_model.model_config,
            scale=variant.ctc_score_scale,
        )
    else:
        ctc_decoder_label_scorer_config = None

    ctc_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_onnx_model=ctc_onnx_model,
        decoder_label_scorer_config=ctc_decoder_label_scorer_config,
        scale=variant.ctc_score_scale,
        use_gpu=use_gpu,
    )

    assert aed_model.model_config.label_target_size + 1 == ctc_model.model_config.target_size
    bpe_size = vocab_to_bpe_size(aed_model.model_config.label_target_size)

    lstm_lm_label_scorer_config = None
    if variant.bpe_lstm_lm_scale != 0.0:
        lstm_lm_label_scorer_config = librispeech_lm.get_bpe_lstm_label_scorer_config(
            bpe_size=bpe_size,
            scale=variant.bpe_lstm_lm_scale,
            use_gpu=use_gpu,
        )

    if isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams):
        return list(filter(None, [aed_label_scorer_config, ctc_label_scorer_config, lstm_lm_label_scorer_config]))
    else:
        assert ctc_label_scorer_config is not None
        return list(filter(None, [ctc_label_scorer_config, aed_label_scorer_config, lstm_lm_label_scorer_config]))


def _get_feature_extraction_serializers(aed_model: TrainedAEDModel, ctc_model: TrainedCTCModel) -> Collection:
    assert aed_model.model_config.logmel_cfg == ctc_model.model_config.logmel_cfg
    return get_model_serializers(LogMelFeatureExtractionV1Model, aed_model.model_config.logmel_cfg)


def _run_single_variant(
    aed_model: TrainedAEDModel,
    ctc_model: TrainedCTCModel,
    variant: AEDCTCRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    if isinstance(
        variant.search_algorithm_params, (LexiconfreeTimesyncRecogParams, LibrispeechTreeTimesyncRecogParams)
    ):
        blank_index = ctc_model.model_config.target_size - 1
    else:
        blank_index = None

    return run_single_bpe_variant(
        encoder_serializers=_get_feature_extraction_serializers(aed_model=aed_model, ctc_model=ctc_model),
        label_scorer_configs=_get_label_scorer_configs(aed_model=aed_model, ctc_model=ctc_model, variant=variant),
        bpe_size=vocab_to_bpe_size(ctc_model.model_config.target_size - 1),
        blank_index=blank_index,
        sentence_end_index=0,
        variant=variant,
        corpora=corpora,
        checkpoint=None,  # Checkpoints are contained in ONNX models and loaded by RASR
    )


def run(
    aed_model: TrainedAEDModel,
    ctc_model: TrainedCTCModel,
    variants: Optional[List[AEDCTCRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = librispeech_datasets.EVAL_SETS

    results = []

    for variant in variants:
        results.extend(_run_single_variant(aed_model=aed_model, ctc_model=ctc_model, variant=variant, corpora=corpora))
    return results
