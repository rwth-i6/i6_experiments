from dataclasses import dataclass, fields, replace
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_core.returnn import CodeWrapper, PtCheckpoint
from i6_experiments.common.setups.serialization import Collection

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.aed.label_scorer_config import get_aed_label_scorer_config
from ....model_pipelines.aed.pytorch_modules import AEDConfig, AEDEncoder
from ....model_pipelines.common.label_scorer_config import get_encoder_decoder_label_scorer_config
from ....model_pipelines.common.python_encoder import (
    get_pytorch_encoder_serializers,
    get_rasr_python_encoder_init_hook_serializer,
)
from ....model_pipelines.common.pytorch_modules import NoConfig, RawAudioModel
from ....model_pipelines.common.recog import OfflineRecogParameters, RecogResult
from ....model_pipelines.common.recog_rasr_config import LexiconfreeLabelsyncRecogParams, LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.common.train import TrainedModel
from ....model_pipelines.ctc.label_scorer_config import get_ctc_prefix_label_scorer_config
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCConfig, ConformerCTCRecogConfig, ConformerCTCRecogModel
from .common import BaseRecogVariant, run_single_bpe_variant

AED_PYTHON_ENCODER_TYPE = "aed-python-encoder"
CTC_PYTHON_ENCODER_TYPE = "ctc-python-encoder"


@dataclass
class AEDCTCRecogVariant(BaseRecogVariant):
    aed_epoch: Optional[int] = None
    ctc_epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    ctc_score_scale: float = 0.0
    ctc_prior_scale: float = 0.0
    ctc_blank_penalty: float = 0.0


def run(
    aed_model: TrainedModel[AEDConfig],
    ctc_model: TrainedModel[ConformerCTCConfig],
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


def default_recog_variants() -> List[AEDCTCRecogVariant]:
    return [
        default_lexfree_recog_variant(),
        default_lexfree_timesync_recog_variant(),
        default_tree_recog_variant(),
        default_tree_4gram_recog_variant(),
    ]


def default_lexfree_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="lexfree_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            score_thresholds=[4.0, 4.0],
            max_beam_sizes=[8, 4],
            length_norm_scale=0.0,
        ),
        ctc_score_scale=0.5,
    )


def default_lexfree_timesync_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="lexfree_timesync",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            max_beam_sizes=[1024, 32],
            score_thresholds=[4.0, 4.0],
            collapse_repeated_labels=True,
        ),
        ctc_score_scale=0.4,
    )


def default_tree_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="tree_timesync",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16, 8],
        ),
        ctc_score_scale=0.3,
    )


def default_tree_4gram_recog_variant() -> AEDCTCRecogVariant:
    return AEDCTCRecogVariant(
        descriptor="tree_timesync_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16, 8],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        ctc_prior_scale=0.2,
        ctc_score_scale=0.3,
    )


def _get_label_scorer_configs(
    aed_model: TrainedModel[AEDConfig],
    ctc_model: TrainedModel[ConformerCTCConfig],
    aed_checkpoint: PtCheckpoint,
    variant: AEDCTCRecogVariant,
) -> List[RasrConfig]:
    assert variant.ctc_score_scale != 0 and variant.ctc_prior_scale != 1

    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0

    aed_encoder = RasrConfig()
    aed_encoder.type = AED_PYTHON_ENCODER_TYPE
    aed_encoder.device = "cuda" if use_gpu else "cpu"
    aed_decoder_label_scorer_config = get_aed_label_scorer_config(
        model_config=aed_model.model_config,
        checkpoint=aed_checkpoint,
        use_gpu=use_gpu,
    )
    aed_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_config=aed_encoder,
        decoder_label_scorer_config=aed_decoder_label_scorer_config,
        scale=1.0 - variant.ctc_score_scale,
    )

    if isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams):
        ctc_decoder_label_scorer_config = get_ctc_prefix_label_scorer_config(model_config=ctc_model.model_config)
    else:
        ctc_decoder_label_scorer_config = None

    ctc_encoder = RasrConfig()
    ctc_encoder.type = CTC_PYTHON_ENCODER_TYPE
    ctc_encoder.device = "cuda" if use_gpu else "cpu"
    ctc_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_config=ctc_encoder,
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


def _get_ctc_recog_config(
    ctc_model: TrainedModel[ConformerCTCConfig],
    ctc_checkpoint: PtCheckpoint,
    variant: AEDCTCRecogVariant,
) -> ConformerCTCRecogConfig:
    if variant.ctc_prior_scale != 0.0:
        prior_file = compute_priors(
            prior_data_config=librispeech_datasets.get_default_prior_data(),
            model_config=ctc_model.model_config,
            checkpoint=ctc_checkpoint,
        )
    else:
        prior_file = None

    return ConformerCTCRecogConfig(
        **{f.name: getattr(ctc_model.model_config, f.name) for f in fields(ctc_model.model_config)},
        prior_file=prior_file,
        prior_scale=variant.ctc_prior_scale,
        blank_penalty=variant.ctc_blank_penalty,
    )


def _get_encoder_serializers(
    aed_model_config: AEDConfig,
    aed_checkpoint: PtCheckpoint,
    ctc_recog_config: ConformerCTCRecogConfig,
    ctc_checkpoint: PtCheckpoint,
) -> Collection:
    raw_audio_serializers = get_model_serializers(RawAudioModel, NoConfig()).serializer_objects
    aed_encoder_serializers = get_pytorch_encoder_serializers(
        encoder_type_name=AED_PYTHON_ENCODER_TYPE,
        model_class=AEDEncoder,
        model_config=aed_model_config,
        checkpoint=aed_checkpoint,
    ).serializer_objects
    ctc_encoder_serializers = get_pytorch_encoder_serializers(
        encoder_type_name=CTC_PYTHON_ENCODER_TYPE,
        model_class=ConformerCTCRecogModel,
        model_config=ctc_recog_config,
        checkpoint=ctc_checkpoint,
    ).serializer_objects
    return Collection(
        raw_audio_serializers
        + aed_encoder_serializers
        + ctc_encoder_serializers
        + [
            get_rasr_python_encoder_init_hook_serializer(
                [
                    f"register_{AED_PYTHON_ENCODER_TYPE.replace('-', '_')}",
                    f"register_{CTC_PYTHON_ENCODER_TYPE.replace('-', '_')}",
                ]
            ),
        ]
    )


def _run_single_variant(
    aed_model: TrainedModel[AEDConfig],
    ctc_model: TrainedModel[ConformerCTCConfig],
    variant: AEDCTCRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    if (
        isinstance(variant.search_mode_params, OfflineRecogParameters)
        and variant.search_mode_params.encoder_frame_shift_seconds is None
    ):
        variant = replace(
            variant,
            search_mode_params=replace(variant.search_mode_params, encoder_frame_shift_seconds=0.06),
        )

    if isinstance(
        variant.search_algorithm_params, (LexiconfreeTimesyncRecogParams, LibrispeechTreeTimesyncRecogParams)
    ):
        blank_index = ctc_model.model_config.target_size - 1
    else:
        blank_index = None

    aed_checkpoint = aed_model.get_checkpoint(variant.aed_epoch)
    ctc_checkpoint = ctc_model.get_checkpoint(variant.ctc_epoch)
    ctc_recog_config = _get_ctc_recog_config(
        ctc_model=ctc_model,
        ctc_checkpoint=ctc_checkpoint,
        variant=variant,
    )

    return run_single_bpe_variant(
        model_descriptor=f"{aed_model.descriptor}__{ctc_model.descriptor}",
        encoder_serializers=_get_encoder_serializers(
            aed_model_config=aed_model.model_config,
            aed_checkpoint=aed_checkpoint,
            ctc_recog_config=ctc_recog_config,
            ctc_checkpoint=ctc_checkpoint,
        ),
        rasr_init_hook=CodeWrapper("register_rasr_python_encoders"),
        label_scorer_configs=_get_label_scorer_configs(
            aed_model=aed_model,
            ctc_model=ctc_model,
            aed_checkpoint=aed_checkpoint,
            variant=variant,
        ),
        bpe_size=vocab_to_bpe_size(ctc_model.model_config.target_size - 1),
        blank_index=blank_index,
        sentence_end_index=0,
        variant=variant,
        corpora=corpora,
        checkpoint=None,
    )
