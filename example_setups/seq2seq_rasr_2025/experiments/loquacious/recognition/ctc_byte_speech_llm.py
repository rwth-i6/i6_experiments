from dataclasses import dataclass, fields
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_core.returnn import CodeWrapper, PtCheckpoint
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import tk

from ....data.loquacious import datasets as loquacious_datasets
from ....data.loquacious.recog import LoquaciousTreeTimesyncRecogParams
from ....model_pipelines.common.label_scorer_config import get_encoder_decoder_label_scorer_config
from ....model_pipelines.common.python_encoder import (
    get_pytorch_encoder_serializers,
    get_rasr_python_encoder_init_hook_serializer,
)
from ....model_pipelines.common.pytorch_modules import NoConfig, RawAudioModel
from ....model_pipelines.common.recog import RecogResult
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.common.train import TrainedModel
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCConfig, ConformerCTCRecogConfig, ConformerCTCRecogModel
from ....model_pipelines.speech_llm.label_scorer_config import get_speech_lm_label_scorer_config
from ....model_pipelines.speech_llm.python_encoder import (
    SPEECH_LM_PYTHON_ENCODER_TYPE,
    register_speech_lm_encoder_type,
)
from .common import BaseRecogVariant, run_single_hf_tokenized_byte_tree_variant

CTC_PYTHON_ENCODER_TYPE = "ctc-python-encoder"


@dataclass
class CTCByteSpeechLmRecogVariant(BaseRecogVariant):
    ctc_epoch: Optional[int] = None
    speech_lm_score_scale: float = 0.7
    ctc_score_scale: float = 0.3
    ctc_prior_scale: float = 0.0
    ctc_blank_penalty: float = 0.0


def run(
    model: TrainedModel[ConformerCTCConfig],
    speech_lm_model_kwargs: dict,
    speech_lm_checkpoint: PtCheckpoint,
    huggingface_repo_dir: tk.Path,
    train_corpus_key: loquacious_datasets.TrainSet = "train.medium",
    variants: Optional[List[CTCByteSpeechLmRecogVariant]] = None,
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
                model=model,
                speech_lm_model_kwargs=speech_lm_model_kwargs,
                speech_lm_checkpoint=speech_lm_checkpoint,
                huggingface_repo_dir=huggingface_repo_dir,
                train_corpus_key=train_corpus_key,
                variant=variant,
                corpora=corpora,
            )
        )
    return results


def default_recog_variants() -> List[CTCByteSpeechLmRecogVariant]:
    return [default_tree_speech_lm_recog_variant()]


def default_tree_speech_lm_recog_variant() -> CTCByteSpeechLmRecogVariant:
    return CTCByteSpeechLmRecogVariant(
        descriptor="recog_tree_speech_llm",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[64, 64],
            score_thresholds=[10.0, 10.0],
            max_word_end_beam_size=16,
            word_end_score_threshold=6.0,
            recombination_mode="on",
        ),
    )


def _get_label_scorer_configs(
    speech_lm_model_kwargs: dict,
    speech_lm_checkpoint: PtCheckpoint,
    variant: CTCByteSpeechLmRecogVariant,
) -> List[RasrConfig]:
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0
    ctc_encoder = RasrConfig()
    ctc_encoder.type = CTC_PYTHON_ENCODER_TYPE
    ctc_encoder.device = "cuda" if use_gpu else "cpu"
    ctc_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_config=ctc_encoder,
        scale=variant.ctc_score_scale,
        use_gpu=use_gpu,
    )

    speech_lm_encoder = RasrConfig()
    speech_lm_encoder.type = SPEECH_LM_PYTHON_ENCODER_TYPE
    speech_lm_encoder.checkpoint_path = str(speech_lm_checkpoint)
    speech_lm_encoder.device = "cuda" if use_gpu else "cpu"
    speech_lm_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_config=speech_lm_encoder,
        decoder_label_scorer_config=get_speech_lm_label_scorer_config(
            model_kwargs=speech_lm_model_kwargs,
            checkpoint=speech_lm_checkpoint,
            scale=1.0,
            use_gpu=use_gpu,
            only_score_exits=True,
        ),
        scale=variant.speech_lm_score_scale,
        use_gpu=use_gpu,
    )

    return [ctc_label_scorer_config, speech_lm_label_scorer_config]


def _get_ctc_recog_config(
    model: TrainedModel[ConformerCTCConfig],
    ctc_checkpoint: PtCheckpoint,
    variant: CTCByteSpeechLmRecogVariant,
    train_corpus_key: loquacious_datasets.TrainSet,
) -> ConformerCTCRecogConfig:
    prior_file = None
    if variant.ctc_prior_scale != 0.0:
        prior_file = compute_priors(
            prior_data_config=loquacious_datasets.get_prior_data(train_corpus_key=train_corpus_key),
            model_config=model.model_config,
            checkpoint=ctc_checkpoint,
        )

    return ConformerCTCRecogConfig(
        **{f.name: getattr(model.model_config, f.name) for f in fields(model.model_config)},
        prior_file=prior_file,
        prior_scale=variant.ctc_prior_scale,
        blank_penalty=variant.ctc_blank_penalty,
    )


def _get_encoder_serializers(ctc_recog_config: ConformerCTCRecogConfig, ctc_checkpoint: PtCheckpoint) -> Collection:
    raw_audio_serializers = get_model_serializers(RawAudioModel, NoConfig()).serializer_objects
    ctc_encoder_serializers = get_pytorch_encoder_serializers(
        encoder_type_name=CTC_PYTHON_ENCODER_TYPE,
        model_class=ConformerCTCRecogModel,
        model_config=ctc_recog_config,
        checkpoint=ctc_checkpoint,
    ).serializer_objects
    return Collection(
        raw_audio_serializers
        + ctc_encoder_serializers
        + [
            Import(f"{register_speech_lm_encoder_type.__module__}.{register_speech_lm_encoder_type.__name__}"),
            get_rasr_python_encoder_init_hook_serializer(
                [
                    f"register_{CTC_PYTHON_ENCODER_TYPE.replace('-', '_')}",
                    register_speech_lm_encoder_type.__name__,
                ]
            ),
        ]
    )


def _run_single_variant(
    model: TrainedModel[ConformerCTCConfig],
    speech_lm_model_kwargs: dict,
    speech_lm_checkpoint: PtCheckpoint,
    huggingface_repo_dir: tk.Path,
    train_corpus_key: loquacious_datasets.TrainSet,
    variant: CTCByteSpeechLmRecogVariant,
    corpora: List[loquacious_datasets.EvalSet],
) -> List[RecogResult]:
    ctc_checkpoint = model.get_checkpoint(variant.ctc_epoch)
    ctc_recog_config = _get_ctc_recog_config(
        model=model,
        ctc_checkpoint=ctc_checkpoint,
        variant=variant,
        train_corpus_key=train_corpus_key,
    )
    return run_single_hf_tokenized_byte_tree_variant(
        model_descriptor=f"{model.descriptor}__speech_llm",
        huggingface_repo_dir=huggingface_repo_dir,
        encoder_serializers=_get_encoder_serializers(ctc_recog_config=ctc_recog_config, ctc_checkpoint=ctc_checkpoint),
        rasr_init_hook=CodeWrapper("register_rasr_python_encoders"),
        label_scorer_configs=_get_label_scorer_configs(
            speech_lm_model_kwargs=speech_lm_model_kwargs,
            speech_lm_checkpoint=speech_lm_checkpoint,
            variant=variant,
        ),
        blank_index=model.model_config.target_size - 1,
        variant=variant,
        corpora=corpora,
        recog_data_config_fn=(lambda _corpus: loquacious_datasets.get_hf_recog_data(_corpus)),
    )
