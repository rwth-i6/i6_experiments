from dataclasses import dataclass
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
from ....model_pipelines.ffnn_transducer.label_scorer_config import get_ffnn_transducer_label_scorer_config
from ....model_pipelines.ffnn_transducer.pytorch_modules import FFNNTransducerConfig, FFNNTransducerEncoder
from ....model_pipelines.speech_llm.label_scorer_config import get_speech_lm_label_scorer_config
from ....model_pipelines.speech_llm.python_encoder import (
    SPEECH_LM_PYTHON_ENCODER_TYPE,
    register_speech_lm_encoder_type,
)
from .common import BaseRecogVariant, run_single_hf_tokenized_byte_tree_variant

TRANSDUCER_PYTHON_ENCODER_TYPE = "ffnn-transducer-python-encoder"


@dataclass
class TransducerByteSpeechLmRecogVariant(BaseRecogVariant):
    transducer_epoch: Optional[int] = None
    speech_lm_score_scale: float = 0.3
    transducer_score_scale: float = 0.7
    ilm_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model: TrainedModel[FFNNTransducerConfig],
    speech_lm_model_kwargs: dict,
    speech_lm_checkpoint: PtCheckpoint,
    huggingface_repo_dir: tk.Path,
    variants: Optional[List[TransducerByteSpeechLmRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = loquacious_datasets.EVAL_SETS

    results = []
    for variant in variants:
        results.extend(
            _run_single_variant(
                model=model,
                speech_lm_model_kwargs=speech_lm_model_kwargs,
                speech_lm_checkpoint=speech_lm_checkpoint,
                huggingface_repo_dir=huggingface_repo_dir,
                variant=variant,
                corpora=corpora,
            )
        )
    return results


def default_recog_variants() -> List[TransducerByteSpeechLmRecogVariant]:
    return [default_tree_speech_lm_recog_variant()]


def default_tree_speech_lm_recog_variant() -> TransducerByteSpeechLmRecogVariant:
    return TransducerByteSpeechLmRecogVariant(
        descriptor="recog_tree_speech_llm",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256, 16],
            score_thresholds=[14.0, 8.0],
            max_word_end_beam_size=16,
            word_end_score_threshold=2.0,
            recombination_mode="on",
        ),
    )


def _get_label_scorer_configs(
    model: TrainedModel[FFNNTransducerConfig],
    speech_lm_model_kwargs: dict,
    speech_lm_checkpoint: PtCheckpoint,
    variant: TransducerByteSpeechLmRecogVariant,
) -> List[RasrConfig]:
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0
    transducer_checkpoint = model.get_checkpoint(variant.transducer_epoch)
    transducer_encoder = RasrConfig()
    transducer_encoder.type = TRANSDUCER_PYTHON_ENCODER_TYPE
    transducer_encoder.device = "cuda" if use_gpu else "cpu"
    transducer_label_scorer_config = get_encoder_decoder_label_scorer_config(
        encoder_config=transducer_encoder,
        decoder_label_scorer_config=get_ffnn_transducer_label_scorer_config(
            model_config=model.model_config,
            checkpoint=transducer_checkpoint,
            ilm_scale=variant.ilm_scale,
            blank_penalty=variant.blank_penalty,
            use_gpu=use_gpu,
        ),
        scale=variant.transducer_score_scale,
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

    return [transducer_label_scorer_config, speech_lm_label_scorer_config]


def _get_encoder_serializers(model_config: FFNNTransducerConfig, transducer_checkpoint: PtCheckpoint) -> Collection:
    raw_audio_serializers = get_model_serializers(RawAudioModel, NoConfig()).serializer_objects
    transducer_encoder_serializers = get_pytorch_encoder_serializers(
        encoder_type_name=TRANSDUCER_PYTHON_ENCODER_TYPE,
        model_class=FFNNTransducerEncoder,
        model_config=model_config,
        checkpoint=transducer_checkpoint,
    ).serializer_objects
    return Collection(
        raw_audio_serializers
        + transducer_encoder_serializers
        + [
            Import(f"{register_speech_lm_encoder_type.__module__}.{register_speech_lm_encoder_type.__name__}"),
            get_rasr_python_encoder_init_hook_serializer(
                [
                    f"register_{TRANSDUCER_PYTHON_ENCODER_TYPE.replace('-', '_')}",
                    register_speech_lm_encoder_type.__name__,
                ]
            ),
        ]
    )


def _run_single_variant(
    model: TrainedModel[FFNNTransducerConfig],
    speech_lm_model_kwargs: dict,
    speech_lm_checkpoint: PtCheckpoint,
    huggingface_repo_dir: tk.Path,
    variant: TransducerByteSpeechLmRecogVariant,
    corpora: List[loquacious_datasets.EvalSet],
) -> List[RecogResult]:
    transducer_checkpoint = model.get_checkpoint(variant.transducer_epoch)
    return run_single_hf_tokenized_byte_tree_variant(
        model_descriptor=f"{model.descriptor}__speech_llm",
        huggingface_repo_dir=huggingface_repo_dir,
        encoder_serializers=_get_encoder_serializers(
            model_config=model.model_config,
            transducer_checkpoint=transducer_checkpoint,
        ),
        rasr_init_hook=CodeWrapper("register_rasr_python_encoders"),
        label_scorer_configs=_get_label_scorer_configs(
            model=model,
            speech_lm_model_kwargs=speech_lm_model_kwargs,
            speech_lm_checkpoint=speech_lm_checkpoint,
            variant=variant,
        ),
        blank_index=model.model_config.target_size - 1,
        variant=variant,
        corpora=corpora,
    )
