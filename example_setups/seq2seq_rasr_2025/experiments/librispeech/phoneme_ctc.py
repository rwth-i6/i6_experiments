from dataclasses import dataclass, fields
from typing import List, Literal, Optional, Tuple

import torch
from i6_core.returnn import PtCheckpoint, ReturnnTrainingJob
from i6_models.assemblies.conformer import ConformerRelPosBlockV1Config, ConformerRelPosEncoderV1Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV2Config,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ...data.librispeech import datasets as librispeech_datasets
from ...data.librispeech import lm as librispeech_lm
from ...data.librispeech.lexicon import get_bliss_phoneme_lexicon
from ...data.librispeech.phoneme import PHONEME_SIZE
from ...model_pipelines.common.learning_rates import OCLRConfig
from ...model_pipelines.common.optimizer import AdamWConfig
from ...model_pipelines.common.recog import (
    RecogResult,
    recog_rasr_offline,
    recog_rasr_streaming,
)
from ...model_pipelines.common.recog_rasr_config import (
    get_no_op_label_scorer_config,
    get_tree_timesync_recog_config,
)
from ...model_pipelines.common.serializers import get_model_serializers
from ...model_pipelines.common.train import TrainOptions
from ...model_pipelines.ctc.prior import compute_priors
from ...model_pipelines.ctc.pytorch_modules import (
    ConformerCTCConfig,
    ConformerCTCRecogConfig,
    ConformerCTCRecogModel,
    SpecaugmentByLengthConfig,
)
from ...model_pipelines.ctc.train import train


def get_model_config() -> ConformerCTCConfig:
    return ConformerCTCConfig(
        logmel_cfg=LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
            n_fft=400,
        ),
        specaug_cfg=SpecaugmentByLengthConfig(
            start_epoch=21,
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=5,
            freq_mask_max_size=16,
        ),
        conformer_cfg=ConformerRelPosEncoderV1Config(
            num_layers=12,
            frontend=ModuleFactoryV1(
                GenericFrontendV1,
                GenericFrontendV1Config(
                    in_features=80,
                    layer_ordering=[
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Activation,
                        FrontendLayerType.Pool2d,
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Conv2d,
                        FrontendLayerType.Activation,
                        FrontendLayerType.Pool2d,
                    ],
                    conv_kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)],
                    conv_out_dims=[32, 64, 64, 32],
                    conv_strides=None,
                    conv_paddings=None,
                    pool_kernel_sizes=[(2, 1), (2, 1)],
                    pool_strides=None,
                    pool_paddings=None,
                    activations=[torch.nn.ReLU(), torch.nn.ReLU()],
                    out_features=512,
                ),
            ),
            block_cfg=ConformerRelPosBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=512,
                    hidden_dim=2048,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    dropout_broadcast_axes=None,
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=512,
                    num_att_heads=8,
                    att_weights_dropout=0.1,
                    dropout=0.1,
                    with_bias=True,
                    learnable_pos_emb=False,
                    rel_pos_clip=16,
                    with_linear_pos=True,
                    with_pos_bias=True,
                    separate_pos_emb_per_head=True,
                    pos_emb_dropout=0.0,
                    dropout_broadcast_axes=None,
                ),
                conv_cfg=ConformerConvolutionV2Config(
                    channels=512,
                    kernel_size=31,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    norm=LayerNormNC(512),
                    dropout_broadcast_axes=None,
                ),
                modules=["ff", "conv", "mhsa", "ff"],
                scales=[0.5, 1.0, 1.0, 0.5],
            ),
        ),
        dim=512,
        target_size=PHONEME_SIZE + 1,
        dropout=0.1,
    )


def get_train_options() -> TrainOptions:
    return TrainOptions(
        train_data_config=librispeech_datasets.get_default_phoneme_train_data(),
        cv_data_config=librispeech_datasets.get_default_phoneme_cv_data(),
        save_epochs=list(range(1500, 1900, 100)) + list(range(1900, 2001, 20)),
        batch_size=24_000 * 160,
        accum_grad_multiple_step=1,
        optimizer_config=AdamWConfig(
            epsilon=1e-16,
            weight_decay=0.01,
        ),
        lr_config=OCLRConfig(
            init_lr=7e-06,
            peak_lr=5e-04,
            decayed_lr=5e-05,
            final_lr=1e-07,
            inc_epochs=960,
            dec_epochs=960,
            final_epochs=80,
        ),
        gradient_clip=1.0,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        max_seqs=None,
        max_seq_length=None,
    )


def run_training(
    model_config: Optional[ConformerCTCConfig] = None,
    train_options: Optional[TrainOptions] = None,
) -> Tuple[ReturnnTrainingJob, ConformerCTCConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    train_job = train(options=train_options, model_config=model_config)

    return train_job, model_config


@dataclass
class RecogVariant:
    descriptor: str = "recog"
    use_streaming: bool = False
    compute_search_errors: bool = False
    prior_scale: float = 0.0
    blank_penalty: float = 0.0
    word_lm: Optional[Literal["4gram", "trafo", "kazuki-trafo"]] = None
    word_lm_scale: float = 0.0
    max_beam_size: int = 1
    max_word_end_beam_size: Optional[int] = None
    score_threshold: float = 0.0
    word_end_score_threshold: Optional[float] = None
    maximum_stable_delay: int = 15
    maximum_stable_delay_pruning_interval: int = 5
    chunk_history_seconds: float = 10.0
    chunk_center_seconds: float = 1.0
    chunk_future_seconds: float = 1.0
    encoder_frame_shift_seconds: float = 0.04
    mem_rqmt: int = 16
    gpu_mem_rqmt: int = 0


def default_offline_4gram_recog_variant() -> RecogVariant:
    return RecogVariant(
        descriptor="recog_word-4gram",
        use_streaming=False,
        prior_scale=0.2,
        word_lm="4gram",
        word_lm_scale=0.6,
        max_beam_size=1024,
        max_word_end_beam_size=16,
        score_threshold=14.0,
        word_end_score_threshold=0.5,
    )


def default_offline_trafo_recog_variant() -> RecogVariant:
    return RecogVariant(
        descriptor="recog_word-trafoLM",
        use_streaming=False,
        prior_scale=0.2,
        word_lm="kazuki-trafo",
        word_lm_scale=0.8,
        max_beam_size=512,
        max_word_end_beam_size=16,
        score_threshold=14.0,
        word_end_score_threshold=0.5,
        gpu_mem_rqmt=24,
    )


def default_streaming_4gram_recog_variant() -> RecogVariant:
    return RecogVariant(
        descriptor="recog_streaming_word-4gram",
        use_streaming=True,
        prior_scale=0.2,
        word_lm="4gram",
        word_lm_scale=0.6,
        max_beam_size=1024,
        score_threshold=14.0,
        word_end_score_threshold=0.5,
        maximum_stable_delay=15,
    )


def default_recog_variants() -> List[RecogVariant]:
    return [
        default_offline_4gram_recog_variant(),
        # default_offline_trafo_recog_variant(),
        default_streaming_4gram_recog_variant(),
    ]


def run_recog_variants(
    checkpoint: Optional[PtCheckpoint] = None,
    model_config: Optional[ConformerCTCConfig] = None,
    train_options: Optional[TrainOptions] = None,
    variants: Optional[List[RecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if model_config is None:
        model_config = get_model_config()
    if checkpoint is None:
        train_job, _ = run_training(model_config=model_config, train_options=train_options)
        checkpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
        assert checkpoint is not None

    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = librispeech_datasets.EVAL_SETS

    recog_results = []

    blank_index = model_config.target_size - 1

    if any(variant.prior_scale != 0.0 for variant in variants):
        prior_file = compute_priors(
            prior_data_config=librispeech_datasets.get_default_prior_data(),
            model_config=model_config,
            checkpoint=checkpoint,
        )
    else:
        prior_file = None

    for variant in variants:
        model_serializers = get_model_serializers(
            ConformerCTCRecogModel,
            ConformerCTCRecogConfig(
                **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
                prior_file=prior_file if variant.prior_scale != 0.0 else None,
                prior_scale=variant.prior_scale,
                blank_penalty=variant.blank_penalty,
            ),
        )

        label_scorer_config = get_no_op_label_scorer_config()

        align_rasr_config_file = None

        lexicon_file = get_bliss_phoneme_lexicon()
        if variant.word_lm == "4gram":
            lm_config = librispeech_lm.get_arpa_lm_config(
                lm_name="4gram", lexicon_file=lexicon_file, scale=variant.word_lm_scale
            )
        elif variant.word_lm == "trafo":
            lm_config = librispeech_lm.get_transformer_lm_config(lm_scale=variant.word_lm_scale)
        elif variant.word_lm == "kazuki-trafo":
            lm_config = librispeech_lm.get_kazuki_trafo_lm_config(lm_scale=variant.word_lm_scale)
        else:
            lm_config = None

        recog_rasr_config_file = get_tree_timesync_recog_config(
            lexicon_file=lexicon_file,
            collapse_repeated_labels=True,
            label_scorer_config=label_scorer_config,
            lm_config=lm_config,
            blank_index=blank_index,
            max_beam_sizes=variant.max_beam_size,
            max_word_end_beam_size=variant.max_word_end_beam_size,
            score_thresholds=variant.score_threshold,
            word_end_score_threshold=variant.word_end_score_threshold,
            maximum_stable_delay=variant.maximum_stable_delay if variant.use_streaming else None,
            maximum_stable_delay_pruning_interval=(
                variant.maximum_stable_delay_pruning_interval if variant.use_streaming else None
            ),
            logfile_suffix="recog",
        )

        if variant.compute_search_errors:
            align_rasr_config_file = get_tree_timesync_recog_config(
                lexicon_file=lexicon_file,
                collapse_repeated_labels=True,
                label_scorer_config=label_scorer_config,
                lm_config=lm_config,
                blank_index=blank_index,
                max_beam_sizes=4096,
                score_thresholds=22.0,
                logfile_suffix="align",
            )

        for recog_corpus in corpora:
            if variant.use_streaming:
                recog_results.append(
                    recog_rasr_streaming(
                        descriptor=variant.descriptor,
                        checkpoint=checkpoint,
                        recog_rasr_config_file=recog_rasr_config_file,
                        recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                        recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                        encoder_serializers=model_serializers,
                        encoder_frame_shift_seconds=variant.encoder_frame_shift_seconds,
                        chunk_history_seconds=variant.chunk_history_seconds,
                        chunk_center_seconds=variant.chunk_center_seconds,
                        chunk_future_seconds=variant.chunk_future_seconds,
                        sample_rate=16000,
                        gpu_mem_rqmt=variant.gpu_mem_rqmt,
                        mem_rqmt=variant.mem_rqmt,
                    )
                )
            else:
                recog_results.append(
                    recog_rasr_offline(
                        descriptor=variant.descriptor,
                        checkpoint=checkpoint,
                        recog_rasr_config_file=recog_rasr_config_file,
                        align_rasr_config_file=align_rasr_config_file,
                        recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                        recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                        encoder_serializers=model_serializers,
                        sample_rate=16000,
                        gpu_mem_rqmt=variant.gpu_mem_rqmt,
                        mem_rqmt=variant.mem_rqmt,
                    )
                )
    return recog_results


def run_all() -> List[RecogResult]:
    train_job, model_config = run_training()
    checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
    return run_recog_variants(checkpoint=checkpoint, model_config=model_config)
