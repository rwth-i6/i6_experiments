from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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

from ....data.loquacious import datasets as loquacious_datasets
from ....data.loquacious.bpe import bpe_to_vocab_size, get_bpe_vocab_file, vocab_to_bpe_size
from ....data.loquacious.datasets import (
    get_medium_bpe_cv_data,
    get_medium_bpe_train_data,
)
from ....model_pipelines.aed.label_scorer_config import get_aed_label_scorer_config, get_ctc_prefix_label_scorer_config
from ....model_pipelines.aed.pytorch_modules import (
    AdditiveAttentionConfig,
    AEDConfig,
    AEDEncoder,
    AttentionLSTMDecoderV1Config,
)
from ....model_pipelines.aed.train import AEDTrainOptions, train
from ....model_pipelines.common.learning_rates import ConstConstDecayLRConfig
from ....model_pipelines.common.optimizer import RAdamConfig
from ....model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig
from ....model_pipelines.common.recog import (
    RecogResult,
    recog_rasr_offline,
)
from ....model_pipelines.common.recog_rasr_config import get_lexiconfree_labelsync_recog_config
from ....model_pipelines.common.serializers import get_model_serializers


def get_model_config(bpe_size: int = 128) -> AEDConfig:
    vocab_size = bpe_to_vocab_size(bpe_size=bpe_size)
    return AEDConfig(
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
            start_epoch=51,
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=3,
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
                    pool_kernel_sizes=[(3, 1), (2, 1)],
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
        final_dropout=0.1,
        enc_dim=512,
        decoder_config=AttentionLSTMDecoderV1Config(
            encoder_dim=512,
            vocab_size=vocab_size,
            target_embed_dim=640,
            target_embed_dropout=0.1,
            lstm_hidden_size=1024,
            zoneout_drop_h=0.05,
            zoneout_drop_c=0.15,
            output_proj_dim=1024,
            output_dropout=0.3,
            attention_cfg=AdditiveAttentionConfig(
                attention_dim=1024,
                att_weights_dropout=0.1,
            ),
        ),
        label_target_size=vocab_size,
    )


def get_train_options(bpe_size: int = 128) -> AEDTrainOptions:
    train_data_config = get_medium_bpe_train_data(bpe_size=bpe_size)
    assert train_data_config.target_config
    train_data_config.target_config["seq_postfix"] = [0]

    cv_data_config = get_medium_bpe_cv_data(bpe_size=bpe_size)
    assert cv_data_config.target_config
    cv_data_config.target_config["seq_postfix"] = [0]

    partition_epoch = train_data_config.partition_epoch

    num_epochs = 40
    save_epochs = list(range(num_epochs * 3 // 4, num_epochs - 5, 5)) + list(range(num_epochs - 5, num_epochs + 1))
    save_subepochs = [epoch * partition_epoch for epoch in save_epochs]

    return AEDTrainOptions(
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=save_subepochs,
        batch_size=24_000 * 160,
        accum_grad_multiple_step=1,
        optimizer_config=RAdamConfig(
            epsilon=1e-12,
            weight_decay=0.01,
            decoupled_weight_decay=True,
        ),
        lr_config=ConstConstDecayLRConfig(
            const_lr_1=5e-05,
            const_lr_2=5e-04,
            decayed_lr=5e-05,
            final_lr=1e-07,
            const_epochs_1=2 * partition_epoch,
            const_epochs_2=(num_epochs // 2 - 4) * partition_epoch,
            dec_epochs=(num_epochs // 2 - 2) * partition_epoch,
            final_epochs=4 * partition_epoch,
        ),
        gradient_clip=1.0,
        ctc_loss_scale=0.7,
        label_smoothing=0.1,
        label_smoothing_start_epoch=3 * partition_epoch + 1,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        max_seqs=None,
        max_seq_length=None,
    )


def run_training(
    model_config: Optional[AEDConfig] = None,
    train_options: Optional[AEDTrainOptions] = None,
) -> Tuple[ReturnnTrainingJob, AEDConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    train_job = train(options=train_options, model_config=model_config)

    return train_job, model_config


@dataclass
class RecogVariant:
    descriptor: str = "recog"
    compute_search_errors: bool = False
    max_beam_sizes: Union[int, List[int]] = 64
    ctc_score_scale: float = 0.0
    length_norm_scale: float = 1.2
    max_labels_per_time_step: int = 1
    max_word_end_beam_size: Optional[int] = None
    score_thresholds: Optional[Union[float, List[float]]] = 8.0
    encoder_frame_shift_seconds: float = 0.04
    mem_rqmt: int = 16
    gpu_mem_rqmt: int = 0


def default_lexfree_recog_variant() -> RecogVariant:
    return RecogVariant(
        descriptor="recog_lexfree",
    )


def default_lexfree_aed_ctc_recog_variant() -> RecogVariant:
    return RecogVariant(
        descriptor="recog_lexfree_+ctc",
        max_beam_sizes=[64, 32],
        score_thresholds=None,
        ctc_score_scale=0.1,
    )


def default_recog_variants() -> List[RecogVariant]:
    return [
        default_lexfree_recog_variant(),
        default_lexfree_aed_ctc_recog_variant(),
    ]


def run_recog_variants(
    checkpoint: Optional[PtCheckpoint] = None,
    model_config: Optional[AEDConfig] = None,
    train_options: Optional[AEDTrainOptions] = None,
    variants: Optional[List[RecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
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
        corpora = loquacious_datasets.EVAL_SETS

    recog_results = []

    bpe_size = vocab_to_bpe_size(model_config.label_target_size)

    for variant in variants:
        model_serializers = get_model_serializers(AEDEncoder, model_config)

        use_ctc_scorer = variant.ctc_score_scale != 0.0

        label_scorer_config = get_aed_label_scorer_config(
            model_config=model_config,
            checkpoint=checkpoint,
            scale=1.0 - variant.ctc_score_scale,
            execution_provider_type="cuda" if variant.gpu_mem_rqmt > 0 else None,
        )

        if use_ctc_scorer:
            ctc_label_scorer_config = get_ctc_prefix_label_scorer_config(
                model_config=model_config,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                execution_provider_type="cuda" if variant.gpu_mem_rqmt > 0 else None,
            )
            label_scorer_config = [label_scorer_config, ctc_label_scorer_config]

        align_rasr_config_file = None

        vocab_file = get_bpe_vocab_file(bpe_size=bpe_size, corpus_key="train.medium", add_blank=False)
        recog_rasr_config_file = get_lexiconfree_labelsync_recog_config(
            vocab_file=vocab_file,
            label_scorer_config=label_scorer_config,
            sentence_end_index=0,
            max_beam_sizes=variant.max_beam_sizes,
            score_thresholds=variant.score_thresholds,
            length_norm_scale=variant.length_norm_scale,
            max_labels_per_time_step=variant.max_labels_per_time_step,
            logfile_suffix="recog",
        )
        if variant.compute_search_errors:
            align_rasr_config_file = get_lexiconfree_labelsync_recog_config(
                vocab_file=vocab_file,
                label_scorer_config=label_scorer_config,
                sentence_end_index=0,
                max_beam_sizes=[256, 256] if use_ctc_scorer else 256,
                score_thresholds=[18.0, 14.0] if use_ctc_scorer else 14.0,
                length_norm_scale=variant.length_norm_scale,
                max_labels_per_time_step=variant.max_labels_per_time_step,
                logfile_suffix="align",
            )

        for recog_corpus in corpora:
            recog_results.append(
                recog_rasr_offline(
                    descriptor=variant.descriptor,
                    checkpoint=checkpoint,
                    recog_rasr_config_file=recog_rasr_config_file,
                    align_rasr_config_file=align_rasr_config_file,
                    recog_data_config=loquacious_datasets.get_default_recog_data(recog_corpus),
                    recog_corpus=loquacious_datasets.get_default_score_corpus(recog_corpus),
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
