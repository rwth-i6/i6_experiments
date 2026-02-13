from typing import Optional

import torch
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

from .....data.loquacious.bpe import bpe_to_vocab_size
from .....data.loquacious.datasets import get_medium_bpe_cv_data, get_medium_bpe_train_data
from .....model_pipelines.aed.pytorch_modules import AdditiveAttentionConfig, AttentionLSTMDecoderV1Config
from .....model_pipelines.combination_model.pytorch_modules import CombinationModelConfig
from .....model_pipelines.combination_model.train import CombinationTrainOptions, TrainedCombinationModel, train
from .....model_pipelines.common.learning_rates import OCLRConfig
from .....model_pipelines.common.optimizer import AdamWConfig
from .....model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig


def get_model_config(bpe_size: int = 128) -> CombinationModelConfig:
    vocab_size = bpe_to_vocab_size(bpe_size=bpe_size)
    return CombinationModelConfig(
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
        enc_dim=512,
        attention_decoder_config=AttentionLSTMDecoderV1Config(
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
        transducer_pred_num_layers=2,
        transducer_pred_dim=640,
        transducer_pred_activation=torch.nn.Tanh(),
        transducer_context_history_size=1,
        transducer_context_embedding_dim=256,
        transducer_joiner_dim=1024,
        transducer_joiner_activation=torch.nn.Tanh(),
        transducer_decoder_dropout=0.1,
        ctc_dropout=0.1,
        target_size=vocab_size + 1,
    )


def get_train_options(bpe_size: int = 128) -> CombinationTrainOptions:
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

    return CombinationTrainOptions(
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=save_subepochs,
        batch_size=8_000 * 160,
        accum_grad_multiple_step=3,
        optimizer_config=AdamWConfig(
            epsilon=1e-16,
            weight_decay=0.01,
        ),
        lr_config=OCLRConfig(
            init_lr=7e-06,
            peak_lr=5e-04,
            decayed_lr=5e-05,
            final_lr=1e-07,
            inc_epochs=(num_epochs - 4) // 2 * partition_epoch,
            dec_epochs=(num_epochs - 4) // 2 * partition_epoch,
            final_epochs=4 * partition_epoch,
        ),
        gradient_clip=1.0,
        ctc_loss_scale=0.7,
        transducer_loss_scale=1.0,
        attention_loss_scale=1.0,
        attention_label_smoothing=0.1,
        attention_label_smoothing_start_epoch=3 * partition_epoch + 1,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        max_seqs=None,
        max_seq_length=None,
    )


def run(
    model_config: Optional[CombinationModelConfig] = None,
    train_options: Optional[CombinationTrainOptions] = None,
) -> TrainedCombinationModel:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    return train(options=train_options, model_config=model_config)
