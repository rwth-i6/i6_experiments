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

from .....data.loquacious import datasets as loquacious_datasets
from .....data.loquacious.phoneme import PHONEME_SIZE
from .....model_pipelines.common.learning_rates import OCLRConfig
from .....model_pipelines.common.optimizer import RAdamConfig
from .....model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig
from .....model_pipelines.ffnn_transducer.pytorch_modules import FFNNTransducerConfig
from .....model_pipelines.ffnn_transducer.train import FFNNTransducerTrainOptions, TrainedFFNNTransducerModel, train


def get_model_config(
    num_layers: int = 12,
    layer_size: int = 512,
    num_att_heads: int = 8,
) -> FFNNTransducerConfig:
    return FFNNTransducerConfig(
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
            freq_max_num_masks=5,
            freq_mask_max_size=16,
        ),
        conformer_cfg=ConformerRelPosEncoderV1Config(
            num_layers=num_layers,
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
                    out_features=layer_size,
                ),
            ),
            block_cfg=ConformerRelPosBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                    input_dim=layer_size,
                    hidden_dim=4 * layer_size,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    dropout_broadcast_axes=None,
                ),
                mhsa_cfg=ConformerMHSARelPosV1Config(
                    input_dim=layer_size,
                    num_att_heads=num_att_heads,
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
                    channels=layer_size,
                    kernel_size=31,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    norm=LayerNormNC(layer_size),
                    dropout_broadcast_axes=None,
                ),
                modules=["ff", "conv", "mhsa", "ff"],
                scales=[0.5, 1.0, 1.0, 0.5],
            ),
        ),
        dropout=0.1,
        enc_dim=layer_size,
        pred_num_layers=2,
        pred_dim=640,
        pred_activation=torch.nn.Tanh(),
        context_history_size=1,
        context_embedding_dim=256,
        joiner_dim=1024,
        joiner_activation=torch.nn.Tanh(),
        target_size=PHONEME_SIZE + 1,
    )


def get_train_options() -> FFNNTransducerTrainOptions:
    train_data_config = loquacious_datasets.get_medium_phoneme_train_data()
    cv_data_config = loquacious_datasets.get_phoneme_cv_data()

    partition_epoch = train_data_config.oggzip_config.partition_epoch

    num_epochs = 40
    save_epochs = list(range(num_epochs * 3 // 4, num_epochs - 5, 5)) + list(range(num_epochs - 5, num_epochs + 1))
    save_subepochs = [epoch * partition_epoch for epoch in save_epochs]

    return FFNNTransducerTrainOptions(
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=save_subepochs,
        batch_size=12_000 * 160,
        accum_grad_multiple_step=2,
        optimizer_config=RAdamConfig(
            epsilon=1e-12,
            weight_decay=0.01,
            decoupled_weight_decay=True,
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
        enc_loss_scale=0.5,
        pred_loss_scale=0.0,
        gradient_clip=1.0,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=48,
        max_seqs=None,
        max_seq_length=None,
    )


def run(
    model_config: Optional[FFNNTransducerConfig] = None,
    train_options: Optional[FFNNTransducerTrainOptions] = None,
) -> TrainedFFNNTransducerModel:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    return train(options=train_options, model_config=model_config)
