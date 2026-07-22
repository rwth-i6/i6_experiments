__all__ = ["run", "get_model_config", "get_train_options"]

from typing import Optional, Union

import torch
from i6_models.config import ModuleFactoryV1
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech.bpe import bpe_to_vocab_size
from ....model_pipelines.common.learning_rates import OCLRConfig
from ....model_pipelines.common.optimizer import AdamWConfig
from ....model_pipelines.common.train import TrainOptions, TrainedModel, train
from ....model_pipelines.qat_ctc.pytorch_modules import (
    QATConformerCTCConfig,
    QATConformerCTCModel,
    SpecaugmentByLengthConfig,
)

from ....model_pipelines.common.assemblies.conformer import ConformerEncoderQuantV1Config, ConformerBlockQuantV1Config, ConformerPositionwiseFeedForwardQuantV4Config, QuantizedConformerMHSARelPosV1Config, ConformerConvolutionQuantV4Config, WeightPruningConfig

from ....model_pipelines.qat_ctc.train import get_train_step_import


def run(
    descriptor: str,
    qat_args: dict,
    model_config: Optional[QATConformerCTCConfig] = None,
    train_options: Optional[TrainOptions] = None,
) -> TrainedModel[QATConformerCTCConfig]:
    if model_config is None:
        if qat_args is None:
            raise ValueError("Must specify either model_config or qat_args")
        model_config = get_model_config(**qat_args)
    if train_options is None:
        train_options = get_train_options()

    return train(
        descriptor=descriptor,
        model_class=QATConformerCTCModel,
        model_config=model_config,
        options=train_options,
        train_step_import=get_train_step_import(),
    )


def get_model_config(weight_bit_prec: Union[int, dict], activation_bit_prec: int, weight_dropout: float, weight_pruning_config: WeightPruningConfig, bpe_size: int = 128, layer_size: int = 512) -> QATConformerCTCConfig:
    
    if isinstance(weight_bit_prec, dict):
        ff_prec = weight_bit_prec["ff"]
        mhsa_prec = weight_bit_prec["mhsa"]
        conv_prec = weight_bit_prec["conv"]
    else:
        ff_prec = mhsa_prec = conv_prec = weight_bit_prec
    
    prior_train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    qat_args = dict(
        dropout=0.1,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        weight_quant_dtype=torch.qint8,
        weight_quant_method="per_tensor_symmetric",
        activation_quant_dtype=torch.qint8,
        activation_quant_method="per_tensor_symmetric",
        dot_quant_dtype=torch.qint8,
        dot_quant_method="per_tensor_symmetric",
        Av_quant_dtype=torch.qint8,
        Av_quant_method="per_tensor_symmetric",
        moving_average=None,
        quantize_output=False,
        converter_hardware_settings=prior_train_dac_settings,
        pos_enc_converter_hardware_settings=prior_train_dac_settings,
        quant_in_linear=True,
        num_cycles=0,
        correction_settings=None,
        weight_noise=None,
        weight_noise_func=None,
        weight_noise_values=None,
        weight_noise_start_epoch=None
    )
    
    return QATConformerCTCConfig(
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
        conformer_cfg=ConformerEncoderQuantV1Config(
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
            block_cfg=ConformerBlockQuantV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardQuantV4Config(
                    input_dim=512,
                    hidden_dim=2048,
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    weight_bit_prec=ff_prec,
                    activation_bit_prec=activation_bit_prec,
                    weight_quant_dtype=qat_args["weight_quant_dtype"],
                    weight_quant_method=qat_args["weight_quant_method"],
                    activation_quant_dtype=qat_args["activation_quant_dtype"],
                    activation_quant_method=qat_args["activation_quant_method"],
                    moving_average=qat_args["moving_average"],
                    converter_hardware_settings=qat_args["converter_hardware_settings"],
                    num_cycles=qat_args["num_cycles"],
                    weight_noise=qat_args["weight_noise"],
                    correction_settings=qat_args["correction_settings"],
                    weight_dropout=weight_dropout,
                    weight_pruning=weight_pruning_config,
                ),
                mhsa_cfg=QuantizedConformerMHSARelPosV1Config(
                    input_dim=512,
                    num_att_heads=8,
                    att_weights_dropout=qat_args["att_weights_dropout"],
                    dropout=0.1,
                    with_bias=True,
                    learnable_pos_emb=False,
                    rel_pos_clip=16,
                    with_linear_pos=True,
                    with_pos_bias=True,
                    separate_pos_emb_per_head=True,
                    pos_emb_dropout=0.0,
                    dropout_broadcast_axes=None,
                    bit_prec_W_i=mhsa_prec,
                    bit_prec_W_o=mhsa_prec,
                    bit_prec_learn_emb=mhsa_prec,
                    activation_bit_prec=activation_bit_prec,
                    quant_in_linear=qat_args["quant_in_linear"],
                    weight_quant_dtype=qat_args["weight_quant_dtype"],
                    weight_quant_method=qat_args["weight_quant_method"],
                    activation_quant_dtype=qat_args["activation_quant_dtype"],
                    activation_quant_method=qat_args["activation_quant_method"],
                    dot_quant_dtype=qat_args["dot_quant_dtype"],
                    dot_quant_method=qat_args["dot_quant_method"], 
                    Av_quant_dtype=qat_args["Av_quant_dtype"],
                    Av_quant_method=qat_args["Av_quant_method"],
                    moving_average=qat_args["moving_average"],
                    converter_hardware_settings=qat_args["converter_hardware_settings"],
                    pos_enc_converter_hardware_settings=qat_args["pos_enc_converter_hardware_settings"],
                    num_cycles=qat_args["num_cycles"],
                    correction_settings=qat_args["correction_settings"],
                    weight_noise=qat_args["weight_noise"],
                    weight_dropout=weight_dropout,
                    weight_pruning=weight_pruning_config
                ),
                conv_cfg=ConformerConvolutionQuantV4Config(
                    channels=512,
                    kernel_size=qat_args["conv_kernel_size"],
                    dropout=0.1,
                    activation=torch.nn.SiLU(),
                    norm=LayerNormNC(512),
                    weight_bit_prec=conv_prec,
                    activation_bit_prec=activation_bit_prec,
                    weight_quant_dtype=qat_args["weight_quant_dtype"],
                    weight_quant_method=qat_args["weight_quant_method"],
                    activation_quant_dtype=qat_args["activation_quant_dtype"],
                    activation_quant_method=qat_args["activation_quant_method"],
                    moving_average=qat_args["moving_average"],
                    converter_hardware_settings=qat_args["converter_hardware_settings"],
                    num_cycles=qat_args["num_cycles"],
                    correction_settings=qat_args["correction_settings"],
                    weight_noise=qat_args["weight_noise"],
                    weight_dropout=weight_dropout,
                    weight_pruning=weight_pruning_config
                ),
                modules=["ff", "conv", "mhsa", "ff"],
                scales=[0.5, 1.0, 1.0, 0.5],
            ),
        ),
        dim=layer_size,
        target_size=bpe_to_vocab_size(bpe_size=bpe_size) + 1,
        dropout=0.1,
    )


def get_train_options(bpe_size: int = 128, num_epochs: int = 100) -> TrainOptions:
    train_data_config = librispeech_datasets.get_default_bpe_train_data(bpe_size=bpe_size)
    cv_data_config = librispeech_datasets.get_default_bpe_cv_data(bpe_size=bpe_size)

    partition_epoch = train_data_config.partition_epoch

    save_epochs = list(range(num_epochs * 3 // 4, num_epochs - 5, 5)) + list(range(num_epochs - 5, num_epochs + 1))
    save_subepochs = [epoch * partition_epoch for epoch in save_epochs]

    return TrainOptions(
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=save_subepochs,
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
            inc_epochs=(num_epochs - 4) // 2 * partition_epoch,
            dec_epochs=(num_epochs - 4) // 2 * partition_epoch,
            final_epochs=4 * partition_epoch,
        ),
        gradient_clip=1.0,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=48,
        max_seqs=None,
        max_seq_length=None,
    )
