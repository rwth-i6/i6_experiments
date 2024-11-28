from typing import List, Literal

import torch
from i6_models.assemblies.conformer.conformer_v2 import ConformerBlockV2Config, ConformerEncoderV2Config
from i6_models.config import ModuleFactoryV1
from i6_models.parts.conformer import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.generic_frontend import FrontendLayerType, GenericFrontendV1, GenericFrontendV1Config
from i6_models.primitives.feature_extraction import RasrCompatibleLogMelFeatureExtractionV1Config

from ..data.bpe import DataConfig, get_bpe_vocab_file
from .configs import (
    CTCFlashlightRecogRoutineConfig,
    CTCPriorRoutineConfig,
    CTCRasrGreedyRecogRoutineConfig,
    RasrGreedyRecogRoutineConfig,
    OCLRConfig,
    PipelineConfig,
    RasrBeamRecogRoutineConfig,
    TrainRoutineConfig,
)
from .pytorch_modules import FFNNTransducerConfig, SpecaugmentByLengthConfig

default_train_data = DataConfig(
    dataset_type="train",
    corpus_names=["train-other-960"],
    bpe_size=128,
    speed_perturbation=True,
    ogg_segments=200,
    partition_epoch=10,
    seq_ordering="laplace:.1000",
    preemphasis=0.97,
)

default_cv_data = DataConfig(
    dataset_type="dev",
    corpus_names=["dev-clean", "dev-other"],
    bpe_size=128,
    speed_perturbation=False,
    ogg_segments=1,
    partition_epoch=1,
    seq_ordering="sorted",
    preemphasis=0.97,
)

default_prior_data = DataConfig(
    dataset_type="forward_data",
    corpus_names=["train-other-960"],
    bpe_size=None,
    speed_perturbation=False,
    ogg_segments=200,
    partition_epoch=10,
    seq_ordering="sorted",
    preemphasis=0.97,
)


def default_recog_data(corpus_name: Literal["dev-clean", "dev-other", "test-clean", "test-other"]) -> DataConfig:
    return DataConfig(
        dataset_type="forward_data",
        corpus_names=[corpus_name],
        bpe_size=None,
        speed_perturbation=False,
        ogg_segments=1,
        partition_epoch=1,
        seq_ordering="sorted",
        preemphasis=0.97,
    )


default_model_config = FFNNTransducerConfig(
    logmel_cfg=RasrCompatibleLogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        min_amp=1.175494e-38,
        num_filters=80,
        alpha=0.0,
    ),
    specaug_cfg=SpecaugmentByLengthConfig(
        start_epoch=11,
        time_min_num_masks=2,
        time_max_mask_per_n_frames=25,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=5,
        freq_mask_max_size=16,
    ),
    conformer_cfg=ConformerEncoderV2Config(
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
        block_cfg=ConformerBlockV2Config(
            ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                input_dim=512,
                hidden_dim=2048,
                dropout=0.1,
                activation=torch.nn.SiLU(),
            ),
            mhsa_cfg=ConformerMHSAV1Config(
                input_dim=512,
                num_att_heads=8,
                att_weights_dropout=0.1,
                dropout=0.1,
            ),
            conv_cfg=ConformerConvolutionV1Config(
                channels=512,
                kernel_size=31,
                dropout=0.1,
                activation=torch.nn.SiLU(),
                norm=LayerNormNC(512),
            ),
            modules=["ff", "conv", "mhsa", "ff"],
            scales=[0.5, 1.0, 1.0, 0.5],
        ),
    ),
    dropout=0.1,
    enc_dim=512,
    enc_output_indices=[5, 11],
    pred_num_layers=2,
    pred_dim=640,
    pred_activation=torch.nn.Tanh(),
    context_history_size=1,
    context_embedding_dim=256,
    joiner_dim=1024,
    joiner_activation=torch.nn.Tanh(),
    target_size=185,
)

default_train_routine_config = TrainRoutineConfig(
    train_data_config=default_train_data,
    cv_data_config=default_cv_data,
    save_epochs=[10] + list(range(100, 901, 100)) + list(range(900, 1001, 20)),
    batch_frames=12_000,
    accum_grad_multiple_step=2,
    lr_config=OCLRConfig(
        init_lr=7e-06,
        peak_lr=5e-04,
        decayed_lr=5e-05,
        final_lr=1e-07,
        inc_epochs=480,
        dec_epochs=480,
        final_epochs=40,
    ),
    enc_loss_scales={5: 0.3, 11: 0.7},
    gradient_clip=1.0,
    weight_decay=0.001,
)

default_ctc_prior_routine_config = CTCPriorRoutineConfig(
    enc_layer=11,
    prior_data_config=default_prior_data,
    batch_frames=20_000,
)


def default_ctc_rasr_greedy_recog_routine_configs(
    descriptor: str,
    corpus_names: List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]],
    epoch: int,
) -> List[CTCRasrGreedyRecogRoutineConfig]:
    return [
        CTCRasrGreedyRecogRoutineConfig(
            enc_layer=11,
            descriptor=descriptor,
            corpus_name=corpus_name,
            recog_data_config=default_recog_data(corpus_name),
            epoch=epoch,
            prior_config=default_ctc_prior_routine_config,
            vocab_file=get_bpe_vocab_file(bpe_size=128),
            device="cpu",
            prior_scale=0.0,
            blank_penalty=0.0,
        )
        for corpus_name in corpus_names
    ]


def default_ctc_flashlight_greedy_recog_routine_configs(
    descriptor: str,
    corpus_names: List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]],
    epoch: int,
) -> List[CTCFlashlightRecogRoutineConfig]:
    return [
        CTCFlashlightRecogRoutineConfig(
            enc_layer=11,
            descriptor=descriptor,
            corpus_name=corpus_name,
            recog_data_config=default_recog_data(corpus_name),
            epoch=epoch,
            prior_config=default_ctc_prior_routine_config,
            vocab_file=get_bpe_vocab_file(bpe_size=128),
            lexicon_file=None,
            lm_file=None,
            beam_size=1,
            beam_size_token=None,
            beam_threshold=float("inf"),
            lm_scale=0.0,
            device="cpu",
            prior_scale=0.0,
            blank_penalty=0.0,
        )
        for corpus_name in corpus_names
    ]


def default_rasr_greedy_recog_routine_configs(
    descriptor: str,
    corpus_names: List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]],
    epoch: int,
) -> List[RasrGreedyRecogRoutineConfig]:
    return [
        RasrGreedyRecogRoutineConfig(
            descriptor=descriptor,
            corpus_name=corpus_name,
            recog_data_config=default_recog_data(corpus_name),
            epoch=epoch,
            vocab_file=get_bpe_vocab_file(bpe_size=128),
            device="cpu",
            ilm_scale=0.2,
            blank_penalty=1.0,
        )
        for corpus_name in corpus_names
    ]


def default_rasr_beam_recog_routine_configs(
    descriptor: str,
    corpus_names: List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]],
    epoch: int,
    max_beam_size: int = 8,
) -> List[RasrBeamRecogRoutineConfig]:
    return [
        RasrBeamRecogRoutineConfig(
            descriptor=descriptor,
            corpus_name=corpus_name,
            recog_data_config=default_recog_data(corpus_name),
            epoch=epoch,
            vocab_file=get_bpe_vocab_file(bpe_size=128),
            device="cpu",
            max_beam_size=max_beam_size,
            top_k_tokens=None,
            score_threshold=None,
            ilm_scale=0.2,
            blank_penalty=1.0,
        )
        for corpus_name in corpus_names
    ]


default_pipeline_config = PipelineConfig(
    model_config=default_model_config,
    train_config=default_train_routine_config,
    recog_configs=[
        default_rasr_greedy_recog_routine_configs(
            descriptor=f"intermediate_e-{epoch}",
            corpus_names=["dev-other"],
            epoch=epoch,
        )[0]
        for epoch in default_train_routine_config.save_epochs
    ]
    + default_rasr_greedy_recog_routine_configs(
        descriptor="rasr_greedy", corpus_names=["dev-clean", "dev-other"], epoch=1000
    )
    + default_ctc_rasr_greedy_recog_routine_configs(
        descriptor="ctc_encoder_rasr_greedy", corpus_names=["dev-clean", "dev-other"], epoch=1000
    )
    + default_ctc_flashlight_greedy_recog_routine_configs(
        descriptor="ctc_encoder_flashlight_greedy", corpus_names=["dev-clean", "dev-other"], epoch=1000
    )
    + [
        default_rasr_beam_recog_routine_configs(
            descriptor=f"rasr_beam-{max_beam_size}",
            corpus_names=["dev-other"],
            epoch=1000,
            max_beam_size=max_beam_size,
        )[0]
        for max_beam_size in [2, 4, 8, 16, 32, 64]
    ],
)
