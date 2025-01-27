from typing import List, Literal, Optional

import torch
from i6_core.returnn import PtCheckpoint
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
from sisyphus import tk

from ...data.librispeech.bpe import bpe_to_vocab_size, get_bpe_vocab_file
from ...data.librispeech.datasets import (
    get_default_bpe_cv_data,
    get_default_bpe_train_data,
    get_default_recog_data,
    get_default_score_corpus,
)
from ...model_pipelines.bpe_aed.label_scorer_config import get_aed_label_scorer_config
from ...model_pipelines.bpe_aed.pytorch_modules import (
    AdditiveAttentionConfig,
    AEDConfig,
    AEDEncoder,
    AttentionLSTMDecoderV1Config,
)
from ...model_pipelines.bpe_aed.train import AEDTrainOptions, train
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.imports import get_model_serializers
from ...model_pipelines.common.learning_rates import ConstConstDecayLRConfig
from ...model_pipelines.common.optimizer import RAdamConfig
from ...model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig
from ...model_pipelines.common.recog import RecogResult, recog_rasr
from ...model_pipelines.common.recog_rasr_config import RasrRecogOptions, get_rasr_config_file
from ...model_pipelines.common.report import create_report

BPE_SIZE = 128


def get_baseline_model_config() -> AEDConfig:
    vocab_size = bpe_to_vocab_size(BPE_SIZE)
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
            start_epoch=41,
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


def get_baseline_train_options() -> AEDTrainOptions:
    train_data_config = get_default_bpe_train_data(BPE_SIZE)
    assert train_data_config.target_config
    train_data_config.target_config["seq_postfix"] = [0]

    cv_data_config = get_default_bpe_cv_data(BPE_SIZE)
    assert cv_data_config.target_config
    cv_data_config.target_config["seq_postfix"] = [0]

    return AEDTrainOptions(
        descriptor="baseline",
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=list(range(1500, 1900, 100)) + list(range(1900, 2001, 20)),
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
            const_epochs_1=40,
            const_epochs_2=920,
            dec_epochs=960,
            final_epochs=80,
        ),
        gradient_clip=1.0,
        ctc_loss_scale=0.7,
        label_smoothing=0.1,
        label_smoothing_start_epoch=61,
        num_workers_per_gpu=2,
        stop_on_inf_nan_score=True,
    )


def get_baseline_recog_options() -> RasrRecogOptions:
    return RasrRecogOptions(
        blank_index=None,
        sentence_end_index=0,
        vocab_file=get_bpe_vocab_file(bpe_size=BPE_SIZE, add_blank=False),
        max_beam_size=8,
        top_k_tokens=8,
        score_threshold=12.0,
        allow_label_loop=False,
    )


def run_recog(
    descriptor: str,
    corpus_name: str,
    checkpoint: PtCheckpoint,
    model_config: AEDConfig,
    recog_options: Optional[RasrRecogOptions] = None,
    device: Literal["cpu", "gpu"] = "cpu",
) -> RecogResult:
    recog_options = recog_options or get_baseline_recog_options()

    aed_scorer_config = get_aed_label_scorer_config(model_config=model_config, checkpoint=checkpoint)

    rasr_config_file = get_rasr_config_file(
        recog_options=recog_options,
        label_scorer_config=aed_scorer_config,
    )

    return recog_rasr(
        descriptor=descriptor,
        recog_data_config=get_default_recog_data(corpus_name=corpus_name),
        recog_corpus=get_default_score_corpus(corpus_name=corpus_name),
        model_serializers=get_model_serializers(model_class=AEDEncoder, model_config=model_config),
        rasr_config_file=rasr_config_file,
        sample_rate=16000,
        device=device,
        checkpoint=checkpoint,
    )


def run_bpe_aed_baseline(prefix: str = "librispeech/bpe_aed") -> List[RecogResult]:
    with ExperimentContext(prefix):
        model_config = get_baseline_model_config()
        train_config = get_baseline_train_options()

        train_job = train(options=train_config, model_config=model_config)
        checkpoint: PtCheckpoint = train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore

        recog_results = []
        for corpus_name in ["dev-clean", "dev-other", "test-clean", "test-other"]:
            recog_results.append(
                run_recog(
                    descriptor="bpe-aed_recog-rasr",
                    corpus_name=corpus_name,
                    model_config=model_config,
                    checkpoint=checkpoint,
                )
            )

        tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)
    return recog_results
