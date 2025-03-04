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

from ...data.tedlium2.bpe import (
    bpe_to_vocab_size,
    get_bpe_vocab_file,
)
from ...data.tedlium2.datasets import (
    get_default_bpe_cv_data,
    get_default_bpe_train_data,
    get_default_prior_data,
    get_default_recog_data,
    get_default_score_corpus,
)
from ...model_pipelines.bpe_aed.pytorch_modules import AdditiveAttentionConfig, AttentionLSTMDecoderV1Config
from ...model_pipelines.bpe_combination_model.label_scorer_config import (
    get_attention_label_scorer_config,
    get_ctc_label_scorer_config,
    get_ctc_prefix_label_scorer_config,
    get_transducer_label_scorer_config,
)
from ...model_pipelines.bpe_combination_model.prior import compute_priors
from ...model_pipelines.bpe_combination_model.pytorch_modules import CombinationModelConfig, CombinationModelEncoder
from ...model_pipelines.bpe_combination_model.train import CombinationTrainOptions, train
from ...model_pipelines.bpe_lstm_lm.label_scorer_config import get_lstm_lm_label_scorer_config
from ...model_pipelines.bpe_lstm_lm.pytorch_modules import LstmLmConfig
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.imports import get_model_serializers
from ...model_pipelines.common.learning_rates import ConstConstDecayLRConfig
from ...model_pipelines.common.optimizer import RAdamConfig
from ...model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig
from ...model_pipelines.common.recog import RecogResult, recog_rasr
from ...model_pipelines.common.recog_rasr_config import RasrRecogOptions, get_rasr_config_file
from ...model_pipelines.common.report import create_report
from .bpe_lstm_lm_baseline import run_bpe_lstm_lm_baseline

BPE_SIZE = 128


def get_baseline_model_config() -> CombinationModelConfig:
    vocab_size = bpe_to_vocab_size(BPE_SIZE)
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
            start_epoch=9,
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
        enc_dim=512,
        ctc_dropout=0.1,
        target_size=vocab_size + 1,
    )


def get_baseline_train_options() -> CombinationTrainOptions:
    train_data_config = get_default_bpe_train_data(BPE_SIZE)
    assert train_data_config.target_config
    train_data_config.target_config["seq_postfix"] = [0]

    cv_data_config = get_default_bpe_cv_data(BPE_SIZE)
    assert cv_data_config.target_config
    cv_data_config.target_config["seq_postfix"] = [0]

    return CombinationTrainOptions(
        descriptor="baseline",
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=list(range(300, 380, 20)) + list(range(380, 401, 4)),
        batch_size=12_000 * 160,
        accum_grad_multiple_step=2,
        optimizer_config=RAdamConfig(
            epsilon=1e-12,
            weight_decay=0.001,
            decoupled_weight_decay=True,
        ),
        lr_config=ConstConstDecayLRConfig(
            const_lr_1=3e-05,
            const_lr_2=3e-04,
            decayed_lr=3e-05,
            final_lr=1e-07,
            const_epochs_1=8,
            const_epochs_2=184,
            dec_epochs=192,
            final_epochs=16,
        ),
        gradient_clip=1.0,
        ctc_loss_scale=0.7,
        transducer_loss_scale=1.0,
        attention_loss_scale=1.0,
        attention_label_smoothing=0.1,
        attention_label_smoothing_start_epoch=13,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
    )


def get_baseline_recog_options(blank: bool, sentence_end: bool, label_loop: bool) -> RasrRecogOptions:
    return RasrRecogOptions(
        blank_index=bpe_to_vocab_size(bpe_size=BPE_SIZE) if blank else None,
        sentence_end_index=0 if sentence_end else None,
        length_norm_scale=0.5 if sentence_end else None,
        vocab_file=get_bpe_vocab_file(bpe_size=BPE_SIZE, add_blank=blank),
        max_beam_size_per_scorer=64,
        max_beam_size=16,
        score_threshold=12.0,
        allow_label_loop=label_loop,
    )


def run_recog(
    descriptor: str,
    corpus_name: str,
    checkpoint: PtCheckpoint,
    model_config: CombinationModelConfig,
    ctc_score_scale: float = 0.0,
    ctc_prefix_score_scale: float = 0.0,
    transducer_score_scale: float = 0.0,
    attention_score_scale: float = 0.0,
    ctc_blank_penalty: float = 0.0,
    ctc_prior_scale: float = 0.0,
    transducer_ilm_scale: float = 0.2,
    transducer_blank_penalty: float = 0.0,
    lm_checkpoint: Optional[PtCheckpoint] = None,
    lm_config: Optional[LstmLmConfig] = None,
    lm_scale: float = 0.0,
    max_beam_size: Optional[int] = None,
    score_threshold: Optional[float] = None,
    device: Literal["cpu", "gpu"] = "cpu",
) -> RecogResult:
    recog_options = get_baseline_recog_options(
        blank=(ctc_score_scale != 0 or transducer_score_scale != 0),
        sentence_end=attention_score_scale != 0,
        label_loop=ctc_score_scale != 0 and transducer_score_scale == 0 and attention_score_scale == 0,
    )
    if max_beam_size is not None:
        recog_options.max_beam_size = max_beam_size
    if score_threshold is not None:
        recog_options.score_threshold = score_threshold

    prior_file = compute_priors(
        prior_data_config=get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    label_scorer_configs = []

    if ctc_score_scale != 0:
        label_scorer_configs.append(
            get_ctc_label_scorer_config(
                model_config=model_config,
                checkpoint=checkpoint,
                prior_file=prior_file,
                prior_scale=ctc_prior_scale,
                blank_penalty=ctc_blank_penalty,
                scale=ctc_score_scale,
            )
        )

    if transducer_score_scale != 0:
        label_scorer_configs.append(
            get_transducer_label_scorer_config(
                model_config=model_config,
                checkpoint=checkpoint,
                ilm_scale=transducer_ilm_scale,
                blank_penalty=transducer_blank_penalty,
                scale=transducer_score_scale,
            )
        )

    if attention_score_scale != 0:
        label_scorer_configs.append(
            get_attention_label_scorer_config(
                model_config=model_config,
                checkpoint=checkpoint,
                scale=attention_score_scale,
            )
        )

    if ctc_prefix_score_scale != 0:
        label_scorer_configs.append(
            get_ctc_prefix_label_scorer_config(
                model_config=model_config,
                checkpoint=checkpoint,
                prior_file=prior_file,
                prior_scale=ctc_prior_scale,
                scale=ctc_prefix_score_scale,
            )
        )

    if lm_scale != 0:
        assert lm_config is not None
        assert lm_checkpoint is not None
        label_scorer_configs.append(
            get_lstm_lm_label_scorer_config(
                model_config=lm_config,
                checkpoint=lm_checkpoint,
                scale=lm_scale,
            )
        )

    rasr_config_file = get_rasr_config_file(
        recog_options=recog_options,
        label_scorer_config=label_scorer_configs,
    )

    return recog_rasr(
        descriptor=descriptor,
        recog_data_config=get_default_recog_data(corpus_name=corpus_name),
        recog_corpus=get_default_score_corpus(corpus_name=corpus_name),
        model_serializers=get_model_serializers(model_class=CombinationModelEncoder, model_config=model_config),
        rasr_config_file=rasr_config_file,
        sample_rate=16000,
        device=device,
        checkpoint=checkpoint,
    )


def run_bpe_combination_baseline(prefix: str = "tedlium2/bpe_combination") -> List[RecogResult]:
    with ExperimentContext(prefix):
        model_config = get_baseline_model_config()
        train_config = get_baseline_train_options()

        train_job = train(options=train_config, model_config=model_config)
        checkpoint: PtCheckpoint = train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore
        lstm_lm_config, lstm_lm_checkpoint = run_bpe_lstm_lm_baseline()

        recog_results = []
        for corpus_name in ["dev"]:
            recog_results.append(
                run_recog(
                    descriptor="bpe-combination_recog-rasr_ctc-only",
                    corpus_name=corpus_name,
                    model_config=model_config,
                    checkpoint=checkpoint,
                    ctc_score_scale=1.0,
                    max_beam_size=1,
                    score_threshold=0.0,
                )
            )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc+lm",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_score_scale=1.0,
            #         lm_config=lstm_lm_config,
            #         lm_checkpoint=lstm_lm_checkpoint,
            #         lm_scale=0.8,
            #     )
            # )

            recog_results.append(
                run_recog(
                    descriptor="bpe-combination_recog-rasr_transducer-only",
                    corpus_name=corpus_name,
                    model_config=model_config,
                    checkpoint=checkpoint,
                    ctc_score_scale=0.0,
                    transducer_score_scale=1.0,
                )
            )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_transducer+lm",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         transducer_score_scale=1.0,
            #         transducer_blank_penalty=blank_penalty,
            #         lm_config=lstm_lm_config,
            #         lm_checkpoint=lstm_lm_checkpoint,
            #         lm_scale=0.6,
            #     )
            # )

            recog_results.append(
                run_recog(
                    descriptor="bpe-combination_recog-rasr_attention-only",
                    corpus_name=corpus_name,
                    model_config=model_config,
                    checkpoint=checkpoint,
                    attention_score_scale=1.0,
                )
            )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_attention+lm",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         attention_score_scale=1.0,
            #         lm_config=lstm_lm_config,
            #         lm_checkpoint=lstm_lm_checkpoint,
            #         lm_scale=0.4,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc+attention",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_score_scale=0.3,
            #         ctc_blank_penalty=1.0,
            #         attention_score_scale=0.7,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc-prefix+attention",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_prefix_score_scale=0.3,
            #         attention_score_scale=0.7,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc+attention+lm",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_score_scale=0.3,
            #         attention_score_scale=0.7,
            #         lm_config=lstm_lm_config,
            #         lm_checkpoint=lstm_lm_checkpoint,
            #         lm_scale=0.6,
            #         ctc_blank_penalty=1.0,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc-prefix+transducer",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_prefix_score_scale=0.3,
            #         transducer_score_scale=0.7,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_transducer+attention",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         transducer_score_scale=0.5,
            #         attention_score_scale=0.5,
            #         transducer_blank_penalty=2.0,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_transducer+attention+lm",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         transducer_score_scale=0.5,
            #         attention_score_scale=0.5,
            #         lm_config=lstm_lm_config,
            #         lm_checkpoint=lstm_lm_checkpoint,
            #         lm_scale=0.6,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc+transducer+attention",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_score_scale=0.2,
            #         transducer_score_scale=0.4,
            #         attention_score_scale=0.4,
            #         ctc_blank_penalty=1.0,
            #     )
            # )

            # recog_results.append(
            #     run_recog(
            #         descriptor=f"bpe-combination_recog-rasr_ctc+transducer+attention+lm",
            #         corpus_name=corpus_name,
            #         model_config=model_config,
            #         checkpoint=checkpoint,
            #         ctc_score_scale=0.2,
            #         ctc_blank_penalty=1.0,
            #         transducer_score_scale=0.4,
            #         attention_score_scale=0.4,
            #         lm_config=lstm_lm_config,
            #         lm_checkpoint=lstm_lm_checkpoint,
            #         lm_scale=0.5,
            #     )
            # )

        tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)
    return recog_results
