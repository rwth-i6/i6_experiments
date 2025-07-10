from typing import List

import torch
from i6_core.returnn import PtCheckpoint
from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab
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
from ...data.librispeech.lexicon import get_bpe_bliss_lexicon
from ...data.librispeech.lm import get_arpa_lm_config, get_transformer_lm_config
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.learning_rates import OCLRConfig
from ...model_pipelines.common.optimizer import RAdamConfig
from ...model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig
from ...model_pipelines.common.recog_rasr import RecogResult, recog_rasr
from ...model_pipelines.common.recog_rasr_config import (
    get_lexiconfree_timesync_recog_config,
    get_tree_timesync_recog_config,
)
from ...model_pipelines.common.report import create_report
from ...model_pipelines.common.serializers import get_model_serializers
from ...model_pipelines.ffnn_transducer.label_scorer_config import get_ffnn_transducer_label_scorer_config
from ...model_pipelines.ffnn_transducer.pytorch_modules import FFNNTransducerConfig, FFNNTransducerEncoder
from ...model_pipelines.ffnn_transducer.train import FFNNTransducerTrainOptions, train

BPE_SIZE = 128


def get_baseline_model_config() -> FFNNTransducerConfig:
    vocab_size = bpe_to_vocab_size(BPE_SIZE)
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
        dropout=0.1,
        enc_dim=512,
        pred_num_layers=2,
        pred_dim=640,
        pred_activation=torch.nn.Tanh(),
        context_history_size=1,
        context_embedding_dim=256,
        joiner_dim=1024,
        joiner_activation=torch.nn.Tanh(),
        target_size=vocab_size + 1,
    )


def get_baseline_train_options() -> FFNNTransducerTrainOptions:
    train_data_config = get_default_bpe_train_data(BPE_SIZE)
    assert train_data_config.target_config
    train_data_config.target_config["seq_postfix"] = [0]

    cv_data_config = get_default_bpe_cv_data(BPE_SIZE)
    assert cv_data_config.target_config
    cv_data_config.target_config["seq_postfix"] = [0]

    return FFNNTransducerTrainOptions(
        descriptor="baseline",
        train_data_config=get_default_bpe_train_data(BPE_SIZE),
        cv_data_config=get_default_bpe_cv_data(BPE_SIZE),
        save_epochs=list(range(1500, 1900, 100)) + list(range(1900, 2001, 20)),
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
            inc_epochs=960,
            dec_epochs=960,
            final_epochs=80,
        ),
        gradient_clip=1.0,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        enc_loss_scale=0.5,
        pred_loss_scale=0.2,
        max_seqs=None,
        max_seq_length=None,
    )


def run_bpe_ffnn_transducer_baseline(prefix: str = "librispeech/bpe_ffnn_transducer") -> List[RecogResult]:
    for enc_loss_scale in [0.0, 0.5]:
        for pred_loss_scale in [0.0, 0.2]:
            with ExperimentContext(f"{prefix}_enc-{enc_loss_scale}_pred-{pred_loss_scale}"):
                model_config = get_baseline_model_config()
                train_config = get_baseline_train_options()
                train_config.enc_loss_scale = enc_loss_scale
                train_config.pred_loss_scale = pred_loss_scale

                train_job = train(options=train_config, model_config=model_config)
                checkpoint: PtCheckpoint = train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore

                label_scorer_config = get_ffnn_transducer_label_scorer_config(
                    model_config=model_config,
                    checkpoint=checkpoint,
                    ilm_scale=0.0,
                    blank_penalty=0.0,
                )

                vocab_file = get_bpe_vocab_file(bpe_size=BPE_SIZE, add_blank=True)
                lm_vocab_file = get_lm_vocab(output_prefix="").vocab
                lexicon_file = get_bpe_bliss_lexicon(bpe_size=BPE_SIZE, add_blank=True)
                blank_index = bpe_to_vocab_size(bpe_size=BPE_SIZE)

                arpa_lm_config = get_arpa_lm_config(lm_name="4gram", lexicon_file=lexicon_file)
                arpa_lm_config.scale = 0.6

                recog_data = {
                    corpus_name: get_default_recog_data(corpus_name)
                    for corpus_name in ["dev-clean", "dev-other", "test-clean", "test-other"]
                }

                score_corpora = {
                    corpus_name: get_default_score_corpus(corpus_name)
                    for corpus_name in ["dev-clean", "dev-other", "test-clean", "test-other"]
                }

                recog_results = []

                # =====================================
                # === Lexiconfree Search without LM ===
                # =====================================

                for recog_corpus in ["dev-clean", "dev-other", "test-clean", "test-other"]:
                    recog_results.append(
                        recog_rasr(
                            descriptor="bpe-transducer_lexiconfree",
                            checkpoint=checkpoint,
                            recog_data_config=recog_data[recog_corpus],
                            recog_corpus=score_corpora[recog_corpus],
                            model_serializers=get_model_serializers(FFNNTransducerEncoder, model_config=model_config),
                            rasr_config_file=get_lexiconfree_timesync_recog_config(
                                vocab_file=vocab_file,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                blank_index=blank_index,
                                max_beam_size=64,
                                score_threshold=12.0,
                            ),
                            rasr_align_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon_file,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                blank_index=blank_index,
                                max_beam_size=256,
                                score_threshold=22.0,
                            ),
                            sample_rate=16000,
                            device="cpu",
                        )
                    )

                # =====================================
                # === Tree Search without LM ==========
                # =====================================

                for recog_corpus in ["dev-clean", "dev-other", "test-clean", "test-other"]:
                    recog_results.append(
                        recog_rasr(
                            descriptor="bpe-transducer_tree",
                            checkpoint=checkpoint,
                            recog_data_config=recog_data[recog_corpus],
                            recog_corpus=score_corpora[recog_corpus],
                            model_serializers=get_model_serializers(FFNNTransducerEncoder, model_config=model_config),
                            rasr_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon_file,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                blank_index=blank_index,
                                max_beam_size=64,
                                score_threshold=12.0,
                            ),
                            rasr_align_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon_file,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                blank_index=blank_index,
                                max_beam_size=256,
                                score_threshold=22.0,
                            ),
                            sample_rate=16000,
                            device="cpu",
                        )
                    )

                # =====================================
                # === Tree Search with 4gram LM =======
                # =====================================

                for recog_corpus in ["dev-clean", "dev-other", "test-clean", "test-other"]:
                    arpa_lm_config.scale = 0.6
                    recog_results.append(
                        recog_rasr(
                            descriptor="bpe-transducer_tree_4gram",
                            checkpoint=checkpoint,
                            recog_data_config=recog_data[recog_corpus],
                            recog_corpus=score_corpora[recog_corpus],
                            model_serializers=get_model_serializers(FFNNTransducerEncoder, model_config=model_config),
                            rasr_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon_file,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                lm_config=arpa_lm_config,
                                blank_index=blank_index,
                                max_beam_size=64,
                                score_threshold=12.0,
                            ),
                            rasr_align_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon_file,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                lm_config=arpa_lm_config,
                                blank_index=blank_index,
                                max_beam_size=256,
                                score_threshold=22.0,
                            ),
                            sample_rate=16000,
                            device="cpu",
                        )
                    )

                # =====================================
                # === Tree Search with TraFo LM =======
                # =====================================

                for trafo_layers in [24, 48, 96]:
                    trafo_lm_config = get_transformer_lm_config(
                        num_layers=trafo_layers, vocab_file=lm_vocab_file, lm_scale=0.6
                    )
                    for recog_corpus in ["dev-clean", "dev-other", "test-clean", "test-other"]:
                        recog_results.append(
                            recog_rasr(
                                descriptor=f"bpe-transducer_tree_trafo-{trafo_layers}l",
                                checkpoint=checkpoint,
                                recog_data_config=recog_data[recog_corpus],
                                recog_corpus=score_corpora[recog_corpus],
                                model_serializers=get_model_serializers(
                                    FFNNTransducerEncoder, model_config=model_config
                                ),
                                rasr_config_file=get_tree_timesync_recog_config(
                                    lexicon_file=lexicon_file,
                                    collapse_repeated_labels=False,
                                    label_scorer_config=label_scorer_config,
                                    lm_config=trafo_lm_config,
                                    blank_index=blank_index,
                                    max_beam_size=64,
                                    score_threshold=12.0,
                                ),
                                rasr_align_config_file=get_tree_timesync_recog_config(
                                    lexicon_file=lexicon_file,
                                    collapse_repeated_labels=False,
                                    label_scorer_config=label_scorer_config,
                                    lm_config=trafo_lm_config,
                                    blank_index=blank_index,
                                    max_beam_size=256,
                                    score_threshold=22.0,
                                ),
                                sample_rate=16000,
                                device="cpu",
                            )
                        )

        tk.register_report(
            f"{prefix}_enc-{enc_loss_scale}_pred-{pred_loss_scale}/report.txt",
            values=create_report(recog_results),
            required=True,
        )

    return recog_results
