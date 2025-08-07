from typing import List

import torch
from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lexicon import get_bpe_bliss_lexicon
from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.lm import get_arpa_lm_config
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
from ...model_pipelines.aed.label_scorer_config import get_aed_label_scorer_config
from ...model_pipelines.aed.pytorch_modules import (
    AdditiveAttentionConfig,
    AEDConfig,
    AEDEncoder,
    AttentionLSTMDecoderV1Config,
)
from ...model_pipelines.aed.train import AEDTrainOptions, train
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.learning_rates import ConstConstDecayLRConfig
from ...model_pipelines.common.optimizer import RAdamConfig
from ...model_pipelines.common.pytorch_modules import SpecaugmentByLengthConfig
from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.recog_rasr import recog_rasr
from ...model_pipelines.common.recog_rasr_config import (
    LabelsyncGlobalPruningStrategy,
    get_lexiconfree_labelsync_recog_config,
    get_tree_labelsync_recog_config,
)
from ...model_pipelines.common.report import create_report
from ...model_pipelines.common.serializers import get_model_serializers

# from .bpe_lstm_lm_baseline import run_bpe_lstm_lm_baseline

BPE_SIZE = 5000


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
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        max_seqs=None,
        max_seq_length=None,
    )


def run_bpe_aed_baseline(prefix: str = "librispeech/bpe_aed") -> List[RecogResult]:
    with ExperimentContext(prefix):
        model_config = get_baseline_model_config()
        train_config = get_baseline_train_options()

        train_job = train(options=train_config, model_config=model_config)
        checkpoint: PtCheckpoint = train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore

        vocab_file = get_bpe_vocab_file(bpe_size=BPE_SIZE, add_blank=False)

        # lstm_lm_config, lstm_lm_checkpoint = run_bpe_lstm_lm_baseline()
        # lstm_lm_checkpoint = PtCheckpoint(
        #     tk.Path(
        #         "/work/asr4/rossenbach/sisyphus_work_folders/tts_decoder_asr_work/i6_core/returnn/training/ReturnnTrainingJob.EuWaxahLY8Ab/output/models/epoch.300.pt"
        #     )
        # )

        lexicon_file = get_bpe_bliss_lexicon(bpe_size=BPE_SIZE, add_blank=False)

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

        # for recog_corpus in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        for recog_corpus in ["dev-other"]:
            for max_beam_size in [8, 64, 1024]:
                for score_threshold in [0.01, 0.05]:
                    recog_results.append(
                        recog_rasr(
                            descriptor=f"bpe-aed_beam-{max_beam_size}_score-{score_threshold}",
                            recog_data_config=recog_data[recog_corpus],
                            recog_corpus=score_corpora[recog_corpus],
                            model_serializers=get_model_serializers(model_class=AEDEncoder, model_config=model_config),
                            rasr_config_file=get_lexiconfree_labelsync_recog_config(
                                vocab_file=vocab_file,
                                label_scorer_config=get_aed_label_scorer_config(
                                    model_config=model_config,
                                    checkpoint=checkpoint,
                                ),
                                sentence_end_index=0,
                                max_beam_size=max_beam_size,
                                score_threshold=score_threshold,
                                length_norm_scale=1.2,
                            ),
                            sample_rate=16000,
                            checkpoint=checkpoint,
                        )
                    )

        # =====================================
        # === Tree Search with 4gram LM =======
        # =====================================

        # for recog_corpus in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        for recog_corpus in ["dev-other"]:
            for max_beam_size in [8, 64, 1024]:
                for max_word_end_beam_size in [8]:
                    for global_max_beam_size in [
                        max_beam_size // 2,
                        max_beam_size,
                        max_beam_size + max_word_end_beam_size,
                    ]:
                        for score_threshold in [0.01, 0.02, 0.03]:
                            for word_end_score_threshold in [1.0, 10.0]:
                                for global_score_threshold in [1.0, 10.0]:
                                    for domination_score_threshold in [0.5, 1.0, 10.0]:
                                        for pruning_strategy in [
                                            LabelsyncGlobalPruningStrategy.ACTIVE_AGAINST_TERMINATED,
                                            # LabelsyncGlobalPruningStrategy.ALL,
                                        ]:
                                            for length_norm_scale in [1.0, 1.2]:
                                                pruning_str = (
                                                    "all"
                                                    if pruning_strategy == LabelsyncGlobalPruningStrategy.ALL
                                                    else "aat"
                                                )
                                                recog_results.append(
                                                    recog_rasr(
                                                        descriptor=f"bpe-aed_beam-{max_beam_size}"
                                                        f"_webeam-{max_word_end_beam_size}"
                                                        f"_gbeam-{global_max_beam_size}"
                                                        f"_score-{score_threshold}"
                                                        f"_wescore-{word_end_score_threshold}"
                                                        f"_gscore-{global_score_threshold}"
                                                        f"_domscore-{domination_score_threshold}"
                                                        f"_ln-{length_norm_scale}"
                                                        f"_prune-{pruning_str}",
                                                        recog_data_config=recog_data[recog_corpus],
                                                        recog_corpus=score_corpora[recog_corpus],
                                                        model_serializers=get_model_serializers(
                                                            model_class=AEDEncoder, model_config=model_config
                                                        ),
                                                        rasr_config_file=get_tree_labelsync_recog_config(
                                                            lexicon_file=lexicon_file,
                                                            label_scorer_config=get_aed_label_scorer_config(
                                                                model_config=model_config,
                                                                checkpoint=checkpoint,
                                                            ),
                                                            lm_config=arpa_lm_config,
                                                            max_beam_size=max_beam_size,
                                                            max_word_end_beam_size=max_word_end_beam_size,
                                                            global_max_beam_size=global_max_beam_size,
                                                            score_threshold=score_threshold,
                                                            word_end_score_threshold=word_end_score_threshold,
                                                            global_score_threshold=global_score_threshold,
                                                            global_pruning_strategy=pruning_strategy,
                                                            domination_score_threshold=domination_score_threshold,
                                                            length_norm_scale=length_norm_scale,
                                                            log_stepwise_statistics=False,
                                                        ),
                                                        sample_rate=16000,
                                                        checkpoint=checkpoint,
                                                    )
                                                )

        tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)
    return recog_results
