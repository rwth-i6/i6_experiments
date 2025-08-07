from typing import List

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

from ...data.switchboard.bpe import bpe_to_vocab_size, get_bpe_vocab_file
from ...data.switchboard.datasets import (
    get_default_bpe_phoneme_cv_data,
    get_default_bpe_phoneme_train_data,
    get_default_prior_data,
    get_default_recog_data,
    get_default_score_corpus,
)
from ...data.switchboard.lexicon import get_bliss_phoneme_lexicon, get_bpe_bliss_lexicon
from ...data.switchboard.lm import get_arpa_lm_config
from ...data.switchboard.phoneme import PHONEME_SIZE
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.learning_rates import OCLRConfig
from ...model_pipelines.common.optimizer import AdamWConfig
from ...model_pipelines.common.recog_rasr import RecogResult, recog_rasr
from ...model_pipelines.common.recog_rasr_config import (
    get_lexiconfree_timesync_recog_config,
    get_tree_timesync_recog_config,
)
from ...model_pipelines.common.report import create_report
from ...model_pipelines.common.serializers import get_model_serializers
from ...model_pipelines.ctc_multi_output.label_scorer_config import get_ctc_label_scorer_config
from ...model_pipelines.ctc_multi_output.prior import compute_priors
from ...model_pipelines.ctc_multi_output.pytorch_modules import (
    ConformerCTCMultiOutputConfig,
    ConformerCTCMultiOutputEncoderModel,
    SpecaugmentByLengthConfig,
)
from ...model_pipelines.ctc_multi_output.train import CTCMultiOutputTrainOptions, train

BPE_SIZE = 128


def get_baseline_model_config() -> ConformerCTCMultiOutputConfig:
    vocab_size = bpe_to_vocab_size(BPE_SIZE)
    return ConformerCTCMultiOutputConfig(
        logmel_cfg=LogMelFeatureExtractionV1Config(
            sample_rate=8000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=3800,
            min_amp=1e-10,
            num_filters=60,
            center=False,
            n_fft=200,
        ),
        specaug_cfg=SpecaugmentByLengthConfig(
            start_epoch=7,
            time_min_num_masks=2,
            time_max_mask_per_n_frames=25,
            time_mask_max_size=20,
            freq_min_num_masks=2,
            freq_max_num_masks=5,
            freq_mask_max_size=12,
        ),
        conformer_cfg=ConformerRelPosEncoderV1Config(
            num_layers=12,
            frontend=ModuleFactoryV1(
                GenericFrontendV1,
                GenericFrontendV1Config(
                    in_features=60,
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
        layer_idx_target_size_list=[(8, PHONEME_SIZE + 1), (11, vocab_size + 1)],
        dropout=0.1,
    )


def get_baseline_train_options() -> CTCMultiOutputTrainOptions:
    return CTCMultiOutputTrainOptions(
        descriptor="baseline",
        train_data_config=get_default_bpe_phoneme_train_data(BPE_SIZE),
        cv_data_config=get_default_bpe_phoneme_cv_data(BPE_SIZE),
        save_epochs=list(range(450, 570, 30)) + list(range(570, 601, 6)),
        batch_size=24_000 * 80,
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
            inc_epochs=288,
            dec_epochs=288,
            final_epochs=24,
        ),
        gradient_clip=1.0,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        max_seqs=None,
        max_seq_length=None,
        target_names=["phoneme", "bpe"],
    )


def run_bpe_phoneme_ctc_baseline(prefix: str = "switchboard/bpe_phoneme_ctc") -> List[RecogResult]:
    model_config = get_baseline_model_config()

    with ExperimentContext(prefix):
        train_config = get_baseline_train_options()

        train_job = train(options=train_config, model_config=model_config)
        checkpoint: PtCheckpoint = train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore

        recog_data = {corpus_name: get_default_recog_data(corpus_name) for corpus_name in ["hub5e00", "hub5e01"]}
        score_corpora = {corpus_name: get_default_score_corpus(corpus_name) for corpus_name in ["hub5e00", "hub5e01"]}

        bpe_vocab_file = get_bpe_vocab_file(bpe_size=BPE_SIZE, add_blank=True)

        lexica = {
            "bpe": get_bpe_bliss_lexicon(bpe_size=BPE_SIZE, add_blank=True),
            "phoneme": get_bliss_phoneme_lexicon(),
        }

        arpa_lm_config = get_arpa_lm_config(
            lm_name="4gram", lexicon_file=lexica["phoneme"]
        )  # also works for bpe lexicon
        arpa_lm_config.scale = 0.6

        blank_indices = {
            "bpe": bpe_to_vocab_size(bpe_size=BPE_SIZE),
            "phoneme": PHONEME_SIZE,
        }

        recog_results = []

        for output_idx, name in enumerate(["phoneme", "bpe"]):
            prior_file = compute_priors(
                prior_data_config=get_default_prior_data(),
                checkpoint=checkpoint,
                model_config=model_config,
                output_idx=output_idx,
            )

            label_scorer_config = get_ctc_label_scorer_config(
                model_config=model_config,
                checkpoint=checkpoint,
                output_idx=output_idx,
                prior_file=prior_file,
                prior_scale=0.2,
                blank_penalty=0.0,
            )

            lexicon = lexica[name]
            blank_index = blank_indices[name]

            encoder_serializers = get_model_serializers(ConformerCTCMultiOutputEncoderModel, model_config=model_config)

            if name == "bpe":

                # =====================================
                # === Lexiconfree Search without LM ===
                # =====================================

                for recog_corpus in ["hub5e00", "hub5e01"]:
                    recog_results.append(
                        recog_rasr(
                            descriptor=f"bpe-phoneme-ctc_{name}-output_lexiconfree",
                            checkpoint=checkpoint,
                            recog_data_config=recog_data[recog_corpus],
                            recog_corpus=score_corpora[recog_corpus],
                            model_serializers=encoder_serializers,
                            rasr_config_file=get_lexiconfree_timesync_recog_config(
                                vocab_file=bpe_vocab_file,
                                collapse_repeated_labels=True,
                                label_scorer_config=get_ctc_label_scorer_config(
                                    model_config=model_config,
                                    checkpoint=checkpoint,
                                    output_idx=output_idx,
                                    prior_file=prior_file,
                                    prior_scale=0.0,
                                    blank_penalty=0.0,
                                ),
                                blank_index=blank_index,
                                max_beam_size=1,  # Lexiconfree search without LM is greedy so only one hyp is needed
                            ),
                            rasr_align_config_file=None,  # Greedy search doesn't have search errors
                            sample_rate=8000,
                        )
                    )

                # =====================================
                # === Tree Search without LM ==========
                # =====================================

                for recog_corpus in ["hub5e00", "hub5e01"]:
                    recog_results.append(
                        recog_rasr(
                            descriptor=f"bpe-phoneme-ctc_{name}-output_tree",
                            checkpoint=checkpoint,
                            recog_data_config=recog_data[recog_corpus],
                            recog_corpus=score_corpora[recog_corpus],
                            model_serializers=encoder_serializers,
                            rasr_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon,
                                collapse_repeated_labels=True,
                                label_scorer_config=label_scorer_config,
                                blank_index=blank_index,
                                max_beam_size=64,
                                score_threshold=12.0,
                                logfile_suffix="recog",
                            ),
                            rasr_align_config_file=get_tree_timesync_recog_config(
                                lexicon_file=lexicon,
                                collapse_repeated_labels=False,
                                label_scorer_config=label_scorer_config,
                                blank_index=blank_index,
                                max_beam_size=256,
                                score_threshold=22.0,
                                logfile_suffix="align",
                            ),
                            sample_rate=8000,
                        )
                    )

            # =====================================
            # === Tree Search with 4gram LM =======
            # =====================================

            for recog_corpus in ["hub5e00", "hub5e01"]:
                arpa_lm_config.scale = 0.6
                recog_results.append(
                    recog_rasr(
                        descriptor=f"bpe-phoneme-ctc_{name}-output_tree_4gram",
                        checkpoint=checkpoint,
                        recog_data_config=recog_data[recog_corpus],
                        recog_corpus=score_corpora[recog_corpus],
                        model_serializers=encoder_serializers,
                        rasr_config_file=get_tree_timesync_recog_config(
                            lexicon_file=lexicon,
                            collapse_repeated_labels=True,
                            label_scorer_config=label_scorer_config,
                            lm_config=arpa_lm_config,
                            blank_index=blank_index,
                            max_beam_size=64,
                            score_threshold=12.0,
                            logfile_suffix="recog",
                        ),
                        rasr_align_config_file=get_tree_timesync_recog_config(
                            lexicon_file=lexicon,
                            collapse_repeated_labels=True,
                            label_scorer_config=label_scorer_config,
                            lm_config=arpa_lm_config,
                            blank_index=blank_index,
                            max_beam_size=256,
                            score_threshold=22.0,
                            logfile_suffix="align",
                        ),
                        sample_rate=8000,
                    )
                )

    tk.register_report(f"{prefix}/report.txt", values=create_report(recog_results), required=True)

    return recog_results
