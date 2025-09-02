from typing import List, Literal, Optional, Tuple

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
from sisyphus import gs, tk

from ...data.switchboard import datasets as switchboard_datasets
from ...data.switchboard import lm as switchboard_lm
from ...data.switchboard.bpe import bpe_to_vocab_size, get_bpe_vocab_file, vocab_to_bpe_size
from ...data.switchboard.lexicon import get_bpe_bliss_lexicon
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
from ...model_pipelines.common.recog import (
    OfflineRecogResult,
    OfflineRecogResultWithSearchErrors,
    RecogResult,
    recog_rasr_offline,
    recog_rasr_offline_with_search_errors,
)
from ...model_pipelines.common.recog_rasr_config import (
    get_lexiconfree_labelsync_recog_config,
    get_tree_labelsync_recog_config,
)
from ...model_pipelines.common.report import (
    create_base_recog_report,
    create_offline_recog_report_with_search_errors,
    create_streaming_recog_report,
)
from ...model_pipelines.common.serializers import get_model_serializers


def get_model_config() -> AEDConfig:
    vocab_size = bpe_to_vocab_size(bpe_size=128)
    return AEDConfig(
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
            start_epoch=13,
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
            target_embed_dim=256,
            target_embed_dropout=0.1,
            lstm_hidden_size=640,
            zoneout_drop_h=0.05,
            zoneout_drop_c=0.10,
            output_proj_dim=640,
            output_dropout=0.1,
            attention_cfg=AdditiveAttentionConfig(
                attention_dim=640,
                att_weights_dropout=0.1,
            ),
        ),
        label_target_size=vocab_size,
    )


def get_train_options() -> AEDTrainOptions:
    train_data_config = switchboard_datasets.get_default_bpe_train_data(bpe_size=128)
    assert train_data_config.target_config
    train_data_config.target_config["seq_postfix"] = [0]

    cv_data_config = switchboard_datasets.get_default_bpe_cv_data(bpe_size=128)
    assert cv_data_config.target_config
    cv_data_config.target_config["seq_postfix"] = [0]

    return AEDTrainOptions(
        train_data_config=train_data_config,
        cv_data_config=cv_data_config,
        save_epochs=list(range(450, 570, 30)) + list(range(570, 601, 6)),
        batch_size=24_000 * 80,
        accum_grad_multiple_step=1,
        optimizer_config=RAdamConfig(
            epsilon=1e-12,
            weight_decay=0.001,
            decoupled_weight_decay=True,
        ),
        lr_config=ConstConstDecayLRConfig(
            const_lr_1=1e-05,
            const_lr_2=1e-04,
            decayed_lr=1e-05,
            final_lr=1e-07,
            const_epochs_1=12,
            const_epochs_2=276,
            dec_epochs=288,
            final_epochs=24,
        ),
        gradient_clip=1.0,
        ctc_loss_scale=0.7,
        label_smoothing=0.1,
        label_smoothing_start_epoch=19,
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


def run_recognitions_offline_lexiconfree(
    checkpoint: PtCheckpoint,
    model_config: AEDConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["hub5e00", "hub5e01"]]] = None,
    max_beam_size: int = 64,
    score_threshold: float = 0.03,
) -> List[OfflineRecogResult]:
    model_serializers = get_model_serializers(AEDEncoder, model_config)

    label_scorer_config = get_aed_label_scorer_config(model_config=model_config, checkpoint=checkpoint)

    rasr_config_file = get_lexiconfree_labelsync_recog_config(
        vocab_file=get_bpe_vocab_file(bpe_size=vocab_to_bpe_size(model_config.label_target_size), add_blank=False),
        label_scorer_config=label_scorer_config,
        sentence_end_index=0,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        length_norm_scale=1.2,
        max_labels_per_time_step=1,
    )

    recog_results = []

    for recog_corpus in corpora or ["hub5e00", "hub5e01"]:
        recog_results.append(
            recog_rasr_offline(
                descriptor=f"{descriptor}_lexiconfree",
                checkpoint=checkpoint,
                recog_data_config=switchboard_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=switchboard_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=rasr_config_file,
                sample_rate=8000,
            )
        )

    return recog_results


def run_recognitions_offline_tree(
    checkpoint: PtCheckpoint,
    model_config: AEDConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["hub5e00", "hub5e01"]]] = None,
    max_beam_size: int = 64,
    score_threshold: float = 0.03,
    max_word_end_beam_size: int = 16,
    word_end_score_threshold: float = 0.5,
    global_score_threshold: float = 0.5,
) -> List[OfflineRecogResultWithSearchErrors]:
    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.label_target_size), add_blank=False)

    model_serializers = get_model_serializers(AEDEncoder, model_config)

    label_scorer_config = get_aed_label_scorer_config(model_config=model_config, checkpoint=checkpoint)

    recog_rasr_config_file = get_tree_labelsync_recog_config(
        lexicon_file=lexicon_file,
        label_scorer_config=label_scorer_config,
        max_beam_size=max_beam_size,
        max_word_end_beam_size=max_word_end_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        global_score_threshold=global_score_threshold,
        length_norm_scale=1.2,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_labelsync_recog_config(
        lexicon_file=lexicon_file,
        label_scorer_config=label_scorer_config,
        max_beam_size=256,
        max_word_end_beam_size=max_word_end_beam_size,
        score_threshold=1.0,
        length_norm_scale=1.2,
        logfile_suffix="align",
    )

    recog_results = []

    for recog_corpus in corpora or ["hub5e00", "hub5e01"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree",
                checkpoint=checkpoint,
                recog_data_config=switchboard_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=switchboard_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=8000,
            )
        )

    return recog_results


def run_recognitions_offline_tree_4gram(
    checkpoint: PtCheckpoint,
    model_config: AEDConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["hub5e00", "hub5e01"]]] = None,
    lm_scale: float = 0.6,
    max_beam_size: int = 64,
    score_threshold: float = 0.03,
    max_word_end_beam_size: int = 16,
    word_end_score_threshold: float = 0.5,
    global_score_threshold: float = 0.5,
) -> List[OfflineRecogResultWithSearchErrors]:
    model_serializers = get_model_serializers(AEDEncoder, model_config)

    label_scorer_config = get_aed_label_scorer_config(model_config=model_config, checkpoint=checkpoint)

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.label_target_size), add_blank=False)
    arpa_lm_config = switchboard_lm.get_arpa_lm_config(lm_name="4gram", lexicon_file=lexicon_file, scale=lm_scale)

    recog_rasr_config_file = get_tree_labelsync_recog_config(
        lexicon_file=lexicon_file,
        label_scorer_config=label_scorer_config,
        lm_config=arpa_lm_config,
        max_beam_size=max_beam_size,
        max_word_end_beam_size=max_word_end_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        global_score_threshold=global_score_threshold,
        length_norm_scale=1.2,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_labelsync_recog_config(
        lexicon_file=lexicon_file,
        label_scorer_config=label_scorer_config,
        max_beam_size=256,
        max_word_end_beam_size=max_word_end_beam_size,
        score_threshold=1.0,
        length_norm_scale=1.2,
        logfile_suffix="align",
    )

    recog_results = []

    for recog_corpus in corpora or ["hub5e00", "hub5e01"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree_word-4gram",
                checkpoint=checkpoint,
                recog_data_config=switchboard_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=switchboard_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=8000,
            )
        )

    return recog_results


def run_base_recognition_suite(
    checkpoint: PtCheckpoint,
    model_config: AEDConfig,
    descriptor: str = "recog",
    lexiconfree_search: bool = True,
    tree_search: bool = True,
    tree_4gram_search: bool = True,
) -> List[RecogResult]:
    offline_recog_results = []
    streaming_recog_results = []

    if lexiconfree_search:
        offline_recog_results.extend(
            run_recognitions_offline_lexiconfree(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if tree_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree(checkpoint=checkpoint, model_config=model_config, descriptor=descriptor)
        )
    if tree_4gram_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree_4gram(checkpoint=checkpoint, model_config=model_config, descriptor=descriptor)
        )

    all_recog_results = offline_recog_results + streaming_recog_results

    if offline_recog_results:
        tk.register_report(
            f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report_offline.txt",
            values=create_offline_recog_report_with_search_errors(offline_recog_results),
            required=True,
        )
    if streaming_recog_results:
        tk.register_report(
            f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report_streaming.txt",
            values=create_streaming_recog_report(streaming_recog_results),
            required=True,
        )
    if all_recog_results:
        tk.register_report(
            f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/report.txt",
            values=create_base_recog_report(all_recog_results),
            required=True,
        )

    return all_recog_results


def run_all() -> List[RecogResult]:
    with ExperimentContext("bpe_aed"):
        train_job, model_config = run_training()
        checkpoint: PtCheckpoint = train_job.out_checkpoints[510]  # type: ignore
        recog_results = run_base_recognition_suite(
            checkpoint=checkpoint,
            model_config=model_config,
            descriptor="bpe_aed",
        )
    return recog_results
