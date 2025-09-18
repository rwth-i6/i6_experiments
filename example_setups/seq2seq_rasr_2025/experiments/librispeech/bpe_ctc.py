from dataclasses import fields
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

from ...data.librispeech import datasets as librispeech_datasets
from ...data.librispeech import lm as librispeech_lm
from ...data.librispeech.bpe import bpe_to_vocab_size, get_bpe_vocab_file, vocab_to_bpe_size
from ...data.librispeech.lexicon import get_bpe_bliss_lexicon, get_tedlium2_bpe_bliss_lexicon
from ...data.tedlium2 import datasets as tedlium2_datasets
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.learning_rates import OCLRConfig
from ...model_pipelines.common.optimizer import AdamWConfig
from ...model_pipelines.common.recog import (
    OfflineRecogResult,
    OfflineRecogResultWithSearchErrors,
    RecogResult,
    StreamingRecogResult,
    recog_rasr_offline,
    recog_rasr_offline_with_search_errors,
    recog_rasr_streaming,
)
from ...model_pipelines.common.recog_rasr_config import (
    get_combine_label_scorer_config,
    get_lexiconfree_timesync_recog_config,
    get_no_op_label_scorer_config,
    get_tree_timesync_recog_config,
)
from ...model_pipelines.common.report import (
    create_base_recog_report,
    create_offline_recog_report_with_search_errors,
    create_streaming_recog_report,
)
from ...model_pipelines.common.serializers import get_model_serializers
from ...model_pipelines.common.train import TrainOptions
from ...model_pipelines.ctc.prior import compute_priors
from ...model_pipelines.ctc.pytorch_modules import (
    ConformerCTCConfig,
    ConformerCTCRecogConfig,
    ConformerCTCRecogModel,
    SpecaugmentByLengthConfig,
)
from ...model_pipelines.ctc.train import train


def get_model_config(
    bpe_size: int = 128,
    layer_size: int = 512,
    num_att_heads: int = 8,
) -> ConformerCTCConfig:
    return ConformerCTCConfig(
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
        gpu_mem_rqmt=24,
        max_seqs=None,
        max_seq_length=None,
    )


def run_training(
    model_config: Optional[ConformerCTCConfig] = None,
    train_options: Optional[TrainOptions] = None,
) -> Tuple[ReturnnTrainingJob, ConformerCTCConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    train_job = train(options=train_options, model_config=model_config)

    return train_job, model_config


def run_recognitions_offline_lexiconfree(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
) -> List[OfflineRecogResult]:
    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=None,
            prior_scale=0.0,
            blank_penalty=0.0,
        ),
    )

    rasr_config_file = get_lexiconfree_timesync_recog_config(
        vocab_file=get_bpe_vocab_file(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=model_config.target_size - 1,
        max_beam_size=1,  # Lexiconfree search without LM is greedy so only one hyp is needed
        score_threshold=0.0,
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_offline(
                descriptor=f"{descriptor}_lexiconfree",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=rasr_config_file,
                sample_rate=16000,
            )
        )

    return recog_results


def run_recognitions_offline_lexiconfree_lstm(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.8,
    max_beam_size: int = 70,
    score_threshold: float = 8.0,
    intermediate_score_threshold: float = 8.0,
    intermediate_max_beam_size: int = 150,
) -> List[OfflineRecogResult]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    lstm_lm_config = librispeech_lm.get_bpe_lstm_label_scorer_config(
        bpe_size=vocab_to_bpe_size(model_config.target_size - 1),
        use_gpu=True,
    )

    recog_rasr_config_file = get_lexiconfree_timesync_recog_config(
        vocab_file=get_bpe_vocab_file(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True),
        collapse_repeated_labels=True,
        label_scorer_config=get_combine_label_scorer_config(
            sub_scorers=[
                (get_no_op_label_scorer_config(), 1.0),
                (lstm_lm_config, lm_scale),
            ]
        ),
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        intermediate_score_threshold=intermediate_score_threshold,
        intermediate_max_beam_size=intermediate_max_beam_size,
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_offline(
                descriptor=f"{descriptor}_lexiconfree_bpe-lstmLM",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=recog_rasr_config_file,
                sample_rate=16000,
                gpu_mem_rqmt=24,
            )
        )

    return recog_results


def run_recognitions_offline_tree(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    max_beam_size: int = 1024,
    score_threshold: float = 14.0,
) -> List[OfflineRecogResultWithSearchErrors]:
    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=None,
            prior_scale=0.0,
            blank_penalty=0.0,
        ),
    )

    recog_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=model_config.target_size - 1,
        max_beam_size=4096,
        score_threshold=22.0,
        logfile_suffix="align",
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=16000,
            )
        )

    return recog_results


def run_recognitions_offline_tree_4gram(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.6,
    max_beam_size: int = 1024,
    score_threshold: float = 14.0,
    word_end_score_threshold: float = 0.5,
) -> List[OfflineRecogResultWithSearchErrors]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)
    arpa_lm_config = librispeech_lm.get_arpa_lm_config(lm_name="4gram", lexicon_file=lexicon_file, scale=lm_scale)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    recog_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=4096,
        score_threshold=22.0,
        logfile_suffix="align",
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree_word-4gram",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=16000,
            )
        )

    return recog_results


def run_recognitions_offline_tree_lstm(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.8,
    max_beam_size: int = 30,
    intermediate_max_beam_size: int = 80,
    word_end_beam_size: int = 20,
    score_threshold: float = 12.0,
    intermediate_score_threshold: float = 12.0,
    word_end_score_threshold: float = 1.0,
) -> List[OfflineRecogResultWithSearchErrors]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)
    arpa_lm_config = librispeech_lm.get_arpa_lm_config(lm_name="4gram", lexicon_file=lexicon_file, scale=lm_scale)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    lstm_lm_config = librispeech_lm.get_bpe_lstm_label_scorer_config(
        bpe_size=vocab_to_bpe_size(model_config.target_size - 1),
        use_gpu=True,
    )

    recog_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_combine_label_scorer_config(
            sub_scorers=[
                (get_no_op_label_scorer_config(), 1.0),
                (lstm_lm_config, lm_scale),
            ]
        ),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        intermediate_score_threshold=intermediate_score_threshold,
        intermediate_max_beam_size=intermediate_max_beam_size,
        word_end_score_threshold=word_end_score_threshold,
        max_word_end_beam_size=word_end_beam_size,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=4096,
        score_threshold=22.0,
        logfile_suffix="align",
    )

    recog_results = []

    # for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for recog_corpus in corpora or ["dev-other", "test-other"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree_bpe-lstmLM",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=16000,
                gpu_mem_rqmt=24,
            )
        )

    return recog_results


def run_recognitions_offline_tree_lstm_4gram(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lstm_lm_scale: float = 0.8,
    lm_scale: float = 0.3,
    max_beam_size: int = 70,
    score_threshold: float = 12.0,
    intermediate_score_threshold: float = 12.0,
    intermediate_max_beam_size: int = 80,
    word_end_score_threshold: float = 1.0,
    max_word_end_beam_size: int = 50,
) -> List[OfflineRecogResultWithSearchErrors]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)
    arpa_lm_config = librispeech_lm.get_arpa_lm_config(lm_name="4gram", lexicon_file=lexicon_file, scale=lm_scale)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    lstm_lm_config = librispeech_lm.get_bpe_lstm_label_scorer_config(
        bpe_size=vocab_to_bpe_size(model_config.target_size - 1),
        use_gpu=True,
    )

    recog_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_combine_label_scorer_config(
            sub_scorers=[
                (get_no_op_label_scorer_config(), 1.0),
                (lstm_lm_config, lstm_lm_scale),
            ]
        ),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        intermediate_max_beam_size=intermediate_max_beam_size,
        max_word_end_beam_size=max_word_end_beam_size,
        score_threshold=score_threshold,
        intermediate_score_threshold=intermediate_score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=4096,
        score_threshold=22.0,
        logfile_suffix="align",
    )

    recog_results = []

    # for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for recog_corpus in corpora or ["dev-other", "test-other"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree_word-4gram_bpe-lstmLM",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=16000,
                gpu_mem_rqmt=24,
            )
        )

    return recog_results


def run_recognitions_offline_tree_trafo(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.8,
    max_beam_size: int = 1024,
    score_threshold: float = 14.0,
    word_end_score_threshold: float = 0.5,
) -> List[OfflineRecogResultWithSearchErrors]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)
    trafo_lm_config = librispeech_lm.get_transformer_lm_config(lm_scale=lm_scale)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    recog_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=trafo_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=trafo_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=4096,
        score_threshold=22.0,
        logfile_suffix="align",
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree_word-trafoLM",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=16000,
                gpu_mem_rqmt=24,
            )
        )

    return recog_results


def run_recognitions_streaming_lexiconfree(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    chunk_history_seconds: float = 10.0,
    chunk_center_seconds: float = 1.0,
    chunk_future_seconds: float = 1.0,
    encoder_frame_shift_seconds: float = 0.04,
) -> List[StreamingRecogResult]:
    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=None,
            prior_scale=0.0,
            blank_penalty=0.0,
        ),
    )

    rasr_config_file = get_lexiconfree_timesync_recog_config(
        vocab_file=get_bpe_vocab_file(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=model_config.target_size - 1,
        max_beam_size=1,
        score_threshold=0.0,
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_streaming(
                descriptor=f"{descriptor}_lexiconfree_stream",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=rasr_config_file,
                encoder_frame_shift_seconds=encoder_frame_shift_seconds,
                chunk_history_seconds=chunk_history_seconds,
                chunk_center_seconds=chunk_center_seconds,
                chunk_future_seconds=chunk_future_seconds,
                sample_rate=16000,
            )
        )

    return recog_results


def run_recognitions_streaming_tree_4gram(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.6,
    max_beam_size: int = 1024,
    score_threshold: float = 14.0,
    word_end_score_threshold: float = 0.5,
    maximum_stable_delay: int = 15,
    chunk_history_seconds: float = 10.0,
    chunk_center_seconds: float = 1.0,
    chunk_future_seconds: float = 1.0,
    encoder_frame_shift_seconds: float = 0.04,
) -> List[StreamingRecogResult]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)
    arpa_lm_config = librispeech_lm.get_arpa_lm_config(lm_name="4gram", lexicon_file=lexicon_file, scale=lm_scale)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        maximum_stable_delay=maximum_stable_delay,
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_streaming(
                descriptor=f"{descriptor}_tree_word-4gram_stream",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=rasr_config_file,
                encoder_frame_shift_seconds=encoder_frame_shift_seconds,
                chunk_history_seconds=chunk_history_seconds,
                chunk_center_seconds=chunk_center_seconds,
                chunk_future_seconds=chunk_future_seconds,
                sample_rate=16000,
            )
        )

    return recog_results


def run_recognitions_offline_tree_trafo_kazuki(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev-clean", "dev-other", "test-clean", "test-other"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.8,
    max_beam_size: int = 512,
    score_threshold: float = 16.0,
    word_end_score_threshold: float = 0.5,
) -> List[OfflineRecogResultWithSearchErrors]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_bpe_bliss_lexicon(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True)
    trafo_lm_config = librispeech_lm.get_kazuki_trafo_lm_config(lm_scale=lm_scale)

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    recog_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=trafo_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
        logfile_suffix="recog",
    )

    align_rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=trafo_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=4096,
        score_threshold=22.0,
        logfile_suffix="align",
    )

    recog_results = []

    for recog_corpus in corpora or ["dev-clean", "dev-other", "test-clean", "test-other"]:
        recog_results.append(
            recog_rasr_offline_with_search_errors(
                descriptor=f"{descriptor}_tree_word-trafoLM-kazuki",
                checkpoint=checkpoint,
                recog_data_config=librispeech_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=librispeech_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                recog_rasr_config_file=recog_rasr_config_file,
                align_rasr_config_file=align_rasr_config_file,
                sample_rate=16000,
                mem_rqmt=24,
                gpu_mem_rqmt=24,
            )
        )

    return recog_results


def run_recognitions_tedlium_offline_lexiconfree(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev", "test"]]] = None,
) -> List[OfflineRecogResultWithSearchErrors]:

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=None,
            prior_scale=0.0,
            blank_penalty=0.0,
        ),
    )

    rasr_config_file = get_lexiconfree_timesync_recog_config(
        vocab_file=get_bpe_vocab_file(bpe_size=vocab_to_bpe_size(model_config.target_size - 1), add_blank=True),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=model_config.target_size - 1,
        max_beam_size=1,  # Lexiconfree search without LM is greedy so only one hyp is needed
    )

    recog_results = []

    for recog_corpus in corpora or ["dev", "test"]:
        recog_results.append(
            recog_rasr_offline(
                descriptor=f"{descriptor}_lexiconfree",
                checkpoint=checkpoint,
                recog_data_config=tedlium2_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=tedlium2_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=rasr_config_file,
                sample_rate=16000,
            )
        )

    for recog_result in recog_results:
        recog_result.corpus_name = "tedlium-" + recog_result.corpus_name

    return recog_results


def run_recognitions_tedlium_offline_tree_4gram(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    corpora: Optional[List[Literal["dev", "test"]]] = None,
    prior_scale: float = 0.2,
    lm_scale: float = 0.6,
    max_beam_size: int = 1024,
    score_threshold: float = 14.0,
    word_end_score_threshold: float = 0.5,
) -> List[OfflineRecogResultWithSearchErrors]:
    prior_file = compute_priors(
        prior_data_config=librispeech_datasets.get_default_prior_data(),
        model_config=model_config,
        checkpoint=checkpoint,
    )

    lexicon_file = get_tedlium2_bpe_bliss_lexicon(
        bpe_size=vocab_to_bpe_size(model_config.target_size - 1),
        add_blank=True,
    )
    arpa_lm_config = librispeech_lm.get_tedlium2_arpa_lm_config(
        lm_name="4gram", lexicon_file=lexicon_file, scale=lm_scale
    )

    model_serializers = get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            prior_file=prior_file,
            prior_scale=prior_scale,
            blank_penalty=0.0,
        ),
    )

    rasr_config_file = get_tree_timesync_recog_config(
        lexicon_file=lexicon_file,
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        lm_config=arpa_lm_config,
        blank_index=model_config.target_size - 1,
        max_beam_size=max_beam_size,
        score_threshold=score_threshold,
        word_end_score_threshold=word_end_score_threshold,
    )

    recog_results = []

    for recog_corpus in corpora or ["dev", "test"]:
        recog_results.append(
            recog_rasr_offline(
                descriptor=f"{descriptor}_tree_word-4gram",
                checkpoint=checkpoint,
                recog_data_config=tedlium2_datasets.get_default_recog_data(recog_corpus),
                recog_corpus=tedlium2_datasets.get_default_score_corpus(recog_corpus),
                encoder_serializers=model_serializers,
                rasr_config_file=rasr_config_file,
                sample_rate=16000,
            )
        )

    for recog_result in recog_results:
        recog_result.corpus_name = "tedlium-" + recog_result.corpus_name

    return recog_results


def run_base_recognition_suite(
    checkpoint: PtCheckpoint,
    model_config: ConformerCTCConfig,
    descriptor: str = "recog",
    lexiconfree_search: bool = True,
    lexiconfree_lstm_search: bool = True,
    tree_search: bool = True,
    tree_lstm_search: bool = True,
    tree_4gram_search: bool = True,
    tree_4gram_lstm_search: bool = True,
    tree_trafo_search: bool = True,
    tree_trafo_kazuki_search: bool = True,
    lexiconfree_streaming_search: bool = True,
    tree_streaming_search: bool = True,
    lexiconfree_tedlium_search: bool = True,
    tree_4gram_tedlium_search: bool = True,
) -> List[RecogResult]:
    offline_recog_results = []
    streaming_recog_results = []

    if lexiconfree_search:
        offline_recog_results.extend(
            run_recognitions_offline_lexiconfree(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if lexiconfree_lstm_search:
        offline_recog_results.extend(
            run_recognitions_offline_lexiconfree_lstm(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if tree_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree(checkpoint=checkpoint, model_config=model_config, descriptor=descriptor)
        )
    if tree_lstm_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree_lstm(checkpoint=checkpoint, model_config=model_config, descriptor=descriptor)
        )
    if tree_4gram_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree_4gram(checkpoint=checkpoint, model_config=model_config, descriptor=descriptor)
        )
    if tree_4gram_lstm_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree_lstm_4gram(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if tree_trafo_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree_trafo(checkpoint=checkpoint, model_config=model_config, descriptor=descriptor)
        )
    if tree_trafo_kazuki_search:
        offline_recog_results.extend(
            run_recognitions_offline_tree_trafo_kazuki(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if lexiconfree_tedlium_search:
        offline_recog_results.extend(
            run_recognitions_tedlium_offline_lexiconfree(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if tree_4gram_tedlium_search:
        offline_recog_results.extend(
            run_recognitions_tedlium_offline_tree_4gram(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if lexiconfree_streaming_search:
        streaming_recog_results.extend(
            run_recognitions_streaming_lexiconfree(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
        )
    if tree_streaming_search:
        streaming_recog_results.extend(
            run_recognitions_streaming_tree_4gram(
                checkpoint=checkpoint, model_config=model_config, descriptor=descriptor
            )
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
    with ExperimentContext("bpe_ctc"):
        train_job, model_config = run_training()
        checkpoint: PtCheckpoint = train_job.out_checkpoints[max(train_job.out_checkpoints)]  # type: ignore
        recog_results = run_base_recognition_suite(
            checkpoint=checkpoint,
            model_config=model_config,
            descriptor="bpe_ctc",
            tree_trafo_search=False,
            # tree_trafo_kazuki_search=True,
        )
    return recog_results
