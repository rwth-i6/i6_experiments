import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.common.setups.serialization import PartialImport

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_test_datasets
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from ....sup_audio_cluster_to_phoneme.librispeech.configs.config_librispeech_960_v1 import (
    base_config as base_config_,
    get_keep_epochs,
    base_num_epochs,
)
from .config_librispeech_960_wo_sil_v1 import (
    base_config,
    alternate_batching,
    _text_recon_variant,
    _train_reflecting_analysis_masking,
    _recon_variant,
    _text_recon_sweep,
    _TEXT_RECON_SWEEP_MASK_PROBS,
)

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)
test_data_dict_wo_sil = build_test_datasets(sil_prob=0.0, surround_w_sil=False)
test_data_dict = build_test_datasets()


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline_gan-adv-0.1_mask-p-0.1-span-1-1",
        config=dict_update_deep(
            copy.deepcopy(base_config),
            {
                "model_args.discriminator_type": "mlp",
                "train_args": {
                    "adv_loss_scale": 0.1,
                    "text_masking_opts": {
                        "mask_prob": 0.1,
                        "min_span": 1,
                        "max_span": 1,
                    },
                    "audio_masking_opts": {
                        "mask_prob": 0.1,
                        "min_span": 1,
                        "max_span": 1,
                    },
                },
            },
        ),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        # skip_eval=True,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        analysis_opts={
            "checkpoints": get_keep_epochs(base_num_epochs),
            "max_plotted_seqs": 20,
            "cosine_similarity_summary": True,
        },
        recog_variants=[
            {
                "recog_name": "recon_text",
                "input_modality": "text",
                "output_modality": "text",
                "mask_input": True,
                "masking_opts": copy.deepcopy(base_config["train_args"]["text_masking_opts"]),
                "keep_epochs": get_keep_epochs(base_num_epochs),
            },
            # fixed-masking text-recon sweep (copy ceiling + degradation curve), for a fair
            # single-task (text-only) vs multi-task comparison of the text denoiser.
            *_text_recon_sweep(base_num_epochs),
        ],
    )

    # discriminator-architecture sweep for the domain-adversarial loss. Same adv scale + masking as
    # baseline_gan-adv-0.1_mask-p-0.1-span-1-1 (which uses the frame-wise "mlp" discriminator), but
    # with discriminators that see more temporal context:
    #   mlp_2gram/3gram/4gram -> MLP over 2/3/4 consecutive frames concatenated in the feature dim
    #   lstm                  -> LSTM over the whole encoder output sequence
    for discriminator_type, fix_decode_text_seq in (
        ("mlp_2gram", False),
        ("mlp_3gram", False),
        ("mlp_4gram", False),
        ("lstm", False),
        ("lstm", True),
    ):
        run_experiment(
            training_name=f"{prefix_name}/baseline_gan-adv-0.1_disc-{discriminator_type}_mask-p-0.1-span-1-1{'_fix-dec-text-seq' if fix_decode_text_seq else ''}",
            config=dict_update_deep(
                copy.deepcopy(base_config),
                {
                    "model_args.discriminator_type": discriminator_type,
                    **({"model_args.fix_decode_text_seq_for_shared_dec": True} if fix_decode_text_seq else {}),
                    "train_args": {
                        "adv_loss_scale": 0.1,
                        "text_masking_opts": {
                            "mask_prob": 0.1,
                            "min_span": 1,
                            "max_span": 1,
                        },
                        "audio_masking_opts": {
                            "mask_prob": 0.1,
                            "min_span": 1,
                            "max_span": 1,
                        },
                    },
                },
            ),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            # skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            # conditional (audio->phoneme) perplexity of the AED model on the last checkpoint (LSTM
            # discriminator only, per request), scored on the wo-silence reference via a separate PPL
            # dataset; recognition / analysis keep the with-silence test_data_dict.
            ppl_opts={
                "checkpoints": [base_num_epochs],
                "input_modality": "audio",
                "test_data_dict": test_data_dict_wo_sil,
            }
            if discriminator_type == "lstm"
            else None,
            recog_variants=[
                {
                    "recog_name": "recon_text",
                    "input_modality": "text",
                    "output_modality": "text",
                    "mask_input": True,
                    "masking_opts": copy.deepcopy(base_config["train_args"]["text_masking_opts"]),
                    "keep_epochs": get_keep_epochs(base_num_epochs),
                },
                # fixed-masking text-recon sweep (copy ceiling + degradation curve), for a fair
                # single-task (text-only) vs multi-task comparison of the text denoiser.
                *_text_recon_sweep(base_num_epochs),
            ],
        )

    # Wasserstein GAN (WGAN-GP) variant of the domain-adversarial loss: the discriminator is used as
    # a critic emitting raw scores; the adversarial loss is E[D(audio)] - E[D(text)] (critic) with a
    # 1-centered gradient penalty (grad_penalty_scale=10.0, the WGAN-GP default). Under alternate
    # batching the penalty is the real-sample variant (see _Discriminator.gradient_penalty). Uses the
    # LSTM discriminator (a stronger critic than the frame-wise MLP), otherwise identical to
    # baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1 so the only difference is the GAN objective.
    run_experiment(
        training_name=f"{prefix_name}/baseline_gan-wasserstein-adv-0.1-gp-10_disc-lstm_mask-p-0.1-span-1-1_bs-12k",
        config=dict_update_deep(
            copy.deepcopy(base_config),
            {
                "model_args.discriminator_type": "lstm",
                "training.batch_size": 12_000,
                "train_args": {
                    "adv_loss_scale": 0.1,
                    "adv_loss_type": "wasserstein",
                    "grad_penalty_scale": 10.0,
                    "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
                    "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
                },
            },
        ),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        analysis_opts={
            "checkpoints": get_keep_epochs(base_num_epochs),
            "max_plotted_seqs": 20,
            "cosine_similarity_summary": True,
        },
        ppl_opts={
            "checkpoints": [base_num_epochs],
            "input_modality": "audio",
            "test_data_dict": test_data_dict_wo_sil,
        },
        recog_variants=[
            {
                "recog_name": "recon_text",
                "input_modality": "text",
                "output_modality": "text",
                "mask_input": True,
                "masking_opts": copy.deepcopy(base_config["train_args"]["text_masking_opts"]),
                "keep_epochs": get_keep_epochs(base_num_epochs),
            },
            *_text_recon_sweep(base_num_epochs),
        ],
    )

    # Faithful WGAN-GP variant (textbook Gulrajani et al.): the critic loss E[D(audio)] - E[D(text)]
    # and the gradient penalty both compare BOTH modalities jointly and interpolate real (text) <->
    # fake (audio) states, which requires both modalities in the same forward. So this variant turns
    # **alternate batching OFF** (deletes training.torch_batching, accum_grad_multiple_step back to 1)
    # -> the interleaved CombinedDataset yields mixed batches (both modalities on different rows).
    # Single-modality batches (if any) simply skip the interpolation loss for that step. LSTM critic,
    # grad_penalty_scale=10.0, otherwise as close as possible to the other GAN variants.
    run_experiment(
        training_name=f"{prefix_name}/baseline_gan-wasserstein-interp-adv-0.1-gp-10_disc-lstm_mask-p-0.1-span-1-1",
        config=dict_update_deep(
            copy.deepcopy(base_config),
            {
                "model_args.discriminator_type": "lstm",
                "training.accum_grad_multiple_step": 1,  # mixed batches: both modalities per step
                "train_args": {
                    "adv_loss_scale": 0.1,
                    "adv_loss_type": "wasserstein_interp",
                    "grad_penalty_scale": 10.0,
                    "text_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
                    "audio_masking_opts": {"mask_prob": 0.1, "min_span": 1, "max_span": 1},
                },
            },
            ["training.torch_batching"],  # drop alternate_batching -> default (mixed) batching
        ),
        train_data=train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        # NB: no alternate_batching prolog here (mixed batches).
        analysis_opts={
            "checkpoints": get_keep_epochs(base_num_epochs),
            "max_plotted_seqs": 20,
            "cosine_similarity_summary": True,
        },
        ppl_opts={
            "checkpoints": [base_num_epochs],
            "input_modality": "audio",
            "test_data_dict": test_data_dict_wo_sil,
        },
        recog_variants=[
            {
                "recog_name": "recon_text",
                "input_modality": "text",
                "output_modality": "text",
                "mask_input": True,
                "masking_opts": copy.deepcopy(base_config["train_args"]["text_masking_opts"]),
                "keep_epochs": get_keep_epochs(base_num_epochs),
            },
            *_text_recon_sweep(base_num_epochs),
        ],
    )

    # text upsampling: duplicate each text (phoneme) token [min_dup, max_dup]x so the text ENCODER
    # input becomes longer than the (unchanged) text reconstruction target, simulating the audio>text
    # length ratio the model faces at audio->text decoding. The target + masking stay at the original
    # length, so only the encoder/cross-attention sees the longer sequence (see
    # train_steps.util.expand_sequence). Applied on the plain baseline and on the LSTM-discriminator
    # GAN (where the longer text sequences also remove length as a trivial modality cue for the
    # domain-adversarial loss). Both use mask_prob 0.1 / span 1 for both modalities. The loop varies
    # (base variant) x (expansion range), so further ablations are one line in either tuple below.
    _upsample_mask_opts = {"mask_prob": 0.1, "min_span": 1, "max_span": 1}

    for exp_idx, (config, train_name) in enumerate(
        [
            (
                dict_update_deep(
                    copy.deepcopy(base_config),
                    {
                        **disc_model_args,
                        "train_args": {
                            **disc_train_args,
                            "text_expansion_opts": {
                                "min_dup": min_dup,
                                "max_dup": max_dup,
                            },
                            "text_masking_opts": copy.deepcopy(_upsample_mask_opts),
                            "audio_masking_opts": copy.deepcopy(_upsample_mask_opts),
                        },
                        "training.batch_size": batch_size,
                    },
                ),
                f"{base_name}_text-upsample-{min_dup}-{max_dup}_mask-p-0.1-span-1-1_bs-{batch_size}",
            )
            for base_name, disc_model_args, disc_train_args in (
                (
                    "baseline_gan-adv-0.1_disc-lstm",
                    {"model_args.discriminator_type": "lstm"},
                    {"adv_loss_scale": 0.1},
                ),
            )
            for min_dup, max_dup, batch_size in ((1, 2, 12_000), (1, 3, 8_000))
        ]
    ):
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            # skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            # PCA reflects training: mask + upsample the encoder input as in training, so the
            # visualization shows the states the shared encoder actually sees (see
            # _train_reflecting_analysis_masking).
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
                **_train_reflecting_analysis_masking(config),
            },
            recog_variants=[
                # main text->text recon reflects training: masking + text upsampling as in training
                # (both come from this variant's train_args via _text_recon_variant).
                _text_recon_variant(config, base_num_epochs, keep_epochs=get_keep_epochs(base_num_epochs)),
                # the fixed-masking sweep intentionally stays as-is (no upsampling) for comparison.
                *_text_recon_sweep(base_num_epochs),
            ],
        )
