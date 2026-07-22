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

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)
test_data_dict_wo_sil = build_test_datasets(sil_prob=0.0, surround_w_sil=False)
test_data_dict = build_test_datasets()


base_config = dict_update_deep(
    base_config_,
    {
        "__train_step_module": "train_steps.aed_denoising_discrete_shared.train_step",
        "training": {
            "torch_batching": CodeWrapper("alternate_batching"),
            "accum_grad_multiple_step": 2,  # alternate batching
        },
        "train_post_config": {
            "tensorboard_opts": {
                # uneven so that both text and audio losses get logged (alternated batching)
                "log_every_n_train_steps": 51,
            },
        },
        "general.default_target_key": "phon_indices",
        "model_args": {
            "text_out_dim": train_data.datastreams["phon_indices"].vocab_size,
            "audio_out_dim": train_data.datastreams["data"].vocab_size,
        },
        "train_args": {
            "aux_loss_scales": (),
            "text_ce_loss_scale": 0.2,
            "text_masked_ce_loss_scale": 1.0,
            "audio_ce_loss_scale": 0.2,
            "audio_masked_ce_loss_scale": 1.0,
            "text_masking_opts": {
                "mask_prob": 0.3,
                "min_span": 2,  # 1
                "max_span": 10,  # 3
            },
            "audio_masking_opts": {
                "mask_prob": 0.3,
                "min_span": 4,  # 1
                "max_span": 20,  # 3
            },
        },
    },
    [
        "train_args.masking_opts",
        "train_args.ce_loss_scale",
        "train_args.masked_ce_loss_scale",
    ],
)


alternate_batching = PartialImport(
    code_object_path="i6_experiments.users.schmitt.returnn.alternate_batching.alternate_batching",
    import_as="alternate_batching",
    hashed_arguments={},
    unhashed_arguments={},
    unhashed_package_root=None,
)


def _text_recon_variant(config, num_epochs, keep_epochs=None):
    """text->text reconstruction recog, masking the input with the experiment's own training text
    masking settings (scored against the unmasked phoneme reference). If the experiment also sets
    ``train_args.text_expansion_opts`` (text upsampling), the same upsampling is applied to the recon
    input so recognition matches training. ``keep_epochs`` (list) overrides the default last-epoch."""
    v = {
        "recog_name": "recon_text",
        "input_modality": "text",
        "output_modality": "text",
        "mask_input": True,
        "masking_opts": copy.deepcopy(config["train_args"]["text_masking_opts"]),
        "keep_epochs": keep_epochs if keep_epochs is not None else [num_epochs],
    }
    text_expansion_opts = config["train_args"].get("text_expansion_opts")
    if text_expansion_opts is not None:
        v["expansion_opts"] = copy.deepcopy(text_expansion_opts)
    return v


def _train_reflecting_analysis_masking(config):
    """Masking/upsampling opts for the encoder-PCA analysis that reflect the experiment's training
    settings (audio + text masking, and text upsampling if set), so the PCA shows what the shared
    encoder sees during training rather than the raw (unmasked, un-upsampled) input."""
    opts = {
        "audio_masking_opts": copy.deepcopy(config["train_args"]["audio_masking_opts"]),
        "text_masking_opts": copy.deepcopy(config["train_args"]["text_masking_opts"]),
    }
    text_expansion_opts = config["train_args"].get("text_expansion_opts")
    if text_expansion_opts is not None:
        opts["text_expansion_opts"] = copy.deepcopy(text_expansion_opts)
    return opts


def _recon_variant(num_epochs, *, input_modality, output_modality, recog_name, mask_prob=0.0, min_span=2, max_span=10):
    """Generic same-modality reconstruction recog on the last epoch. mask_prob=0.0 -> no masking
    (pure copy through the shared enc+dec, i.e. the autoencoder ceiling)."""
    v = {
        "recog_name": recog_name,
        "input_modality": input_modality,
        "output_modality": output_modality,
        "keep_epochs": [num_epochs],
    }
    if mask_prob > 0.0:
        v["mask_input"] = True
        v["masking_opts"] = {"mask_prob": mask_prob, "min_span": min_span, "max_span": max_span}
    return v


# fixed-masking text-recon sweep (span 2-10) at a common set of mask probs, so the text denoiser can
# be characterized (copy ceiling at 0.0 + degradation curve) and compared fairly across experiments.
# 0.3 is already covered by _text_recon_variant (the base training masking), so it is not repeated.
_TEXT_RECON_SWEEP_MASK_PROBS = (0.0, 0.1, 0.5)


def _text_recon_sweep(num_epochs):
    return [
        _recon_variant(
            num_epochs,
            input_modality="text",
            output_modality="text",
            recog_name=f"recon_text_mask-{p}",
            mask_prob=p,
        )
        for p in _TEXT_RECON_SWEEP_MASK_PROBS
    ]


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
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
        # conditional (audio->phoneme) perplexity of the shared AED model on the last checkpoint,
        # scored on the wo-silence reference (matching the wo-sil model) via a separate PPL dataset;
        # recognition / analysis keep the with-silence test_data_dict.
        ppl_opts={
            "checkpoints": [base_num_epochs],
            "input_modality": "audio",
            "test_data_dict": test_data_dict_wo_sil,
        },
        # same-modality reconstruction on the last checkpoint, masking the input with the same
        # settings as in training, to probe how well the shared denoising model reconstructs each
        # modality (scored against the unmasked input).
        recog_variants=[
            {
                "recog_name": "recon_audio",
                "input_modality": "audio",
                "output_modality": "audio",
                "mask_input": True,
                "masking_opts": copy.deepcopy(base_config["train_args"]["audio_masking_opts"]),
                "keep_epochs": [base_num_epochs],
            },
            {
                "recog_name": "recon_text",
                "input_modality": "text",
                "output_modality": "text",
                "mask_input": True,
                "masking_opts": copy.deepcopy(base_config["train_args"]["text_masking_opts"]),
                "keep_epochs": [base_num_epochs],
            },
            # fixed-masking text-recon sweep (copy ceiling + degradation curve), for a fair
            # single-task (text-only) vs multi-task comparison of the text denoiser.
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
            for base_name, disc_model_args, disc_train_args in (("baseline", {}, {}),)
            for min_dup, max_dup, batch_size in ((1, 2, 12_000), (1, 3, 10_000))
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

    # baseline settings (3 enc / 3 dec layers) + a GumbelVectorQuantizer codebook on top of the
    # shared encoder (à la SpeechT5), to push the audio and text encoder states into a shared
    # discrete space. The codebook diversity loss encourages using the full codebook.
    codebook_config = dict_update_deep(
        copy.deepcopy(base_config),
        {
            "model_args": {"codebook_opts": {}},  # {} -> enable codebook with default settings
            "train_args": {"codebook_diversity_loss_scale": 0.1},
            "training.batch_size": 10_000,
        },
    )
    run_experiment(
        training_name=f"{prefix_name}/baseline_codebook",
        config=copy.deepcopy(codebook_config),
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
        recog_variants=[_text_recon_variant(codebook_config, base_num_epochs)],
    )

    # NOTE: the single-task text-only reference moved to config_librispeech_960_text_only_v1.py

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(base_config),
                        {
                            "model_args": {
                                "num_enc_layers": num_enc_layers,
                                "num_text_dec_layers": num_dec_layers,
                                "num_audio_dec_layers": num_dec_layers,
                            },
                            "training.batch_size": batch_size,
                            # the deeper enc-6 model occasionally yields a single non-finite
                            # train score early on (e.g. ep1 step66) which otherwise aborts the
                            # whole training; skip such steps instead. Kept in the (hashed)
                            # training config since it changes training behavior.
                            "training.stop_on_nonfinite_train_score": False,
                        },
                    ),
                    f"baseline_enc-{num_enc_layers}_dec-{num_dec_layers}_bs-{batch_size}",
                )
                for num_enc_layers, num_dec_layers, batch_size in ((6, 6, 8_000),)
            ]
        ]
    ):
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            recog_variants=[_text_recon_variant(config, num_epochs)],
        )

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(base_config),
                        {
                            "train_args": {
                                "text_masking_opts": {
                                    "mask_prob": mask_prob,
                                    "min_span": min_span,
                                    "max_span": max_span,
                                },
                            },
                        },
                    ),
                    f"baseline_text-mask-p-{mask_prob}-span-{min_span}-{max_span}",
                )
                for mask_prob, min_span, max_span in ((0.4, 2, 10),)
            ]
        ]
    ):
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            recog_variants=[_text_recon_variant(config, num_epochs)],
        )

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(base_config),
                        {
                            "train_args": {
                                "audio_masking_opts": {
                                    "mask_prob": mask_prob,
                                    "min_span": min_span,
                                    "max_span": max_span,
                                },
                            },
                        },
                    ),
                    f"baseline_audio-mask-p-{mask_prob}-span-{min_span}-{max_span}",
                )
                for mask_prob, min_span, max_span in ((0.4, 4, 20),)
            ]
        ]
    ):
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            recog_variants=[_text_recon_variant(config, num_epochs)],
        )

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(base_config),
                        {
                            "train_args": {
                                "audio_masking_opts": {
                                    "mask_prob": mask_prob,
                                    "min_span": min_span,
                                    "max_span": max_span,
                                },
                                "text_masking_opts": {
                                    "mask_prob": mask_prob,
                                    "min_span": min_span,
                                    "max_span": max_span,
                                },
                            },
                        },
                    ),
                    f"baseline_audio-and-text-mask-p-{mask_prob}-span-{min_span}-{max_span}",
                )
                for mask_prob, min_span, max_span in (
                    (0.1, 1, 1),
                    (0.2, 1, 1),
                    (0.3, 1, 1),
                )
            ]
        ]
    ):
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            # skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            recog_variants=[_text_recon_variant(config, num_epochs)],
        )

    for exp_idx, (config, train_name) in enumerate(
        [
            *[
                (
                    dict_update_deep(
                        copy.deepcopy(codebook_config),
                        {
                            "train_args": {
                                "audio_masking_opts": {
                                    "mask_prob": mask_prob,
                                    "min_span": min_span,
                                    "max_span": max_span,
                                },
                                "text_masking_opts": {
                                    "mask_prob": mask_prob,
                                    "min_span": min_span,
                                    "max_span": max_span,
                                },
                            },
                        },
                    ),
                    f"baseline_codebook_audio-and-text-mask-p-{mask_prob}-span-{min_span}-{max_span}",
                )
                for mask_prob, min_span, max_span in (
                    (0.1, 1, 1),
                    # (0.2, 1, 1),
                    # (0.3, 1, 1),
                )
            ]
        ]
    ):
        num_epochs = config["training"]["__num_epochs"]
        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=copy.deepcopy(config),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(num_epochs),
            # skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            recog_variants=[
                {
                    "recog_name": "recon_text",
                    "input_modality": "text",
                    "output_modality": "text",
                    "mask_input": True,
                    "masking_opts": copy.deepcopy(config["train_args"]["text_masking_opts"]),
                    "keep_epochs": get_keep_epochs(base_num_epochs),
                },
                # fixed-masking text-recon sweep (copy ceiling + degradation curve), for a fair
                # single-task (text-only) vs multi-task comparison of the text denoiser.
                *_text_recon_sweep(base_num_epochs),
            ],
            # recog_variants=[_text_recon_variant(config, get_keep_epochs(num_epochs))],
        )
