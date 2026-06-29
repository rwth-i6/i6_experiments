import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.common.setups.serialization import PartialImport

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_training_datasets, build_text_only_training_datasets, build_test_datasets
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


def _text_recon_variant(config, num_epochs):
    """text->text reconstruction recog on the last epoch, masking the input with the experiment's
    own training text masking settings (scored against the unmasked phoneme reference)."""
    return {
        "recog_name": "recon_text",
        "input_modality": "text",
        "output_modality": "text",
        "mask_input": True,
        "masking_opts": copy.deepcopy(config["train_args"]["text_masking_opts"]),
        "keep_epochs": [num_epochs],
    }


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

    # single-task reference: text denoising only (no audio task, no alternate batching). Same shared
    # model and same text data/masking as the multi-task baseline, so comparing its recon_text WER
    # against the multi-task experiments shows how much the joint text+audio setup hurts text
    # reconstruction.
    text_only_train_data = build_text_only_training_datasets(sil_prob=0.0, surround_w_sil=False, settings=settings)
    text_only_config = dict_update_deep(
        copy.deepcopy(base_config),
        {
            "__train_step_module": "train_steps.aed_denoising_discrete_text_only.train_step",
            "general.default_data_key": "phon_indices",
            "general.default_target_key": "phon_indices",
            "training.accum_grad_multiple_step": 1,  # no alternate batching -> no grad accumulation
        },
        [
            # drop alternate batching and all audio-task settings
            "training.torch_batching",
            "train_args.audio_ce_loss_scale",
            "train_args.audio_masked_ce_loss_scale",
            "train_args.audio_masking_opts",
        ],
    )
    run_experiment(
        training_name=f"{prefix_name}/baseline_text-only",
        config=copy.deepcopy(text_only_config),
        train_data=text_only_train_data,
        test_data_dict=test_data_dict,
        keep_epochs=get_keep_epochs(base_num_epochs),
        skip_eval=True,  # audio->text ASR is not meaningful for a text-only model
        recog_variants=[_text_recon_variant(text_only_config, base_num_epochs)],
    )

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
            skip_eval=True,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
            analysis_opts={
                "checkpoints": get_keep_epochs(base_num_epochs),
                "max_plotted_seqs": 20,
                "cosine_similarity_summary": True,
            },
            recog_variants=[_text_recon_variant(config, num_epochs)],
        )
