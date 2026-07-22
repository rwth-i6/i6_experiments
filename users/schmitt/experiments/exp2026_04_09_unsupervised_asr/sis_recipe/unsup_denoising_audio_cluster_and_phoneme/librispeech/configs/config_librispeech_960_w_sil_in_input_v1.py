import copy
from typing import List

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.common.setups.serialization import PartialImport

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_test_datasets, build_training_datasets_w_silence_in_input
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
train_data = build_training_datasets_w_silence_in_input(sil_prob=0.25, surround_w_sil=True, settings=settings)
test_data_dict_wo_sil = build_test_datasets(sil_prob=0.0, surround_w_sil=False)


base_config = dict_update_deep(
    base_config_,
    {
        "__train_step_module": "train_steps.aed_denoising_discrete_shared.train_step",
        "training": {
            "torch_batching": CodeWrapper("alternate_batching"),
            "accum_grad_multiple_step": 2,  # alternate batching
            # "__num_gpus": 1,  # for debug
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
            "fix_decode_text_seq_for_shared_dec": True,
        },
        "train_args": {
            "aux_loss_scales": (),
            "text_ce_loss_scale": 1.0,
            # since input includes silence now while the output does not
            # we cannot easily calculate the loss only on the masked positions of the output
            # since the mask might also be over silence frames in the input
            "text_masked_ce_loss_scale": 0.0,
            "audio_ce_loss_scale": 0.2,
            "audio_masked_ce_loss_scale": 1.0,
            "text_masking_opts": {
                "mask_prob": 0.1,
                "min_span": 1,  # 1
                "max_span": 1,  # 3
            },
            "audio_masking_opts": {
                "mask_prob": 0.1,
                "min_span": 1,  # 1
                "max_span": 1,  # 3
            },
        },
    },
    [
        "train_args.masking_opts",
        "train_args.ce_loss_scale",
        "train_args.masked_ce_loss_scale",
        "general.torch_dataloader_opts",
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
        test_data_dict=test_data_dict_wo_sil,
        keep_epochs=get_keep_epochs(base_num_epochs),
        # skip_eval=True,
        additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        analysis_opts={
            "checkpoints": get_keep_epochs(base_num_epochs),
            "max_plotted_seqs": 20,
            "cosine_similarity_summary": True,
        },
        skip_eval=True,
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

    for max_num_sil in [3, 5, 7]:
        train_data_var_sil = build_training_datasets_w_silence_in_input(
            sil_prob=0.25, surround_w_sil=True, settings=settings, max_num_sil=max_num_sil
        )
        # discriminator-architecture sweep for the domain-adversarial loss. Same adv scale + masking as
        # baseline_gan-adv-0.1_mask-p-0.1-span-1-1 (which uses the frame-wise "mlp" discriminator), but
        # with discriminators that see more temporal context:
        #   mlp_2gram/3gram/4gram -> MLP over 2/3/4 consecutive frames concatenated in the feature dim
        #   lstm                  -> LSTM over the whole encoder output sequence
        for discriminator_type in ("lstm",):
            run_experiment(
                training_name=f"{prefix_name}/baseline_gan-adv-0.1_disc-{discriminator_type}_mask-p-0.1-span-1-1_max-num-sil-{max_num_sil}",
                config=dict_update_deep(
                    copy.deepcopy(base_config),
                    {
                        "model_args.discriminator_type": discriminator_type,
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
                train_data=train_data_var_sil,
                test_data_dict=test_data_dict_wo_sil,
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
