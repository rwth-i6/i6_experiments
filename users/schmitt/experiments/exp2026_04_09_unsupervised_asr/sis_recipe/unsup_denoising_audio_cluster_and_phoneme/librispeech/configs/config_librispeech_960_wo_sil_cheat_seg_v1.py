import copy
from typing import List
import functools

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep
from i6_experiments.common.setups.serialization import PartialImport

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_core.serialization import Collection

from ....train_exp import run_experiment
from ..data.common import build_training_datasets_w_cheating_clusters, build_test_datasets_w_cheating_clusters
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from ....sup_audio_cluster_to_phoneme.librispeech.configs.config_librispeech_960_v1 import (
    base_config as base_config_,
)

from .config_librispeech_960_wo_sil_v1 import (
    _text_recon_sweep,
    _text_recon_variant,
    _train_reflecting_analysis_masking,
    _recon_variant,
)


def get_keep_epochs(num_epochs: int) -> List[int]:
    if num_epochs == 100:
        return [1, 2, 3, 4, 5, 10, 20, 25, 50, 75, 100]

    raise ValueError("Unsupported num_epochs")


base_num_epochs = 100
settings = DatasetSettings(
    train_partition_epoch=1,
    train_seq_ordering="laplace:.1000",
    num_workers=1,
    buffer_size=10,
)


train_data = build_training_datasets_w_cheating_clusters(sil_prob=0.0, surround_w_sil=False, settings=settings)
test_data_dict_wo_sil = build_test_datasets_w_cheating_clusters(sil_prob=0.0, surround_w_sil=False)


base_config = dict_update_deep(
    base_config_,
    {
        "__train_step_module": "train_steps.aed_denoising_discrete_shared.train_step",
        "training": {
            "torch_batching": CodeWrapper("alternate_batching"),
            "accum_grad_multiple_step": 2,  # alternate batching
            "__num_epochs": base_num_epochs,
            "__num_gpus": 1,
            "batch_size": 30_000,
        },
        "train_rqmt": {"gpu_mem": 24},
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
                "min_span": 2,  # 1
                "max_span": 10,  # 3
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


# fixed-masking text-recon sweep (span 2-10) at a common set of mask probs, so the text denoiser can
# be characterized (copy ceiling at 0.0 + degradation curve) and compared fairly across experiments.
# 0.3 is already covered by _text_recon_variant (the base training masking), so it is not repeated.
_TEXT_RECON_SWEEP_MASK_PROBS = (0.0, 0.1, 0.5)


run_experiment = functools.partial(
    run_experiment,
    train_data=train_data,
    test_data_dict=test_data_dict_wo_sil,
    keep_epochs=get_keep_epochs(base_num_epochs),
    additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
    analysis_opts={
        "checkpoints": get_keep_epochs(base_num_epochs),
        "max_plotted_seqs": 20,
        "cosine_similarity_summary": True,
    },
    # decoder cross-attention weights (audio in / phoneme out, i.e. the ASR direction) on the
    # last checkpoint, for the same dev-other seqs the encoder-PCA analysis plots.
    cross_att_opts={
        "checkpoints": get_keep_epochs(base_num_epochs),
        "input_modality": "audio",
        "output_modality": "text",
        "max_plotted_seqs": 20,
    },
    # conditional (audio->phoneme) perplexity of the shared AED model on the last checkpoint,
    # scored on the wo-silence reference (matching the wo-sil model) via a separate PPL dataset;
    # recognition / analysis keep the with-silence test_data_dict.
    ppl_opts={
        "checkpoints": get_keep_epochs(base_num_epochs),
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
            "keep_epochs": get_keep_epochs(base_num_epochs),
        },
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


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    run_experiment(
        training_name=f"{prefix_name}/baseline",
        config=copy.deepcopy(base_config),
    )

    for num_layers, model_dim in ((3, 256), (3, 128)):
        run_experiment(
            training_name=f"{prefix_name}/baseline_fix-dec-text_dec-share-emb_{num_layers}L-{model_dim}D",
            config=dict_update_deep(
                copy.deepcopy(base_config),
                {
                    "model_args": {
                        "num_enc_layers": num_layers,
                        "num_text_dec_layers": num_layers,
                        "num_audio_dec_layers": num_layers,
                        "num_heads": 8,
                        "model_dim": model_dim,
                        "fix_decode_text_seq_for_shared_dec": True,
                        "dec_share_emb": True,
                    },
                },
            ),
        )

    for mask_prob, codebook_prob in ((0.3, 1.0), (0.2, 1.0), (0.1, 1.0), (0.3, 0.5), (0.3, 0.2)):
        codebook_config = dict_update_deep(
            copy.deepcopy(base_config),
            {
                "model_args": {
                    "codebook_opts": {"latent_groups": 1, "codebook_prob": codebook_prob}
                },  # {} -> enable codebook with default settings
                "train_args": {"codebook_diversity_loss_scale": 0.1},
            },
        )

        run_experiment(
            training_name=f"{prefix_name}/baseline_codebook_mask-prob-{mask_prob}_code-prob-{codebook_prob}",
            config=dict_update_deep(
                copy.deepcopy(codebook_config),
                {
                    "train_args.text_masking_opts.mask_prob": mask_prob,
                    "train_args.audio_masking_opts.mask_prob": mask_prob,
                },
            ),
        )

        run_experiment(
            training_name=f"{prefix_name}/baseline_lstm-gan",
            config=dict_update_deep(
                copy.deepcopy(base_config),
                {
                    "model_args.discriminator_type": "lstm",
                    "train_args": {
                        "adv_loss_scale": 0.1,
                    },
                },
            ),
        )
