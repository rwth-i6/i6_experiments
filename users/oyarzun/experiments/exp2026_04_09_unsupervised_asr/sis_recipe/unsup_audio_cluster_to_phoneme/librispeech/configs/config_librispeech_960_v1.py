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

from sisyphus import tk


def get_keep_epochs(num_epochs: int) -> List[int]:
    if num_epochs == 1_000:
        return [250, 500, 750, 1_000]


base_num_epochs = 1_000
num_gpus = 2

settings = DatasetSettings(
    train_partition_epoch=20,
    train_seq_ordering="laplace:.1000",
)
train_data = build_training_datasets(settings=settings)
test_data_dict = build_test_datasets()


alternate_batching = PartialImport(
    code_object_path="i6_experiments.users.schmitt.returnn.alternate_batching.alternate_batching",
    import_as="alternate_batching",
    hashed_arguments={},
    unhashed_arguments={},
    unhashed_package_root=None,
)


base_config = {
    "__network_module": "definitions.conformer_aed_discrete_shared_v1.Model",
    "__train_step_module": "train_steps.aed_denoising_discrete_shared_backtranslation.train_step",
    "__baseline_alias": "v1",
    "__forward_step_module": "recognition.discrete_audio_aed.forward_step.forward_step",
    "__callback_module": "recognition.discrete_audio_aed.callback.RecognitionToTextDictCallback",
    "train_rqmt": {
        "cpu_rqmt": 6,
        "gpu_mem": 11,
    },
    "general": {
        "torch_dataloader_opts": {"num_workers": 1},
        "behavior_version": 25,
        "default_data_key": "data",
        "default_target_key": "target",
    },
    "training": {
        "__num_gpus": num_gpus,
        "__num_epochs": base_num_epochs,
        "__lr_opts": {
            "type": "dyn_lr_piecewise_linear",
            "piecewise_epochs": [0, 0.45 * base_num_epochs, 0.9 * base_num_epochs, base_num_epochs],
            "piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
        },
        "grad_scaler": {},
        "batch_size": 7500,
        **optimizer_configs.v1,
        "max_seqs": 200,
        "accum_grad_multiple_step": 2,  # alternate batching requires this
        "gradient_clip_global_norm": 5.0,
        "torch_batching": CodeWrapper("alternate_batching"),
    },
    "recog": {
        "batch_size": 7500,
    },
    "model_args": {
        "text_aux_loss_layers": (),
        "audio_aux_loss_layers": (),
        "num_enc_layers": 3,
        "num_text_dec_layers": 3,
        "num_audio_dec_layers": 3,
        "num_heads": 8,
        "model_dim": 512,
        "share_decoder": True,
        "text_out_dim": train_data.datastreams["target"].vocab_size,
        "audio_out_dim": train_data.datastreams["data"].vocab_size,
    },
    "train_args": {
        "aux_loss_scales": (),
        "text_ce_loss_scale": 0.2,
        "text_masked_ce_loss_scale": 1.0,
        "audio_ce_loss_scale": 0.2,
        "audio_masked_ce_loss_scale": 1.0,
        "pseudo_audio_text_ce_loss_scale": 1.0,
        "pseudo_text_audio_ce_loss_scale": 1.0,
        "text_masking_opts": {
            "mask_prob": 0.3,
            "min_span": 2,
            "max_span": 10,
        },
        "audio_masking_opts": {
            "mask_prob": 0.3,
            "min_span": 4,
            "max_span": 20,
        },
        "codebook_diversity_loss_scale": 0.0,
    },
}


def py():
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    ablations = [
        (
            "baseline_enc-3_dec-3",
            {
                "num_enc_layers": 3,
                "num_text_dec_layers": 3,
                "num_audio_dec_layers": 3,
            },
            {},
        ),
        (
            "baseline_codebook_enc-3_dec-3",
            {
                "num_enc_layers": 3,
                "num_text_dec_layers": 3,
                "num_audio_dec_layers": 3,
                "codebook_opts": {},
            },
            {
                "codebook_diversity_loss_scale": 0.1,
            },
        ),
        (
            "baseline_enc-6_dec-6",
            {
                "num_enc_layers": 6,
                "num_text_dec_layers": 6,
                "num_audio_dec_layers": 6,
            },
            {},
        ),
        (
            "baseline_codebook_enc-6_dec-6",
            {
                "num_enc_layers": 6,
                "num_text_dec_layers": 6,
                "num_audio_dec_layers": 6,
                "codebook_opts": {},
            },
            {
                "codebook_diversity_loss_scale": 0.1,
            },
        ),
        (
            "masking_text-p-0.4",
            {},
            {
                "text_masking_opts": {
                    "mask_prob": 0.4,
                    "min_span": 2,
                    "max_span": 10,
                },
            },
        ),
        (
            "masking_audio-p-0.4",
            {},
            {
                "audio_masking_opts": {
                    "mask_prob": 0.4,
                    "min_span": 4,
                    "max_span": 20,
                },
            },
        ),
    ]

    for train_name, model_args, train_args in ablations:
        config = copy.deepcopy(base_config)
        config["model_args"].update(model_args)
        config["train_args"].update(train_args)

        run_experiment(
            training_name=f"{prefix_name}/{train_name}",
            config=config,
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            additional_configs=[ReturnnConfig(config={}, python_prolog=[Collection([alternate_batching])])],
        )
