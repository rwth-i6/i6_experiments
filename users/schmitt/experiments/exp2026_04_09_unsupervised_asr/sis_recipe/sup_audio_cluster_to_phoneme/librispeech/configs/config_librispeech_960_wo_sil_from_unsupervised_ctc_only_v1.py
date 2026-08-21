import copy
from typing import List, Dict

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, base_num_epochs
from .config_librispeech_960_wo_sil_v1 import train_data, test_data_dict_wo_sil


def get_keep_epochs(num_epochs: int) -> List[int]:
    if num_epochs == 1_000:
        return [5, 10, 20, 40, 60, 100, 150, 200, 250, 500, 750, 1_000]

    raise ValueError("Unsupported num_epochs")


base_config_ = dict_update_deep(
    copy.deepcopy(base_config),
    {
        "__network_module": "definitions.conformer_ctc_discrete_shared_v1.Model",
        "__train_step_module": "train_steps.aed_denoising_discrete.train_step",
        "__forward_step_module": "recognition.discrete_audio_ctc.forward_step.forward_step",
        "train_args": {"ce_loss_scale": 0.0, "masked_ce_loss_scale": 0.0, "aux_loss_scales": (1.0,)},
        "model_args.text_aux_loss_layers": (3,),
    },
)


def py(checkpoints: Dict):
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    for low_lr, peak_lr, freeze_emb_layers in [
        (1e-4, 1e-4, True),
        (1e-4, 1e-4, False),
    ]:
        run_experiment(
            training_name=f"{prefix_name}/baseline_low-lr-{low_lr}_peak-lr-{peak_lr}_freeze-enc{'_freeze-emb' if freeze_emb_layers else ''}",
            config=dict_update_deep(
                copy.deepcopy(base_config_),
                {
                    "training.preload_from_files": {
                        "pretrained_model": {
                            "ignore_missing": True,  # CTC layer is missing in checkpoint
                            "init_for_train": True,
                            "checkpoint_key": "model",
                            "filename": checkpoints[
                                "unsup_denoising_audio_cluster_and_phoneme/librispeech/config_librispeech_960_w_sil_in_input_v1/baseline_gan-adv-0.1_disc-lstm_mask-p-0.1-span-1-1_max-num-sil-7_max-surround-1"
                            ][500],
                        }
                    },
                    "training.__lr_opts": {
                        "piecewise_epochs": [
                            0,
                            0.45 * base_num_epochs,
                            0.9 * base_num_epochs,
                            base_num_epochs,
                        ],
                        "piecewise_values": [low_lr, peak_lr, 1e-5, 1e-6],
                    },
                    "model_args.freeze_params_list": [
                        rf"encoder\.",
                        *([r"text_embedding", r"audio_embedding"] if freeze_emb_layers else []),
                    ],
                },
            ),
            train_data=train_data,
            test_data_dict=test_data_dict_wo_sil,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            # score_data_dict=test_data_dict_wo_sil,
        )
