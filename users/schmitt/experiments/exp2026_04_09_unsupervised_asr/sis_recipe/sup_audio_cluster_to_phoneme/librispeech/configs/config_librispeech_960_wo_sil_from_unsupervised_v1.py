import copy
from typing import List, Dict

from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ....train_exp import run_experiment
from ....data.common import DatasetSettings
from .... import optimizer_configs
from ... import __setup_base_name__

from .config_librispeech_960_v1 import base_config, get_keep_epochs, base_num_epochs
from .config_librispeech_960_wo_sil_v1 import train_data, test_data_dict_wo_sil, test_data_dict


def py(checkpoints: Dict):
    prefix_name = f"{__setup_base_name__}/librispeech/{__name__.split('.')[-1]}"

    for low_lr, peak_lr, add_eps in [
        (1e-5, 1e-3, []),
        (1e-4, 1e-4, [5]),
        (5e-5, 5e-5, []),
    ]:
        run_experiment(
            training_name=f"{prefix_name}/baseline_low-lr-{low_lr}_peak-lr-{peak_lr}_freeze-enc",
            config=dict_update_deep(
                copy.deepcopy(base_config),
                {
                    "training.preload_from_files": {
                        "pretrained_model": {
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
                    "model_args.freeze_params_list": [rf"encoder\."],
                },
            ),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            # conditional (audio->phoneme) perplexity of the AED model on the last checkpoint, scored on
            # the wo-silence reference (matching this wo-sil model) via a separate PPL dataset; recognition
            # keeps the with-silence test_data_dict. Both expose audio + text (paired MetaDataset).
            ppl_opts={
                "checkpoints": [base_num_epochs],
                "input_modality": "audio",
                "test_data_dict": test_data_dict_wo_sil,
            },
            cross_att_opts={
                "checkpoints": get_keep_epochs(base_num_epochs) + add_eps,
                "input_modality": "audio",
                "output_modality": "text",
                "max_plotted_seqs": 20,
            },
            # score_data_dict=test_data_dict_wo_sil,
        )

    for freeze_dec_layers in [(0,), (0, 1), (0, 1, 2)]:
        run_experiment(
            training_name=f"{prefix_name}/baseline_low-lr-{1e-4}_peak-lr-{1e-4}_freeze-enc_freeze-dec-{freeze_dec_layers}",
            config=dict_update_deep(
                copy.deepcopy(base_config),
                {
                    "training.preload_from_files": {
                        "pretrained_model": {
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
                        *[rf"decoder\.module_list\.{i}\." for i in freeze_dec_layers],
                    ],
                },
            ),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            # conditional (audio->phoneme) perplexity of the AED model on the last checkpoint, scored on
            # the wo-silence reference (matching this wo-sil model) via a separate PPL dataset; recognition
            # keeps the with-silence test_data_dict. Both expose audio + text (paired MetaDataset).
            ppl_opts={
                "checkpoints": [base_num_epochs],
                "input_modality": "audio",
                "test_data_dict": test_data_dict_wo_sil,
            },
            cross_att_opts={
                "checkpoints": get_keep_epochs(base_num_epochs) + [5, 20, 50, 100, 150, 200],
                "input_modality": "audio",
                "output_modality": "text",
                "max_plotted_seqs": 20,
                "test_data_dict": test_data_dict_wo_sil,
            },
            score_data_dict=test_data_dict_wo_sil,
        )

    for freeze_emb_layers in [(0,), (1,), (2,), (1, 2), (0, 1, 2)]:
        run_experiment(
            training_name=f"{prefix_name}/baseline_low-lr-{1e-4}_peak-lr-{1e-4}_freeze-enc_freeze-dec_freeze-emb-{freeze_emb_layers}",
            config=dict_update_deep(
                copy.deepcopy(base_config),
                {
                    "training.preload_from_files": {
                        "pretrained_model": {
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
                        *[rf"decoder\.module_list\.{i}\." for i in range(3)],
                        *([r"text_embedding", r"audio_embedding"] if 0 in freeze_emb_layers else []),
                        *([r"decoder\.input_embedding"] if 1 in freeze_emb_layers else []),
                        *([r"decoder\.out_logits"] if 2 in freeze_emb_layers else []),
                    ],
                },
            ),
            train_data=train_data,
            test_data_dict=test_data_dict,
            keep_epochs=get_keep_epochs(base_num_epochs),
            skip_eval=False,
            # conditional (audio->phoneme) perplexity of the AED model on the last checkpoint, scored on
            # the wo-silence reference (matching this wo-sil model) via a separate PPL dataset; recognition
            # keeps the with-silence test_data_dict. Both expose audio + text (paired MetaDataset).
            ppl_opts={
                "checkpoints": [base_num_epochs],
                "input_modality": "audio",
                "test_data_dict": test_data_dict_wo_sil,
            },
            cross_att_opts={
                "checkpoints": get_keep_epochs(base_num_epochs) + [5, 20, 50, 100, 150, 200],
                "input_modality": "audio",
                "output_modality": "text",
                "max_plotted_seqs": 20,
                "test_data_dict": test_data_dict_wo_sil,
            },
            score_data_dict=test_data_dict_wo_sil,
        )
