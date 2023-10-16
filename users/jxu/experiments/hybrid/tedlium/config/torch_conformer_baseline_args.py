import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection

from i6_experiments.users.jxu.experiments.hybrid.tedlium.default_tools import PACKAGE


def get_pytorch_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
    evaluation_epochs = list(range(num_epochs, num_epochs + 1, 10))

    batch_size = {"classes": 4 * 2000, "data": 4 * 320000}
    chunking = ({"classes": 100, "data": 100 * 160}, {"classes": 50, "data": 50 * 160})
    returnn_configs = get_pytorch_returnn_configs(
        num_epochs=num_epochs, num_inputs=80, num_outputs=num_outputs, batch_size=batch_size, chunking=chunking,
        evaluation_epochs=evaluation_epochs, debug=debug,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_epochs=num_epochs, num_inputs=80, num_outputs=num_outputs, batch_size=batch_size, chunking=chunking,
        evaluation_epochs=evaluation_epochs,
        recognition=True, debug=debug,
    )

    training_args = {
        "log_verbosity": 5,
        "num_epochs": num_epochs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 8,
        "cpu_rqmt": 3,
    }

    recognition_args = {
        "dev-other": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.6, 0.7],
            "pronunciation_scales": [6.0],
            "lm_scales": [20.0, 15.0, 10.3],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 16.0,
                "beam-pruning-limit": 100000,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 15000,
            },
            "lattice_to_ctm_kwargs": {
                "fill_empty_segments": True,
                "best_path_algo": "bellman-ford",
            },
            "optimize_am_lm_scale": True,
            "rtf": 50,
            "mem": 7,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": False,
        },
    }
    test_recognition_args = None

    nn_args = HybridArgs(
        returnn_training_configs=returnn_configs,
        returnn_recognition_configs=returnn_recog_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args

def dynamic_learning_rate(*, global_train_step, learning_rate, **kwargs):
    lr_peak = learning_rate
    total_num_updates = 2400*260
    lr_peak = lr_peak
    lr_1 = lr_2 = lr_peak / 10
    lr_final = 1e-7
    num_P1 = total_num_updates*0.45
    num_P2 = total_num_updates * 0.45
    num_p3 = total_num_updates * 0.1
    if global_train_step <= num_P1:
        factor = global_train_step / num_P1
        lr = lr_1 + factor * (lr_peak-lr_1)
    elif num_P1 < global_train_step <= (num_P2+num_P1):
        factor = (global_train_step-num_P1) / num_P2
        lr = lr_peak - factor * (lr_peak - lr_2)
    elif (num_P2+num_P1) < global_train_step <= total_num_updates:
        factor = (global_train_step - (num_P1+num_P2)) / num_p3
        lr = lr_2 - factor * (lr_2 - lr_final)
    else:
        lr = lr_final
    return lr


def get_pytorch_returnn_configs(
        num_epochs: int, num_inputs: int, num_outputs: int, batch_size: int, chunking: str,
        evaluation_epochs: List[int],
        recognition=False, debug=False,
):
    # ******************** blstm base ********************

    base_config = {
        "extern_data": {
            'data': {"dim": num_inputs, "shape": (None, num_inputs), "available_for_inference": True},
            # input: 80-dimensional logmel features
            'classes': {"dim": num_outputs, "shape": (None,), "available_for_inference": True, "sparse": True,
                        "dtype": "int16"}
        }
    }
    base_post_config = {
        "backend": "torch",
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
        "tf_log_memory_usage": True,
        "cache_size": "0",

    }
    blstm_base_config = copy.deepcopy(base_config)
    blstm_base_config.update(
        {
            "behavior_version": 15,
            "batch_size": batch_size,  # {"classes": batch_size, "data": batch_size},
            "chunking": "50:25",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "gradient_noise": 0.3,
            "learning_rates": list(np.linspace(2.5e-5, 3e-4, 50)) + list(np.linspace(3e-4, 2.5e-5, 50)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            # "min_learning_rate": 1e-5,
            "min_seq_length": {"classes": 1},
            "newbob_learning_rate_decay": 0.707,
            "newbob_multi_num_epochs": 40,
            "newbob_multi_update_interval": 1,
        }
    )
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }

    # those are hashed
    # package = PACKAGE + ".hybrid"
    package = PACKAGE
    pytorch_package = package + ".pytorch_networks"

    def construct_from_net_kwargs(base_config, net_kwargs, dynamic_peak_lr=None, explicit_hash=None):
        model_type = net_kwargs.pop("model_type")
        pytorch_model_import = Import(
            package + ".pytorch_networks.%s.Model" % model_type
        )
        pytorch_train_step = Import(
            package + ".pytorch_networks.%s.train_step" % model_type
        )
        pytorch_model = PyTorchModel(
            model_class_name=pytorch_model_import.object_name,
            model_kwargs=net_kwargs,
        )
        serializer_objects = [
            pytorch_model_import,
            pytorch_train_step,
            pytorch_model,
        ]
        if recognition:
            pytorch_export = Import(
                package + ".pytorch_networks.%s.export" % model_type
            )
            serializer_objects.append(pytorch_export)
        i6_models_repo = CloneGitRepositoryJob(
            url="https://github.com/rwth-i6/i6_models",
            commit="48ade422412bf73e1835a36ee72f12af73106cc6",
            checkout_folder_name="i6_models",
        ).out_repository
        i6_models_repo.hash_overwrite = "TEDLIUM2_DEFAULT_I6_MODELS"
        i6_models = ExternalImport(import_path=i6_models_repo)
        serializer_objects.insert(0, i6_models)
        if explicit_hash:
            serializer_objects.append(ExplicitHash(explicit_hash))
        serializer = Collection(
            serializer_objects=serializer_objects,
            make_local_package_copy=not debug,
            packages={
                pytorch_package,
            },
        )
        if dynamic_peak_lr is None:
            returnn_config = ReturnnConfig(
                config=base_config,
                post_config=base_post_config,
                python_epilog=[serializer],
                pprint_kwargs={"sort_dicts": False},
            )
        else:
            base_config["learning_rate"] = dynamic_peak_lr
            dynamic_learning_rate.func_defaults = (dynamic_peak_lr,)
            returnn_config = ReturnnConfig(
                config=base_config,
                post_config=base_post_config,
                python_prolog = [dynamic_learning_rate],
                python_epilog=[serializer],
                pprint_kwargs={"sort_dicts": False},
            )
        return returnn_config

    conformer_peak_lr_8e_04_config = copy.deepcopy(blstm_base_config)
    conformer_peak_lr_8e_04_config["batch_size"] = 14000
    conformer_peak_lr_8e_04_config["chunking"] = "400:200"
    peak_lr = 8e-4
    learning_rates = list(np.linspace(peak_lr / 10, peak_lr, 50)) + list(np.linspace(peak_lr, peak_lr / 10, 50)) + list(
        np.linspace(peak_lr / 10, 1e-8, 25))
    conformer_peak_lr_8e_04_config["learning_rates"] = learning_rates

    conformer_peak_lr_8e_04_with_warmup_config = copy.deepcopy(conformer_peak_lr_8e_04_config)
    learning_rates = list(np.linspace(1e-7, peak_lr / 10, 25)) + list(np.linspace(peak_lr / 10, peak_lr, 50)) + list(
        np.linspace(peak_lr, peak_lr / 10, 50)) + list(
        np.linspace(peak_lr / 10, 1e-8, 25))
    conformer_peak_lr_8e_04_with_warmup_config["learning_rates"] = learning_rates

    peak_lr = 3e-4
    learning_rates = list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr / 10, 1e-8, 30))
    conformer_peak_lr_3e_04_config = copy.deepcopy(conformer_peak_lr_8e_04_config)
    conformer_peak_lr_3e_04_config["learning_rates"] = learning_rates

    peak_lr = 5e-4
    learning_rates = list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr / 10, 1e-8, 30))
    conformer_peak_lr_5e_04_config = copy.deepcopy(conformer_peak_lr_8e_04_config)
    conformer_peak_lr_5e_04_config["learning_rates"] = learning_rates

    peak_lr = 6e-4
    learning_rates = list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr / 10, 1e-8, 30))
    conformer_peak_lr_6e_04_config = copy.deepcopy(conformer_peak_lr_8e_04_config)
    conformer_peak_lr_6e_04_config["learning_rates"] = learning_rates

    peak_lr = 7e-4
    learning_rates = list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr / 10, 1e-8, 30))
    conformer_peak_lr_7e_04_config = copy.deepcopy(conformer_peak_lr_8e_04_config)
    conformer_peak_lr_7e_04_config["learning_rates"] = learning_rates

    peak_lr = 1e-3
    learning_rates = list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2)) + list(
        np.linspace(peak_lr / 10, 1e-8, 30))
    conformer_peak_lr_1e_03_config = copy.deepcopy(conformer_peak_lr_8e_04_config)
    conformer_peak_lr_1e_03_config["learning_rates"] = learning_rates

    dynamic_lr_12e_03_config = copy.deepcopy(conformer_peak_lr_7e_04_config)
    dynamic_lr_12e_03_config["learning_rates"] = [1.2e-3] * num_epochs

    dynamic_lr_13e_03_config = copy.deepcopy(conformer_peak_lr_7e_04_config)
    dynamic_lr_13e_03_config["learning_rates"] = [1.3e-3] * num_epochs



    return {
        "conformer_peak_lr_1e_03_specaug_20_5_epochs_{}".format(num_epochs): construct_from_net_kwargs(
            conformer_peak_lr_1e_03_config,
            {"model_type": "i6_conformer_baseline",
             "model_size": 384,
             "num_layers": 12,
             "kernel_size": 31,
             "time_min_num_masks": 2,
             "time_max_mask_per_n_frames": 25,
             "time_mask_max_size": 20,
             "freq_min_num_masks": 2,
             "freq_max_num_masks": 5,
             "freq_mask_max_size": 8, },)
    }
