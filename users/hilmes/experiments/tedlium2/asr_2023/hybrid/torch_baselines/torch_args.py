import copy
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.hilmes.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection


# from i6_experiments.common.baselines.tedlium2.default_tools import PACKAGE
from onnxruntime.quantization import CalibrationMethod

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
    evaluation_epochs = list(range(num_epochs, num_epochs + 1, 10))

    batch_size = {"classes": 4 * 2000, "data": 4 * 320000}  # TODO change to 14000 probably more
    chunking = (
        {"classes": 400, "data": 100 * 160},
        {"classes": 50, "data": 50 * 160},
    )  # TODO change to 400:200/500:250
    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=chunking,
        evaluation_epochs=evaluation_epochs,
        debug=debug,
        num_epochs=num_epochs,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=chunking,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        debug=debug,
    )

    training_args = ReturnnTrainingJobArgs(
        log_verbosity=5, num_epochs=num_epochs, save_interval=1, keep_epochs=None, time_rqmt=168, mem_rqmt=8, cpu_rqmt=3
    )

    recognition_args = {
        "dev": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.5, 0.7, 0.8, 0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0, 7.5, 5.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 14.0,
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
            "needs_features_size": True,
        },
        "best-cp": {
            "epochs": ["best", "avrg"],
            "feature_flow_key": "fb",
            "prior_scales": [0.7, 0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0, 7.5],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 14.0,
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
            "needs_features_size": True,
            "training_whitelist": ["torch_jj_config2", "torch_jj_config2_large"],
        },
        # "quant": {
        #     "epochs": evaluation_epochs,
        #     "feature_flow_key": "fb",
        #     "prior_scales": [0.7, 0.9],
        #     "pronunciation_scales": [0.0],
        #     "lm_scales": [10.0, 7.5],
        #     "lm_lookahead": True,
        #     "lookahead_options": None,
        #     "create_lattice": True,
        #     "eval_single_best": True,
        #     "eval_best_in_lattice": True,
        #     "search_parameters": {
        #         "beam-pruning": 15.0,
        #         "beam-pruning-limit": 10000,
        #         "word-end-pruning": 0.5,
        #         "word-end-pruning-limit": 15000,
        #     },
        #     "lattice_to_ctm_kwargs": {
        #         "fill_empty_segments": True,
        #         "best_path_algo": "bellman-ford",
        #     },
        #     "optimize_am_lm_scale": True,
        #     "rtf": 50,
        #     "mem": 7,
        #     "lmgc_mem": 16,
        #     "cpu": 2,
        #     "parallelize_conversion": True,
        #     "needs_features_size": True,
        #     #"quantize": [10, 15, 25, 100, 250, 500, 750, 1000, 2500, 5000],
        #     "quantize": [5, 11, 10, 25, 100, 500, 1000, 5000, 10000, 50000, 100000],
        #     "quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        #     "quant_sym_modes": [False],
        #     "quant_avg_modes": [False],
        #     "quant_percentiles": [90.0, 95.0, 99.0, 99.999, 99.9],
        #     "quant_num_bin_ls": [512, 1024, 2048, 4096],
        #     "training_whitelist": [
        #         "torch_jj_config2",
        #     ],
        # },
        "quant-base": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.7],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 15.0,
                "beam-pruning-limit": 10000,
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
            "needs_features_size": True,
            "quantize": [5, 10, 25, 100, 500, 1000, 5000, 10000, 50000, 100000],
            "quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "quant_sym_modes": [False],
            "quant_avg_modes": [False],
            "quant_percentiles": [99.999],
            "quant_num_bin_ls": [2048],
            "training_whitelist": [
                "torch_jj_config2",
            ],
        },
        "quant-ops": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.7],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 15.0,
                "beam-pruning-limit": 10000,
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
            "needs_features_size": True,
            # "quantize": [10, 15, 25, 100, 250, 500, 750, 1000, 2500, 5000],
            "quantize": [10, 25, 100, 500],
            "quant_modes": [CalibrationMethod.MinMax],
            "quant_sym_modes": [False],
            "quant_avg_modes": [False],
            "quant_ops": [None, ["Conv"], ["MatMul"], ["Conv", "MatMul"], ["Conv", "MatMul", "Mul", "Add"]],
            "training_whitelist": [
                "torch_jj_config2",
            ],
        },
        "quant-multiple": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.7],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 15.0,
                "beam-pruning-limit": 10000,
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
            "needs_features_size": True,
            "quantize": [10, 500, 1000],
            "quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile, CalibrationMethod.Entropy],
            #"quant_modes": [CalibrationMethod.MinMax],
            "random_seed_draws": 100,
            "quant_sym_modes": [False],
            "quant_avg_modes": [False],
            "quant_percentiles": [99.999],
            "quant_num_bin_ls": [2048],
            "training_whitelist": [
                "torch_jj_config2",
                "torch_jj_seed_24",
                "torch_jj_seed_5",
                "torch_jj_seed_2005",
            ],
        },
        "quant-filter": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "fb",
            "prior_scales": [0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [5.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 15.0,
                "beam-pruning-limit": 10000,
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
            "needs_features_size": True,
            "quantize": [10],
            #"quant_filter_opts": [None, {"min_seq_len": 1000}, {"max_seq_len": 1000}],
            "quant_filter_opts": [{"max_seq_len": 1000}, {"max_seq_len": 1500},{"max_seq_len": 500},{"min_seq_len": 1000},{"min_seq_len": 1500}, {"min_seq_len": 500}],
            "quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
            "random_seed_draws": 100,
            "quant_sym_modes": [False],
            "quant_avg_modes": [False],
            "quant_percentiles": [99.999],
            "quant_num_bin_ls": [2048],
            "training_whitelist": [
                "torch_jj_config2",
            ],
        },
        # "quant-skip": {
        #     "epochs": evaluation_epochs,
        #     "feature_flow_key": "fb",
        #     "prior_scales": [0.7],
        #     "pronunciation_scales": [0.0],
        #     "lm_scales": [10.0],
        #     "lm_lookahead": True,
        #     "lookahead_options": None,
        #     "create_lattice": True,
        #     "eval_single_best": True,
        #     "eval_best_in_lattice": True,
        #     "search_parameters": {
        #         "beam-pruning": 15.0,
        #         "beam-pruning-limit": 10000,
        #         "word-end-pruning": 0.5,
        #         "word-end-pruning-limit": 15000,
        #     },
        #     "lattice_to_ctm_kwargs": {
        #         "fill_empty_segments": True,
        #         "best_path_algo": "bellman-ford",
        #     },
        #     "optimize_am_lm_scale": True,
        #     "rtf": 50,
        #     "mem": 7,
        #     "lmgc_mem": 16,
        #     "cpu": 2,
        #     "parallelize_conversion": True,
        #     "needs_features_size": True,
        #     # "quantize": [5, 10, 100],
        #     "quantize": [10],
        #     "quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        #     "quant_sym_modes": [False],
        #     "quant_avg_modes": [False],
        #     "quant_percentiles": [99.999],
        #     "quant_num_bin_ls": [2048],
        #     "final_skip_ls": [[1, 2, 3], [1, 5]],
        #     "training_whitelist": [
        #         "torch_jj_config2",
        #     ],
        # },
        # "quant-smooth": {
        #     "epochs": evaluation_epochs,
        #     "feature_flow_key": "fb",
        #     "prior_scales": [0.7],
        #     "pronunciation_scales": [0.0],
        #     "lm_scales": [10.0],
        #     "lm_lookahead": True,
        #     "lookahead_options": None,
        #     "create_lattice": True,
        #     "eval_single_best": True,
        #     "eval_best_in_lattice": True,
        #     "search_parameters": {
        #         "beam-pruning": 15.0,
        #         "beam-pruning-limit": 10000,
        #         "word-end-pruning": 0.5,
        #         "word-end-pruning-limit": 15000,
        #     },
        #     "lattice_to_ctm_kwargs": {
        #         "fill_empty_segments": True,
        #         "best_path_algo": "bellman-ford",
        #     },
        #     "optimize_am_lm_scale": True,
        #     "rtf": 50,
        #     "mem": 7,
        #     "lmgc_mem": 16,
        #     "cpu": 2,
        #     "parallelize_conversion": True,
        #     "needs_features_size": True,
        #     "quantize": [5, 10, 100],
        #     #"quant_modes": [CalibrationMethod.MinMax, CalibrationMethod.Percentile], # TODO implement smooth for percentile
        #     "quant_modes": [CalibrationMethod.MinMax],
        #     "quant_sym_modes": [False],
        #     "quant_avg_modes": [False],
        #     "quant_percentiles": [99.999],
        #     "quant_num_bin_ls": [2048],
        #     "smooth_ls": [-0.1, 0.05, 0.1, 0.3],
        #     "training_whitelist": [
        #         "torch_jj_config2",
        #     ],
        # },

    }
    # from .old_torch_recog_args import speed
    # recognition_args.update(speed)
    test_recognition_args = {
        "dev": {},
        "quant-base": {}
    }

    nn_args = HybridArgs(
        returnn_training_configs=returnn_configs,
        returnn_recognition_configs=returnn_recog_configs,
        training_args=training_args,
        recognition_args=recognition_args,
        test_recognition_args=test_recognition_args,
    )

    return nn_args


def get_pytorch_returnn_configs(
    num_inputs: int,
    num_outputs: int,
    batch_size: int,
    chunking: str,
    evaluation_epochs: List[int],
    recognition=False,
    debug=False,
    num_epochs=None,
):
    # ******************** blstm base ********************

    base_config = {
        "extern_data": {
            "data": {
                "dim": num_inputs,
                "shape": (None, num_inputs),
                "available_for_inference": True,
            },  # input: 80-dimensional logmel features
            "classes": {
                "dim": num_outputs,
                "shape": (None,),
                "available_for_inference": True,
                "sparse": True,
                "dtype": "int16",
            },
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

    medium_lchunk_config = copy.deepcopy(blstm_base_config)
    medium_lchunk_config["learning_rates"] = list(np.linspace(8e-5, 8e-4, 50)) + list(np.linspace(8e-4, 8e-5, 50))
    medium_lchunk_config["chunking"] = "250:200"
    medium_lchunk_config["gradient_clip"] = 1.0
    medium_lchunk_config["optimizer"] = {"class": "adam", "epsilon": 1e-8}

    chunk_400_200_config = copy.deepcopy(medium_lchunk_config)
    chunk_400_200_config["chunking"] = "400:200"

    jj_config = copy.deepcopy(chunk_400_200_config)
    jj_config["batch_size"] = 14000
    peak_lr = 7e-4
    if num_epochs is None:
        num_epochs = 250
    learning_rates = (
        list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr / 10, 1e-8, 30))
    )
    jj_config["learning_rates"] = learning_rates

    jj_larger_chunk = copy.deepcopy(jj_config)
    jj_larger_chunk["chunking"] = "500:250"

    chunk_500_250_config = copy.deepcopy(medium_lchunk_config)
    chunk_500_250_config["chunking"] = "500:250"

    batch_12k_config = copy.deepcopy(medium_lchunk_config)
    batch_12k_config["batch_size"] = 12000

    batch_10k_config = copy.deepcopy(medium_lchunk_config)
    batch_10k_config["batch_size"] = 10000

    batch_10k_chunk_400_config = copy.deepcopy(batch_10k_config)
    batch_10k_chunk_400_config["chunking"] = "400:200"

    batch_12k_chunk_400_config = copy.deepcopy(batch_12k_config)
    batch_12k_chunk_400_config["chunking"] = "400:200"

    batch_10k_chunk_500_config = copy.deepcopy(batch_10k_config)
    batch_10k_chunk_500_config["chunking"] = "500:250"

    wei_lstm_config = copy.deepcopy(base_config)

    wei_lstm_config.update(
        {
            "behavior_version": 15,
            "batch_size": 10000,  # {"classes": batch_size, "data": batch_size},
            "max_seqs": 128,
            "chunking": "64:32",
            "optimizer": {"class": "nadam", "epsilon": 1e-8},
            "learning_rate": 0.0009,
            "gradient_clip": 0,
            "gradient_noise": 0.1,  # together with l2 and dropout for overfit
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_relative_error_relative_lr": True,
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            # "min_learning_rate": 1e-5,
            "min_seq_length": {"classes": 1},
            "newbob_multi_num_epochs": 5,
            "newbob_multi_update_interval": 1,
            "newbob_learning_rate_decay": 0.9,
        }
    )

    # those are hashed
    PACKAGE = __package__
    package = PACKAGE
    pytorch_package = package + ".pytorch_networks"

    def construct_from_net_kwargs(
            base_config,
            net_kwargs,
            explicit_hash=None,
            models_commit="75e03f37ac74d3d0c7358d29bb9b71dcec1bf120",
            grad_acc=None,
            random_seed=None
    ):
        if grad_acc is not None:
            base_config = copy.deepcopy(base_config)
            base_config["accum_grad_multiple_step"] = grad_acc
        if random_seed is not None:
            base_config = copy.deepcopy(base_config)
            base_config["random_seed"] = random_seed
        model_type = net_kwargs.pop("model_type")
        pytorch_model_import = Import(package + ".pytorch_networks.%s.Model" % model_type)
        pytorch_train_step = Import(package + ".pytorch_networks.%s.train_step" % model_type)
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
            pytorch_export = Import(package + ".pytorch_networks.%s.export" % model_type)
            serializer_objects.append(pytorch_export)

            prior_computation = Import(package + ".pytorch_networks.prior.basic.forward_step")
            serializer_objects.append(prior_computation)
            prior_computation = Import(
                package + ".pytorch_networks.prior.prior_callback.ComputePriorCallback", import_as="forward_callback"
            )
            serializer_objects.append(prior_computation)

        i6_models_repo = CloneGitRepositoryJob(
            url="https://github.com/rwth-i6/i6_models",
            commit=models_commit,
            checkout_folder_name="i6_models",
            branch="bene_conf_enc" if models_commit == "75e03f37ac74d3d0c7358d29bb9b71dcec1bf120" else None,
        ).out_repository
        if models_commit == "75e03f37ac74d3d0c7358d29bb9b71dcec1bf120":
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

        returnn_config = ReturnnConfig(
            config=base_config,
            post_config=base_post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return returnn_config

    smaller_config = copy.deepcopy(blstm_base_config)
    smaller_config["batch_size"] = {"classes": 2 * 2000, "data": 2 * 320000}
    smaller_config["accum_grad_multiple_step"] = 2

    return {
        # "torch_conformer_baseline": construct_from_net_kwargs(
        #     medium_lchunk_config, {"model_type": "conformer_baseline"}
        # ),
        # "torch_conformer_no_cast": construct_from_net_kwargs(medium_lchunk_config, {"model_type": "conformer2"}),
        # "torch_conformer_new": construct_from_net_kwargs(medium_lchunk_config, {"model_type": "conformer3"}),
        # "torch_conformer_chunk_400_200": construct_from_net_kwargs(
        #     chunk_400_200_config, {"model_type": "conformer_baseline"}
        # ),
        # "torch_conformer_chunk_500_250": construct_from_net_kwargs(
        #     chunk_500_250_config, {"model_type": "conformer_baseline"}
        # ),
        # "torch_conformer_kernel_7": construct_from_net_kwargs(
        #     medium_lchunk_config, {"model_type": "conformer_baseline", "conv_kernel_size": 7}
        # ),
        # "torch_conformer_kernel_9": construct_from_net_kwargs(
        #     medium_lchunk_config, {"model_type": "conformer_baseline", "conv_kernel_size": 9}
        # ),
        # "torch_conformer_heads_6": construct_from_net_kwargs(
        #     medium_lchunk_config, {"model_type": "conformer_baseline", "att_heads": 6}
        # ),
        # "torch_conformer_ff_dim_1536": construct_from_net_kwargs(
        #     medium_lchunk_config, {"model_type": "conformer_baseline", "ff_dim": 1536}
        # ),
        # "torch_conformer_more_spec": construct_from_net_kwargs(
        #     batch_12k_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "spec_num_time": 15,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        # "torch_conformer_batch_12": construct_from_net_kwargs(batch_12k_config, {"model_type": "conformer_baseline"}),
        # "torch_conformer_batch_10": construct_from_net_kwargs(batch_10k_config, {"model_type": "conformer_baseline"}),
        # "torch_blstm_paper_baseline": construct_from_net_kwargs(wei_lstm_config, {"model_type": "blstm"}),
        # "torch_blstm_baseline": construct_from_net_kwargs(blstm_base_config, {"model_type": "blstm"}),
        # "torch_blstm_baseline2": construct_from_net_kwargs(smaller_config, {"model_type": "blstm"}),
        # "torch_blstm_baseline": construct_from_net_kwargs(blstm_base_config, {"model_type": "blstm"}),
        # "torch_conformer_better": construct_from_net_kwargs(
        #     batch_10k_chunk_400_config, {"model_type": "conformer_baseline", "conv_kernel_size": 7, "att_heads": 6}
        # ),
        # "torch_conformer_better_500": construct_from_net_kwargs(
        #     batch_10k_chunk_500_config, {"model_type": "conformer_baseline", "conv_kernel_size": 7, "att_heads": 6}
        # ),
        # "torch_conformer_better_12k_400": construct_from_net_kwargs(
        #     batch_12k_chunk_400_config, {"model_type": "conformer_baseline", "conv_kernel_size": 7, "att_heads": 6}
        # ),
        # "torch_jj_config": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 9,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        # "torch_jj_default_spec": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 9,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #     },
        # ),
        # "torch_jj_larger_chunk": construct_from_net_kwargs(
        #     jj_larger_chunk,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 9,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        "torch_jj_config2": construct_from_net_kwargs(
            jj_config,
            {
                "model_type": "conformer_baseline",
                "att_heads": 6,
                "conv_kernel_size": 7,
                "pool_1_stride": (3, 1),
                "ff_dim": 1536,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
            },
        ),
        # "torch_jj_config2_lessspec": construct_from_net_kwargs(  # 6.0
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 8,
        #     },
        # ),
        **{f"torch_jj_seed_{x}": construct_from_net_kwargs(
            jj_config,
            {
                "model_type": "conformer_baseline",
                "att_heads": 6,
                "conv_kernel_size": 7,
                "pool_1_stride": (3, 1),
                "ff_dim": 1536,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
            },
            random_seed=x,
        )for x in [24, 5, 2005]},
        # "torch_grad_acc_2": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        #     grad_acc=2
        # ),
        # "torch_grad_acc_3": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        #     grad_acc=3
        # ),
        # "torch_grad_acc_5": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        #     grad_acc=5
        # ),
        # "torch_grad_acc_10": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        #     grad_acc=10
        # ),
        # "torch_grad_acc_25": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        #     grad_acc=25
        # ),
        # "torch_jj_config3": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 11,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        "torch_jj_config2_large": construct_from_net_kwargs(
            jj_config,
            {
                "model_type": "conformer_baseline",
                "att_heads": 8,
                "conv_kernel_size": 9,
                "pool_1_stride": (3, 1),
                "ff_dim": 2048,
                "upsample_kernel": 3,
                "upsample_stride": 3,
                "upsample_padding": 0,
                "spec_num_time": 20,
                "spec_max_time": 20,
                "spec_num_feat": 5,
                "spec_max_feat": 16,
                "conformer_size": 512,
            },
        ),

        # "torch_jj_config2_larger": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 12,
        #         "conv_kernel_size": 13,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 3072,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "conformer_size": 768,
        #     },
        # ),
        # "torch_jj_config_ker_31": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 31,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        # "torch_jj_config_ker_19": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 19,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        # "torch_jj_config_ker_25": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 25,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        # ),
        # "torch_transparent_attention_baseline": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_transparent_att",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_transparent_attention_baseline_larger": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_transparent_att",
        #         "att_heads": 12,
        #         "conv_kernel_size": 13,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 3072,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "conformer_size": 768,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_transparent_attention_baseline_large": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_transparent_att",
        #         "att_heads": 8,
        #         "conv_kernel_size": 9,
        #         "pool_1_stride": (3, 1),
        #         "ff_dim": 2048,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "conformer_size": 512,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_sub4_21": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (2, 1),
        #         "pool_2_stride": (2, 1),
        #         "pool_1_kernel_size": (2, 1),
        #         "pool_2_kernel_size": (2, 1),
        #         "pool_1_padding": (1, 0),
        #         "pool_2_padding": (1, 0),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 4,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "linear_input_dim": 2560,
        #     },
        # ),
        # "torch_sub4_1": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (4, 1),
        #         "pool_2_stride": None,
        #         "pool_1_kernel_size": (4, 1),
        #         #"pool_2_kernel_size": (2, 1),
        #         "pool_1_padding": (2, 0),
        #         #"pool_2_padding": (1, 0),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 4,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "linear_input_dim": 1280,
        #     },
        # ),
        # "torch_sub4_12": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (2, 1),
        #         "pool_2_stride": (2, 1),
        #         "pool_1_kernel_size": (2, 1),
        #         "pool_2_kernel_size": (1, 2),
        #         "pool_1_padding": (1, 0),
        #         "pool_2_padding": None,
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 4,
        #         "upsample_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "linear_input_dim": 2528,
        #     },
        # ),
        # did not converge, instant loss on NaN
        # "torch_sub3": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": None,
        #         "pool_2_stride": (3, 1),
        #         "pool_1_kernel_size": (1, 1),
        #         "pool_2_kernel_size": (3, 1),
        #         "pool_1_padding": None,
        #         "pool_2_padding": (1, 0),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "upsample_out_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "linear_input_dim": 2560,
        #     },
        # ),
        # "torch_sub3_2121": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": None,
        #         "pool_2_stride": (3, 1),
        #         "pool_1_kernel_size": (1, 1),
        #         "pool_2_kernel_size": (2, 1),
        #         "pool_1_padding": None,
        #         "pool_2_padding": (1, 0),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 3,
        #         "upsample_padding": 0,
        #         "upsample_out_padding": 0,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "linear_input_dim": 2560,
        #     },
        # ),
        # "torch_sub4": construct_from_net_kwargs(
        #     jj_config,
        #     {
        #         "model_type": "conformer_baseline",
        #         "att_heads": 6,
        #         "conv_kernel_size": 7,
        #         "pool_1_stride": (2, 1),
        #         "pool_2_stride": (2, 1),
        #         "pool_1_kernel_size": (1, 2),
        #         "pool_2_kernel_size": (1, 2),
        #         "ff_dim": 1536,
        #         "upsample_kernel": 3,
        #         "upsample_stride": 4,
        #         "upsample_padding": 0,
        #         "upsample_out_padding": 1,
        #         "spec_num_time": 20,
        #         "spec_max_time": 20,
        #         "spec_num_feat": 5,
        #         "spec_max_feat": 16,
        #         "linear_input_dim": 2496,
        #     },
        # ),
    }
