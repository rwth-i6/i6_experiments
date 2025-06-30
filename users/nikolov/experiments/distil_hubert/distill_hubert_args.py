import copy
from typing import List, Dict, Any, Optional, Union

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.hilmes.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection


# from i6_experiments.common.baselines.tedlium2.default_tools import PACKAGE

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
#    evaluation_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + list(range(10, num_epochs + 1, 10))
    evaluation_epochs = list(range(num_epochs, num_epochs + 1, 10))

    batch_size = {"classes": 14000}
    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=1,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=None,
        evaluation_epochs=evaluation_epochs,
        debug=debug,
        num_epochs=num_epochs,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=1,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=None,
        evaluation_epochs=evaluation_epochs,
        recognition=True,
        debug=debug,
    )

    training_args = ReturnnTrainingJobArgs(
        log_verbosity=5, num_epochs=num_epochs, save_interval=1, keep_epochs=None, time_rqmt=168, mem_rqmt=8, cpu_rqmt=3
    )

    recognition_args = {
        "dev": {
            "epochs": evaluation_epochs + ["best", "avrg"],
            "feature_flow_key": "fb",
            "prior_scales": [0.7, 0.9],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0],
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
            "mem": 6,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": True,
            "nn_prior": True,
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
        #     "needs_features_size": False,
        #     #"quantize": [10, 15, 25, 100, 250, 500, 750, 1000, 2500, 5000],
        #     "quantize": [10, 15, 25, 100, 250, 500, 750, 1000, 2500, 5000],
        #     "training_whitelist": [],
        # },
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


def get_pytorch_returnn_configs(
    num_inputs: int,
    num_outputs: int,
    batch_size: Union[int, Dict[str, int]],
    chunking: Optional[str],
    evaluation_epochs: List[int],
    recognition=False,
    debug=False,
    num_epochs=None,
):
    # ******************** blstm base ********************

    base_config = {
        "extern_data": {
            "data_raw": {
                "dim": 1,
                "shape": (None, 1),
                "available_for_inference": True,
            },  # input: 1-dimensional waveforms
            "classes": {
                "dim": num_outputs,
                "shape": (None,),
                "available_for_inference": not recognition,
                "sparse": True,
                "dtype": "int16",
            },
        },
        "min_seq_length": {"classes": 1},
        "model_outputs": {
            "output": {"dim": num_outputs}
        },
        "log_batch_size": True,
        "save_ignore_params_prefixes": ["hubert"],
        "behavior_version": 16,
        "torch_log_memory_usage": True,
    }

    base_post_config = {
        "backend": "torch",
        "debug_print_layer_output_template": True,
    }
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }

    hubert_config = copy.deepcopy(base_config)
    peak_lr = 7e-4
    import numpy as np
    if num_epochs is None:
        num_epochs = 250
    hubert_config["learning_rates"] = (
        list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr / 10, 1e-8, 30))
    )
    hubert_config["gradient_clip"] = 1.0
    hubert_config["optimizer"] = {"class": "adam", "epsilon": 1e-8}
    hubert_config["batch_size"] = batch_size
    #hubert_config["max_seqs"] = 50

    chunk_400_200_config = copy.deepcopy(hubert_config)
    chunk_400_200_config["chunking"] = "400:200"

    chunk_raw_config = copy.deepcopy(hubert_config)
    chunk_400 = 400 * 16000
    chunk_200 = 200 * 16000
    chunk_raw_config["chunking"] = f"{chunk_400}:{chunk_200}"

    #if not recognition:
    #    del chunk_400_200_config['extern_data']['data_raw']

    # those are hashed
    PACKAGE = __package__
    package = PACKAGE
    pytorch_package = package + ".pytorch_networks"

    def construct_from_net_kwargs(
        base_config,
        net_kwargs,
        explicit_hash: Optional[str] = None,
        models_commit: str = "75e03f37ac74d3d0c7358d29bb9b71dcec1bf120",
        post_config: Optional[Dict] = None,
        max_seqs: Optional[int] = None,
        grad_acc: Optional[int] = None,
        learning_rate: Optional[float] = None,
        debug=False,
    ):
        if post_config is None:
            post_config = base_post_config
        if any(x is not None for x in [max_seqs, grad_acc, learning_rate]):
            base_config = copy.deepcopy(base_config)
            if max_seqs is not None:
                base_config["max_seqs"] = max_seqs
            if grad_acc is not None:
                base_config["accum_grad_multiple_step"] = grad_acc
            if learning_rate is not None:
                base_config["learning_rate"] = learning_rate
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
        if recognition:  # this really is just abused for prior, not needed for actual recog (due to onnx export)
            #pytorch_export = Import(package + ".pytorch_networks.%s.export" % model_type)
            #serializer_objects.append(pytorch_export)
            base_config = copy.deepcopy(base_config)
            base_config["max_seqs"] = 1
            base_config["forward_data"] = "train"
            base_config["model_outputs"] = {"log_probs": {"dim": num_outputs, "shape": (None, num_outputs)}}
            base_config["extern_data"]["data_raw"] = {
                "dim": 80,
                "shape": (None, 80),
                "available_for_inference": True,
            }
            if "min_seq_length" in base_config:
                del base_config["min_seq_length"]

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
        ).out_repository
        i6_models_repo.hash_overwrite = "TEDLIUM2_HUBERT_DISTILL_I6_MODELS"
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
            post_config=post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return returnn_config

    return {
        **{f"torch_distill_hubert_fe_test": construct_from_net_kwargs(
            chunk_raw_config,
            {
                "model_type": "distill_hubert_v1",
                "hubert_dict": {
                    "model_name": "base-ls960",
                    "distill_scale": x
                },
                "conformer_dict": {
                    "hidden_d": 384,
                    "conv_kernel_size": 7,
                    "att_heads": 6,
                    "ff_dim": 1536,
                    "spec_num_time": 20,
                    "spec_max_time": 20,
                    "spec_num_feat": 5,
                    "spec_max_feat": 16,
                    "pool_1_stride": (3, 1),
                    "pool_1_kernel_size": (1, 2),
                    "pool_1_padding": None,
                    "pool_2_stride": None,
                    "pool_2_kernel_size": (1, 2),
                    "pool_2_padding": None,
                    "num_layers": 12,
                    "upsample_kernel": 3,
                    "upsample_stride": 3,
                    "upsample_padding": 0,
                    "upsample_out_padding": 1,
                    "dropout": 0.2,
                    "feat_extr": True
                },
            },
            models_commit="3c9173691521778b1e8b4070c172cbe929e4826b",
            # max_seqs=2,
            # grad_acc=14,
        ) for x in [0.00]},
    }
