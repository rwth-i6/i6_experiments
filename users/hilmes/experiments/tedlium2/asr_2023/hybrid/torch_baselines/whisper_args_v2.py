import copy
from typing import List, Dict, Any, Optional

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection


# from i6_experiments.common.baselines.tedlium2.default_tools import PACKAGE

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
#    evaluation_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + list(range(10, num_epochs + 1, 10))
    evaluation_epochs = [1, 5] + list(range(10, num_epochs + 1, 10))

    batch_size = 20000
    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=80,
        num_outputs=num_outputs,
        batch_size=batch_size,
        chunking=None,
        evaluation_epochs=evaluation_epochs,
        debug=debug,
        num_epochs=num_epochs,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=80,
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
            "epochs": evaluation_epochs + ["best"],
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
            "mem": 7,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": False,
        },
        "tune": {
            "epochs": evaluation_epochs + ["best"],
            "feature_flow_key": "fb",
            "prior_scales": [0.9, 1.0, 1.1],
            "pronunciation_scales": [0.0],
            "lm_scales": [10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 14.0,
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
            "needs_features_size": False,
            "training_whitelist": [
                "torch_whisper_v2_medium_en_tune_last_0_0001_split_10",
                "torch_whisper_v2_medium_en_tune_last_0_0001_split_10_acc_50"
            ],
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
    batch_size: int,
    chunking: Optional[str],
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
        },
        "min_seq_length": {"classes": 1},
    }
    base_post_config = {
        "backend": "torch",
        "debug_print_layer_output_template": True,
        "log_batch_size": True,
    }
    if not recognition:
        base_post_config["cleanup_old_models"] = {
            "keep_last_n": 5,
            "keep_best_n": 5,
            "keep": evaluation_epochs,
        }

    whisper_config = copy.deepcopy(base_config)
    whisper_config["learning_rate"] = 0.001
    whisper_config["gradient_clip"] = 1.0
    whisper_config["optimizer"] = {"class": "adam", "epsilon": 1e-8}
    whisper_config["batch_size"] = batch_size
    whisper_config["max_seqs"] = 50

    whisper_04 = copy.deepcopy(whisper_config)
    whisper_04["learning_rate"] = 0.0004
    whisper_02 = copy.deepcopy(whisper_config)
    whisper_02["learning_rate"] = 0.0002
    whisper_01 = copy.deepcopy(whisper_config)
    whisper_01["learning_rate"] = 0.0001
    whisper_005 = copy.deepcopy(whisper_config)
    whisper_005["learning_rate"] = 0.00005

    whisper_less_seqs = copy.deepcopy(whisper_config)
    whisper_less_seqs["max_seqs"] = 10

    whisper_04_less = copy.deepcopy(whisper_04)
    whisper_04_less["max_seqs"] = 10

    whisper_jj = copy.deepcopy(whisper_config)
    peak_lr = 7e-4
    if num_epochs is None:
        num_epochs = 250
    import numpy as np
    whisper_jj["learning_rates"] = (
        list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr, peak_lr / 10, (num_epochs - 30) // 2))
        + list(np.linspace(peak_lr / 10, 1e-8, 30))
    )
    del whisper_jj["learning_rate"]
    whisper_keep_peak = copy.deepcopy(whisper_config)
    peak_lr = 7e-4
    if num_epochs is None:
        num_epochs = 250
    import numpy as np
    whisper_keep_peak["learning_rates"] = (
        list(np.linspace(peak_lr / 10, peak_lr, (num_epochs - 30) // 2))
        + [peak_lr] * 30
        + list(np.linspace(peak_lr, 1e-8, (num_epochs - 30) // 2))
    )

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
        torch_amp=None,
    ):
        if post_config is None:
            post_config = base_post_config
        if any(x is not None for x in [max_seqs, grad_acc, learning_rate, torch_amp]):
            base_config = copy.deepcopy(base_config)
            if max_seqs is not None:
                base_config["max_seqs"] = max_seqs
            if grad_acc is not None:
                base_config["accum_grad_multiple_step"] = grad_acc
            if learning_rate is not None:
                base_config["learning_rate"] = learning_rate
            if torch_amp is not None:
                base_config["torch_amp"] = torch_amp
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

            if "hubert" in model_type:
                prior_computation = Import(package + ".pytorch_networks.prior.basic.forward_step")
            else:
                prior_computation = Import(package + ".pytorch_networks.prior.whisper_prior.forward_step")
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
            post_config=post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return returnn_config

    return {
     **{f"torch_whisper_v3_medium_en_keepuntil_{x}_jjlr_split_10_acc_50": construct_from_net_kwargs(
        whisper_jj,
        {
            "model_type": "whisper_tune_v3",
            "just_encoder": True,
            "split_seq": True,
            "whisper_model": "medium.en",
            "keep_layers": x
        },
        models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        post_config=base_post_config,
        max_seqs=3 if not isinstance(x, int) or x < 21 else 2,
        grad_acc=50 if not isinstance(x, int) or x < 21 else 75,
        ) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, [0, 1, 2, 8, 16, 21, 22, 23]]},
    **{f"torch_whisper_v3_medium_en_keeponly_{x}_jjlr_split_10_acc_50": construct_from_net_kwargs(
        whisper_jj,
        {
            "model_type": "whisper_tune_v3",
            "just_encoder": True,
            "split_seq": True,
            "whisper_model": "medium.en",
            "keep_layers": x,
        },
        models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        post_config=base_post_config,
        max_seqs=2,
        grad_acc=75
    ) for x in [[0, 2, 4, 6, 8, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 23], [0, 1, 2, 8, 16, 21, 22, 23]]},
    **{f"torch_whisper_v3_medium_en_keepuntil_{x}_jjlr_bfloat16": construct_from_net_kwargs(
            whisper_jj,
            {
                "model_type": "whisper_tune_v3",
                "just_encoder": True,
                "split_seq": True,
                "whisper_model": "medium.en",
                "keep_layers": x
            },
            models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
            post_config=base_post_config,
            max_seqs=6,
            grad_acc=25,
            torch_amp={"dtype": "bfloat16"}
    ) for x in [9]},
    **{f"torch_whisper_v3_medium_en_keepuntil_{x}_jjlr_bfloat16_test": construct_from_net_kwargs(
            whisper_jj,
            {
                "model_type": "whisper_tune_v3",
                "just_encoder": True,
                "split_seq": True,
                "whisper_model": "medium.en",
                "keep_layers": x
            },
            models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
            post_config=base_post_config,
            max_seqs=3,
            grad_acc=50,
            torch_amp={"dtype": "bfloat16"}
    ) for x in [9]},
    "torch_whisper_v3_medium_en_tune_first12_jjlr_split_10_acc_50": construct_from_net_kwargs(
        whisper_jj,
        {
            "model_type": "whisper_tune_v3",
            "just_encoder": True,
            "finetune_whisper": [x for x in range(24)],
            "split_seq": True,
            "whisper_model": "medium.en",
        },
        models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        post_config=base_post_config,
        max_seqs=3,
        grad_acc=50
    ),
    }
