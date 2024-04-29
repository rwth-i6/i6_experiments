import copy
from typing import List, Dict, Any, Optional

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.users.hilmes.common.setups.rasr.util import HybridArgs, ReturnnTrainingJobArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection

from onnxruntime.quantization import QuantizationMode, CalibrationMethod, QuantType

# from i6_experiments.common.baselines.tedlium2.default_tools import PACKAGE

def get_nn_args(num_outputs: int = 9001, num_epochs: int = 250, debug=False, **net_kwargs):
#    evaluation_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + list(range(10, num_epochs + 1, 10))
    evaluation_epochs = list(range(10, num_epochs + 1, 10))

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
            "mem": 6,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": False,
        },
        "tune": {
            "epochs": evaluation_epochs + ["best"],
            "feature_flow_key": "fb",
            "prior_scales": [0.5, 0.7, 0.9, 1.1],
            "pronunciation_scales": [0.0],
            "lm_scales": [7.5, 10.0, 15.0],
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
            "mem": 6,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": False,
            "training_whitelist": [
                "torch_whisper_v3_medium_en_tune_0_jjlr_split_10_acc_50",
                #"torch_whisper_v3_medium_en_tune_1_jjlr_split_10_acc_50", # TODO
                #"torch_whisper_v3_medium_en_tune_2_jjlr_split_10_acc_50", # TODO
                #"torch_whisper_v3_medium_en_tune_3_jjlr_split_10_acc_50", # TODO
            ],
        },
        "quant": {
            "epochs": [100],
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
            "mem": 6,
            "lmgc_mem": 16,
            "cpu": 2,
            "parallelize_conversion": True,
            "needs_features_size": False,
            #"quantize": [10, 15, 25, 100, 250, 500, 750, 1000, 2500, 5000],
            "quantize": [10, 15, 25, 100],
            "quant_modes": [CalibrationMethod.MinMax],
            "quant_avg_modes": [True, False],
            "quant_sym_modes": [False],
            "quant_activation_types": [QuantType.QInt8, QuantType.QUInt8],
            "quant_weight_types": [QuantType.QInt8, QuantType.QUInt8],
            "quant_ops": [None, ["Conv"], ["MatMul"], ["Conv", "MatMul"], ["Conv", "MatMul", "Mul", "Add"]],
            "training_whitelist": [
                "torch_whisper_v3_medium_en_tune_0_jjlr_split_10_acc_50",
                #"torch_whisper_v3_medium_en_tune_1_jjlr_split_10_acc_50",
                "torch_whisper_v3_medium_en_tune_2_jjlr_split_10_acc_50",
                #"torch_whisper_v3_medium_en_tune_3_jjlr_split_10_acc_50",

            ],
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
        "tf_log_memory_usage": True,
        "cache_size": "0",
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
        # "torch_whisper_large_tune_1_a100": construct_from_net_kwargs(  # 6.6
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "large",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     grad_acc=100,
        #     max_seqs=1,
        # ),
        # "torch_whisper_large_tune_1_a20": construct_from_net_kwargs(  # 7.0
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "large",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     grad_acc=20,
        #     max_seqs=1,
        # ),
        # "torch_whisper_base_en_tune_last_0_0004_a20": construct_from_net_kwargs( # 7.3
        #     whisper_config,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     grad_acc=20,
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     learning_rate=0.0004,
        #     max_seqs=10
        # ),
        # "torch_whisper_base_en_tune_last_0_0004_a50": construct_from_net_kwargs( # 7.0
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     grad_acc=50,
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_v2_large_tune_last_0_0004_split_10_a20": construct_from_net_kwargs( # 6.7
        #     whisper_04,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "large-v2",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=2,
        #     grad_acc=20
        # ),
        # "torch_whisper_v2_large_tune_last_0_0004_split_10_a50": construct_from_net_kwargs(  # 6.4
        #     whisper_04,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "large-v2",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=2,
        #     grad_acc=50
        # ),
        # "torch_whisper_v2_large_tune_last_0_0004_split_10_a100": construct_from_net_kwargs(  # 5.8
        #     whisper_04,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "large-v2",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=2,
        #     grad_acc=100
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0004_split_10_a20": construct_from_net_kwargs(  # 6.3
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=20
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0004_split_10_a50": construct_from_net_kwargs(  # 5.9
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0001_split_10": construct_from_net_kwargs(  # 6.1
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_00005_split_10": construct_from_net_kwargs(  # 6.0
        #     whisper_005,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_00005_split_10_a20": construct_from_net_kwargs(  # 5.9
        #     whisper_005,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=20
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0001_split_10_acc_20": construct_from_net_kwargs(  # 5.9
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=20
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0001_split_10_acc_50": construct_from_net_kwargs(  # 5.7
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v2_medium_en_tune_last_jj_split_10_acc_50": construct_from_net_kwargs(  # 5.9
        #     whisper_jj,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v2_medium_en_tune_last_jjlr_split_10_acc_50": construct_from_net_kwargs(  # 5.9
        #     whisper_jj,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v3_medium_en_tune_last_jjlr_split_10_acc_50": construct_from_net_kwargs(  # 9.7
        #     whisper_jj,
        #     {
        #         "model_type": "whisper_tune_v3",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v3_medium_en_tune_last6_jjlr_split_10_acc_50": construct_from_net_kwargs(  # 9.2
        #     whisper_jj,
        #     {
        #         "model_type": "whisper_tune_v3",
        #         "just_encoder": True,
        #         "finetune_whisper": 6,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
       # "torch_whisper_v3_medium_en_tune_first12_jjlr_split_10_acc_50": construct_from_net_kwargs(  # 6.8
       #      whisper_jj,
       #      {
       #          "model_type": "whisper_tune_v3",
       #          "just_encoder": True,
       #          "finetune_whisper": [x for x in range(12)],
       #          "split_seq": True,
       #          "whisper_model": "medium.en",
       #      },
       #      models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
       #      post_config=base_post_config,
       #      max_seqs=3,
       #      grad_acc=50
       #  ),
       # "torch_whisper_v3_medium_en_tune_last12_jjlr_split_10_acc_50": construct_from_net_kwargs(  # 7.3
       #      whisper_jj,
       #      {
       #          "model_type": "whisper_tune_v3",
       #          "just_encoder": True,
       #          "finetune_whisper": 12,
       #          "split_seq": True,
       #          "whisper_model": "medium.en",
       #      },
       #      models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
       #      post_config=base_post_config,
       #      max_seqs=3,
       #      grad_acc=50
       #  ),
        # "torch_whisper_v3_medium_en_tune_last_0_0001_split_10_acc_50": construct_from_net_kwargs(  # 9.6
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v3",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v3_medium_en_tune_2_0_0001_split_10_acc_50": construct_from_net_kwargs(  # 9.6
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v3",
        #         "just_encoder": True,
        #         "finetune_whisper": 2,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0001_split_10_acc_50_0.2": construct_from_net_kwargs(  # 5.8
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #         "dropout": 0.2,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50,
        # ),
        # "torch_whisper_v2_medium_en_tune_last_0_0001_split_10_acc_50_0.5": construct_from_net_kwargs(  # 5.8
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #         "dropout": 0.5,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50,
        # ),
        # "torch_whisper_v2_medium_tune_last_0_0001_split_10_acc_50": construct_from_net_kwargs(  # 6.2
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "medium",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ),
        # "torch_whisper_v2_base_en_tune_2_0_0004_a20": construct_from_net_kwargs( # 7.3
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 2,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     grad_acc=20
        # ),
        # "torch_whisper_v2_base_en_tune_2_04_acc_20": construct_from_net_kwargs(  # 6.9
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 2,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     grad_acc=20,
        # ),
        # DEPRECATED EXPs might not directly run anymore
        # "torch_whisper_v2_base_en_tune_2_0_0004_a50": construct_from_net_kwargs(
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 2,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     grad_acc=50
        # ),
        # "torch_whisper_base_en_tune_3": construct_from_net_kwargs(
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 3,
        #         "split_seq": True,
        #         "whisper_model": "base.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_finetuning": construct_from_net_kwargs(
        #     whisper_config,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_0_0008": construct_from_net_kwargs(
        #     whisper_08,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_0_0006": construct_from_net_kwargs(
        #     whisper_06,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_0_0004": construct_from_net_kwargs(
        #     whisper_04,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_base_en": construct_from_net_kwargs(
        #     whisper_config,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #         "whisper_model": "base.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_base_0_0004": construct_from_net_kwargs(
        #     whisper_04,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #         "whisper_model": "base.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_tiny_en": construct_from_net_kwargs(
        #     whisper_tiny,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #         "whisper_model": "tiny.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_tiny": construct_from_net_kwargs(
        #     whisper_tiny,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #         "whisper_model": "tiny"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_small_en": construct_from_net_kwargs(
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #         "whisper_model": "small.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_medium_en": construct_from_net_kwargs(
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #         "whisper_model": "medium.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_base_en_tune_last": construct_from_net_kwargs(
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "base.en"
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_0_0002": construct_from_net_kwargs(
        #     whisper_02,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": False,
        #         "split_seq": True,
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_v2_base_en_tune_last_0_0004": construct_from_net_kwargs(
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 1,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_v2_base_en_tune_3_0_0004": construct_from_net_kwargs(
        #     whisper_04_less,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": 3,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        # "torch_whisper_base_en_tune_2": construct_from_net_kwargs(
        #     whisper_less_seqs,
        #     {
        #         "model_type": "whisper_tune",
        #         "just_encoder": True,
        #         "finetune_whisper": 2,
        #         "split_seq": True,
        #         "whisper_model": "base.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        # ),
        **{f"torch_whisper_v3_medium_en_tune_{x}_jjlr_split_10_acc_50": construct_from_net_kwargs(
            whisper_jj,
            {
                "model_type": "whisper_tune_v3",
                "just_encoder": True,
                "finetune_whisper": [x],
                "split_seq": True,
                "whisper_model": "medium.en",
            },
            models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
            post_config=base_post_config,
            max_seqs=3,
            grad_acc=50
        #) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]},
        ) for x in [0, 1, 2, 3, 4, 5, 6]},
        # **{f"torch_whisper_v3_medium_en_tune_{x}_keep7e4_split_10_acc_50": construct_from_net_kwargs(  # 6.3
        #     whisper_keep_peak,
        #     {
        #         "model_type": "whisper_tune_v3",
        #         "just_encoder": True,
        #         "finetune_whisper": x if isinstance(x, list) else [x],
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ) for x in [[0, 1, 2, 3, 4, 5, 6, 7, 8]]},
        # **{f"torch_whisper_v2_medium_en_tune_{x}_01r_split_10_acc_50": construct_from_net_kwargs(
        #     whisper_01,
        #     {
        #         "model_type": "whisper_tune_v2",
        #         "just_encoder": True,
        #         "finetune_whisper": [x],
        #         "split_seq": True,
        #         "whisper_model": "medium.en",
        #     },
        #     models_commit="95f55219d3d882b9386eac8e7c2b52b53e829b97",
        #     post_config=base_post_config,
        #     max_seqs=3,
        #     grad_acc=50
        # ) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
    }
