import copy
import numpy as np
from typing import List

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.setups.rasr.util import HybridArgs

from i6_experiments.common.setups.serialization import Import, ExplicitHash, ExternalImport
from i6_experiments.common.setups.returnn_pytorch.serialization import PyTorchModel, Collection

from i6_experiments.users.jxu.experiments.hybrid.switchboard.default_tools import PACKAGE

RECUSRION_LIMIT = """
import sys
sys.setrecursionlimit(3000)
"""

def get_nn_args(
    num_outputs: int = 9001, num_epochs: int = 500, extra_exps=False, peak_lr=2e-3
):
    # evaluation_epochs = list(np.arange(num_epochs, num_epochs + 1, 10))
    evaluation_epochs = [240, 260]

    returnn_configs = get_pytorch_returnn_configs(
        num_inputs=40,
        num_outputs=num_outputs,
        num_epochs = 260,
        evaluation_epochs=evaluation_epochs,
        batch_size=14000,
        peak_lr=peak_lr,
        debug = True,
    )

    returnn_recog_configs = get_pytorch_returnn_configs(
        num_inputs=40,
        num_outputs=num_outputs,
        num_epochs=260,
        evaluation_epochs=evaluation_epochs,
        batch_size=14000,
        peak_lr=peak_lr,
        recognition=True,
        debug = True,
    )

    training_args = {
        "log_verbosity": 5,
        "num_epochs": num_epochs,
        "num_classes": num_outputs,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 7,
        "cpu_rqmt": 3,
        "partition_epochs": {"train": 6, "dev": 1},
        "use_python_control": False,
    }
    recognition_args = {
        "hub5e00": {
            "epochs": evaluation_epochs,
            "feature_flow_key": "gt",
            "prior_scales": [0.7, 0.8, 0.9],
            "pronunciation_scales": [6.0],
            "lm_scales": [6, 8, 10.0],
            "lm_lookahead": True,
            "lookahead_options": None,
            "create_lattice": True,
            "eval_single_best": True,
            "eval_best_in_lattice": True,
            "search_parameters": {
                "beam-pruning": 15.0,
                "beam-pruning-limit": 100000,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 10000,
            },
            "lattice_to_ctm_kwargs": {
                "fill_empty_segments": True,
                "best_path_algo": "bellman-ford",
            },
            "optimize_am_lm_scale": True,
            "rtf": 50,
            "mem": 8,
            "lmgc_mem": 16,
            "cpu": 4,
            "parallelize_conversion": True,
            # "forward_output_layer": "output",
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


def get_pytorch_returnn_configs(
        num_epochs: int, num_inputs: int, num_outputs: int, batch_size: int, peak_lr: float, evaluation_epochs: List[int],
        recognition=False, debug=False,
):
    # ******************** blstm base ********************

    base_config = {
        "extern_data": {
            "data": {"dim": num_inputs},
            "classes": {"dim": num_outputs, "sparse": True},
        },
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


    conformer_base_config = copy.deepcopy(base_config)
    conformer_base_config.update(
        {
            "batch_size": 12000,  # {"classes": batch_size, "data": batch_size},
            # "batching": 'sort_bin_shuffle:.64',
            "chunking": "500:250",
            "min_chunk_size": 10,
            "optimizer": {"class": "adam", "epsilon": 1e-8},
            "gradient_noise": 0.0,
            "learning_rates": list(np.linspace(peak_lr / 10, peak_lr, 100))
            + list(np.linspace(peak_lr, peak_lr / 10, 100))
            + list(np.linspace(peak_lr / 10, 1e-8, 60)),
            "learning_rate_control": "newbob_multi_epoch",
            "learning_rate_control_min_num_epochs_per_new_lr": 3,
            "learning_rate_control_relative_error_relative_lr": True,
            # "min_learning_rate": 1e-5,
            "newbob_learning_rate_decay": 0.9,
            "newbob_multi_num_epochs": 3,
            "newbob_multi_update_interval": 1,
        }
    )
    conformer_jingjing_config = copy.deepcopy(conformer_base_config)


    # those are hashed
    pytorch_package = PACKAGE + ".pytorch_networks"

    def construct_from_net_kwargs(base_config, net_kwargs, explicit_hash=None, use_tracing=False,
                                  use_custom_engine=False, use_espnet=False, use_i6_models=False,
                                  aux_loss_scale=None, drop_one_layer=None):
        if aux_loss_scale is not None:
            base_config["_kwargs"] = {"aux_loss_scale": aux_loss_scale}
        model_type = net_kwargs.pop("model_type")
        # pytorch_model_import = Import(
        #     PACKAGE + ".pytorch_networks.%s.Model" % model_type
        # )
        pytorch_model_import = Import(
            PACKAGE + ".pytorch_networks.%s.Model" % model_type
        )
        pytorch_train_step = Import(
            PACKAGE + ".pytorch_networks.%s.train_step" % model_type
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
        if use_espnet:
            espnet_path = CloneGitRepositoryJob(
                url="https://github.com/espnet/espnet",
                checkout_folder_name="espnet"
            ).out_repository
            espnet_path.hash_overwrite = "DEFAULT_ESPNET"
            serializer_objects.insert(0, ExternalImport(espnet_path))
        if use_i6_models:
            i6_models_repo = CloneGitRepositoryJob(
                url="https://github.com/rwth-i6/i6_models",
                commit="83af04d84f6a223a980f0bed8db6d2d1466dd690",
                checkout_folder_name="i6_models"
            ).out_repository
            i6_models_repo.hash_overwrite = "LIBRISPEECH_DEFAULT_I6_MODELS"
            i6_models = ExternalImport(import_path=i6_models_repo)
            serializer_objects.append(i6_models)
        if use_custom_engine:
            pytorch_engine = Import(
                PACKAGE + ".pytorch_networks.%s.CustomEngine" % model_type
            )
            serializer_objects.append(pytorch_engine)
        if recognition:
            if use_tracing:
                pytorch_export = Import(
                    PACKAGE + ".pytorch_networks.%s.export_trace" % model_type,
                    import_as="export"
                )
            else:
                pytorch_export = Import(
                    PACKAGE + ".pytorch_networks.%s.export" % model_type
                )
            serializer_objects.append(pytorch_export)
        if explicit_hash:
            serializer_objects.append(ExplicitHash(explicit_hash))
        serializer = Collection(
            serializer_objects=serializer_objects,
            make_local_package_copy=not debug,
            packages={
                pytorch_package,
            },
        )

        blstm_base_returnn_config = ReturnnConfig(
            config=base_config,
            post_config=base_post_config,
            python_epilog=[serializer],
            pprint_kwargs={"sort_dicts": False},
        )

        return blstm_base_returnn_config

    return {
        "i6_conformer_epochs_{}_peak_lr_{}".format(num_epochs, str(peak_lr).replace(".", "_")): construct_from_net_kwargs(
            conformer_jingjing_config,
            {"model_type": "i6_conformer_downsample_3",
             "model_size": 384,
             "num_layers": 12,
             "kernel_size": 9,
             "num_repeat_time": 15,
             "max_dim_time": 20,
             "num_repeat_feat": 5,
             "max_dim_feat": 10},
            use_tracing=True,
            drop_one_layer=1
            ),
    }
