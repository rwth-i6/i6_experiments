import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import ReturnnTrainingJob
import i6_core.rasr as rasr
from i6_core.returnn.forward import ReturnnForwardJobV2

from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.wu.experiments.modality_matching.networks import best_rq_conformer as best_rq
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, ReturnnConfigs

from i6_experiments.users.wu.corpus.librispeech.unsupervised_data import get_librispeech_unsupervised

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 8192
num_subepochs = 500

tools = copy.deepcopy(default_tools_v2)
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, extra_config:dict, lr: dict,
                             batch_size: int, network_args: dict) -> ReturnnConfig:
    model_config = best_rq.get_default_config_v1(num_inputs=80, num_outputs=num_outputs,
                                                       network_args=network_args)

    extra_config.update({
        "train": train_data_config,
        "dev": dev_data_config,
        "extern_data": {"data": {"dim": 1}}
    })
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}
    if variant == ConfigVariant.PRIOR:
        forward_data = copy.deepcopy(dev_data_config)
        forward_data["partition_epoch"] = 1
        extra_config["forward_data"] = forward_data

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="targets",
        extra_python=[best_rq.get_serializer(model_config, variant=variant)],
        extern_data_config=False,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=60,
        initial_lr=lr["initial_lr"],
        peak_lr=lr["peak_lr"],
        final_lr=lr["final_lr"],
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        extra_config: dict,
        lr: dict,
        network_args: dict,
        batch_size: int = 36000
) -> ReturnnConfigs[ReturnnConfig]:
    if "final_lr" not in lr:
        lr["final_lr"] = 1e-8
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "extra_config": extra_config,
                        "lr": lr, "batch_size": batch_size, "network_args": network_args}
    train_generator_kwargs = copy.deepcopy(generator_kwargs)
    train_config = returnn_config_generator(variant=ConfigVariant.TRAIN, **train_generator_kwargs)
    return ReturnnConfigs(
        train_config=train_config,
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def run_lbs_960_best_rq_pretrain():
    prefix = "experiments/ctc/conformer_baseline"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )

    # ********** Step args **********
    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs)
    train_args["mem_rqmt"] = 16

    # ********** Returnn Configs **********
    returnn_root = tk.Path(
        "/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/tools/20241021_returnn/returnn",
    )

    train_data_config, cv_data_config = get_librispeech_unsupervised(
        returnn_root,
    )

    for peak_lr in [1e-5]:
        peak_lr_dict = {
            "initial_lr": peak_lr / 100,
            "peak_lr": peak_lr,
        }

        for cb_distance_measure in ["L2_norm",]:
            for mask_prob in [0.6]:
                for mask_length in [16]:
                    for normalise_after_PCA in [True]:
                        for PCA_update_steps in [18000]:
                            for d_model in [768]:
                                for epoch_per_iter in [20]:
                                    for update_mask in ["sequence_mask"]:
                                        for update_from_labels in ["logmel_targets"]:
                                            peak_lr = 2e-4
                                            peak_lr_dict = {
                                                "initial_lr": peak_lr / 100,
                                                "peak_lr": peak_lr,
                                                "final_lr": 1e-8,
                                            }
                                            network_args = {
                                                "num_layers": 4,
                                                "aux_losses": {"4": 1},
                                                "input_codebook_dim": 16,
                                                "input_codebook_num_vars": 8192,
                                                "mask_replace_val": "zero",
                                                "mask_prob": mask_prob,
                                                "mask_length": mask_length,
                                                "cb_distance_measure": cb_distance_measure,
                                                "internal_subsampling_rate": 4,
                                                "normalise_after_PCA": normalise_after_PCA,
                                                "d_model": d_model,
                                            }

                                            extra_config = {
                                                "preload_from_files":{
                                                    "codebook": {
                                                        "filename": "/u/jxu/setups/pretraining/2025-02-28--best-rq-pretraining/alias/experiments/ctc/conformer_baseline/train/PCA_update_per_20_iterative_cb_refinement_d_768/output/models/epoch.600.pt",
                                                        "init_for_train": True,
                                                        "checkpoint_key": "model",
                                                        "ignore_missing": True,
                                                        "ignore_params_prefixes": ["final_linear", "conformer"],
                                                    }
                                                }
                                            }
                                            config_collection = get_returnn_config_collection(train_data_config, cv_data_config, extra_config, lr=peak_lr_dict,
                                                                                              batch_size=20000 * 160, network_args=network_args)

                                            train_args["returnn_python_exe"] = tk.Path("/usr/bin/python3")
                                            train_args["returnn_root"] = returnn_root
                                            train_job = ReturnnTrainingJob(config_collection.train_config, **train_args)
                                            train_job_name = f"PCA_d_768"
                                            train_job.add_alias(f"train/{train_job_name}")
                                            tk.register_output(f"best_rq_pretrain_lbs_960/{train_job_name}_models", train_job.out_model_dir)
                                            tk.register_output(f"best_rq_pretrain_lbs_960/{train_job_name}_learning_rates", train_job.out_learning_rates)
