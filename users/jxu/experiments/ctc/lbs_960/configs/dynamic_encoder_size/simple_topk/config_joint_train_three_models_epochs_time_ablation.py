import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.dynamic_encoder_size.simple_topk import joint_train_three_model_simple_topk_modwise as conformer_ctc
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.jxu.experiments.ctc.lbs_960.ctc_data import get_librispeech_data_hdf
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 600

tools = copy.deepcopy(default_tools_v2)
tools.returnn_root = tk.Path("/u/jxu/setups/librispeech-100/2023-06-10--torch-model/tools/20230823-returnn",
                       hash_overwrite="/u/berger/repositories/returnn")

# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int, network_args:dict, kwargs:dict) -> ReturnnConfig:

    model_config = conformer_ctc.get_default_config_v1(num_inputs=80, num_outputs=num_outputs,network_args=network_args)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="targets",
        extra_python=[conformer_ctc.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
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
        **kwargs
    )


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        lr: dict,
        network_args: dict,
        batch_size: int,
        kwargs: dict
) -> ReturnnConfigs[ReturnnConfig]:
    if "final_lr" not in lr:
        lr["final_lr"] = 1e-8
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "network_args": network_args, "kwargs":kwargs}
    train_generator_kwargs = copy.deepcopy(generator_kwargs)
    train_generator_kwargs['network_args']["recog_num_mods"] = 48
    train_config = returnn_config_generator(variant=ConfigVariant.TRAIN, **train_generator_kwargs)
    return ReturnnConfigs(
        train_config=train_config,
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def run_lbs_960_torch_conformer_raw_wave_wei_hyper() -> SummaryReport:
    prefix = "experiments/ctc/conformer_baseline"
    gs.ALIAS_AND_OUTPUT_SUBDIR = (
        prefix
    )
    
    data = get_librispeech_data_hdf(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
        blank_index_last=False,
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[num_subepochs],
        prior_scales=[0.2, 0.3],
        lm_scales=[0.8, 0.9],
        feature_type=FeatureType.SAMPLES,
        flow_args={"scale_input": 1}
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring(score_kwargs={"sctk_binary_path": SCTK_BINARY_PATH})

    # ********** Returnn Configs **********

    for peak_lr in [4e-4]:
        for k_annealing_step, k_reduction_per_iter in [(8, 1), (9.5, 1), (12, 1), (13.5, 1)]:
            for layer_dropout_mod_select, layer_dropout_fix_mod in [(0, 0.3)]:
                layer_dropout = {"layer_dropout_mod_select": layer_dropout_mod_select,
                                 "layer_dropout_fix_mod": layer_dropout_fix_mod}
                k_anneal_args = {"k_anneal_num_steps_per_iter": 1220 * k_annealing_step,
                                 "k_reduction_per_iter": k_reduction_per_iter}
                network_args = {"small_model_num_mods": 16, "medium_model_num_mods": 32, "start_select_step": 0,
                                "layer_dropout": layer_dropout,
                                "tau_args": {"initial_tau": 2, "annealing": 0.999992, "min_tau": 0.1,
                                             "gumbel_scale": 0.05},
                                "k_anneal_args": k_anneal_args, "recog_num_mods": 48}

                network_args.update({"time_max_mask_per_n_frames": 25,
                                     "freq_max_num_masks": 5,
                                     "vgg_act": "relu",
                                     "dropout": 0.1,
                                     "num_layers": 12})
                peak_lr_dict = {
                    "initial_lr": peak_lr / 100,
                    "peak_lr": peak_lr,
                }
                str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
                str_layer_dropout = str(layer_dropout_mod_select).replace("-", "_").replace(".", "_")
                str_layer_dropout += "_"
                str_layer_dropout += str(layer_dropout_fix_mod).replace("-", "_").replace(".", "_")
                system.add_experiment_configs(
                    f"thee_model_subepochs_600_peak_lr_{str_peak_lr}_k_annealing_{k_annealing_step}_{k_reduction_per_iter}_layer_dropout_{str_layer_dropout}_mod_48",
                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                  batch_size=15000 * 160, network_args=network_args, kwargs={})
                )

                network_args["recog_num_mods"] = 32
                system.add_experiment_configs(
                    f"thee_model_subepochs_600_peak_lr_{str_peak_lr}_k_annealing_{k_annealing_step}_{k_reduction_per_iter}_layer_dropout_{str_layer_dropout}_mod_32",
                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                  batch_size=15000 * 160, network_args=network_args, kwargs={})
                )

                network_args["recog_num_mods"] = 16
                system.add_experiment_configs(
                    f"thee_model_subepochs_600_peak_lr_{str_peak_lr}_k_annealing_{k_annealing_step}_{k_reduction_per_iter}_layer_dropout_{str_layer_dropout}_mod_16",
                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                  batch_size=15000 * 160, network_args=network_args, kwargs={})
                )


    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

