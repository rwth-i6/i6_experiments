import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.dynamic_encoder_size.zeroout import joint_train_three_model_zeroout_modwise as conformer_ctc
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

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=11)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[num_subepochs],
        prior_scales=[0.2, 0.3],
        lm_scales=[0.8, 0.9, 1.0],
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
        for stage_1_layer_dropout_on_large in [False]:
            for layer_dropout in [0.3, 0.4]:
            # for layer_dropout in [0.3]:
            #     for stage_1_num_steps_per_iter, stage_1_num_zero_per_iter in [([80, 70], [16, 32])]:
                for stage_1_num_steps_per_iter, stage_1_num_zero_per_iter in [([80,70], [16,32]), ([60]+[30]*3,[8,16,24,32]), ([30]+[18]*7,[4,8,12,16,20,24,28,32])]:
                    for zeroout_val in [-3]:
                        stage_1_expected_sparsity_per_iter = [n/48 for n in stage_1_num_zero_per_iter]
                        stage_1_num_steps_per_iter = [n*1400 for n in stage_1_num_steps_per_iter]
                        for sparsity_loss_scale in [5]:
                            for stage_2_small_model_loss_scale in [0.3]:
                                stage_args = {
                                    "stage_1_layer_dropout_on_large": stage_1_layer_dropout_on_large,
                                    "stage_1_num_steps_per_iter": stage_1_num_steps_per_iter,
                                    "stage_1_num_zero_per_iter": stage_1_num_zero_per_iter,
                                    "stage_1_expected_sparsity_per_iter": stage_1_expected_sparsity_per_iter,
                                    "gate_activation": "sigmoid",
                                    "stage_2_small_model_loss_scale": stage_2_small_model_loss_scale,
                                    "zeroout_val": zeroout_val
                                }
                                if stage_2_small_model_loss_scale == 0.3:
                                    del stage_args["stage_2_small_model_loss_scale"]
                                network_args = {"small_model_num_mods": 16, "medium_model_num_mods": 32, "layer_dropout": layer_dropout, "recog_num_mods": 48,
                                                "stage_args": stage_args, "sparsity_loss_scale": sparsity_loss_scale}

                                network_args.update({"time_max_mask_per_n_frames": 25,
                                                "freq_max_num_masks": 5,
                                                "vgg_act": "relu",
                                                "dropout": 0.1,
                                                "num_layers": 12})
                                peak_lr_dict = {
                                    "initial_lr": peak_lr/100,
                                    "peak_lr": peak_lr,
                                }
                                str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
                                str_layer_dropout = str(layer_dropout).replace("-", "_").replace(".", "_")
                                system.add_experiment_configs(
                                    f"batch_10k_zeroout_val_{abs(zeroout_val)}_joint_train_after_selection_iter_{len(stage_1_num_zero_per_iter)}_stage_2_small_model_loss_scale_{stage_2_small_model_loss_scale}_loss_scale_{sparsity_loss_scale}_layer_dropout_{layer_dropout}_num_recog_mods_48_peak_lr_{peak_lr}",
                                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                                  batch_size=10000 * 160, network_args=network_args, kwargs={})
                                    )

                                network_args["recog_num_mods"] = 32
                                system.add_experiment_configs(
                                    f"batch_10k_zeroout_val_{abs(zeroout_val)}_joint_train_after_selection_iter_{len(stage_1_num_zero_per_iter)}_stage_2_small_model_loss_scale_{stage_2_small_model_loss_scale}_loss_scale_{sparsity_loss_scale}_layer_dropout_{layer_dropout}_num_recog_mods_32_peak_lr_{peak_lr}",
                                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                                  batch_size=10000 * 160, network_args=network_args, kwargs={})
                                    )

                                network_args["recog_num_mods"] = 16
                                system.add_experiment_configs(
                                    f"batch_10k_zeroout_val_{abs(zeroout_val)}_joint_train_after_selection_iter_{len(stage_1_num_zero_per_iter)}_stage_2_small_model_loss_scale_{stage_2_small_model_loss_scale}_loss_scale_{sparsity_loss_scale}_layer_dropout_{layer_dropout}_num_recog_mods_16_peak_lr_{peak_lr}",
                                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                                  batch_size=10000 * 160, network_args=network_args, kwargs={})
                                    )

    for peak_lr in [4e-4]:
        for stage_1_layer_dropout_on_large in [False]:
            for layer_dropout in [0.3, 0.4]:
            # for layer_dropout in [0.3]:
            #     for stage_1_num_steps_per_iter, stage_1_num_zero_per_iter in [([80, 70], [16, 32])]:
                for stage_1_num_steps_per_iter, stage_1_num_zero_per_iter in [([80,70], [16,32]), ([60]+[30]*3,[8,16,24,32]), ([30]+[18]*7,[4,8,12,16,20,24,28,32])]:
                    for zeroout_val in [-3]:
                        stage_1_expected_sparsity_per_iter = [n/48 for n in stage_1_num_zero_per_iter]
                        stage_1_num_steps_per_iter = [n*1400 for n in stage_1_num_steps_per_iter]
                        for sparsity_loss_scale in [5]:
                            for stage_2_small_model_loss_scale in [0.3]:
                                stage_args = {
                                    "stage_1_layer_dropout_on_large": stage_1_layer_dropout_on_large,
                                    "stage_1_num_steps_per_iter": stage_1_num_steps_per_iter,
                                    "stage_1_num_zero_per_iter": stage_1_num_zero_per_iter,
                                    "stage_1_expected_sparsity_per_iter": stage_1_expected_sparsity_per_iter,
                                    "gate_activation": "sigmoid",
                                    "stage_2_small_model_loss_scale": stage_2_small_model_loss_scale,
                                    "zeroout_val": zeroout_val
                                }
                                if stage_2_small_model_loss_scale == 0.3:
                                    del stage_args["stage_2_small_model_loss_scale"]
                                network_args = {"small_model_num_mods": 16, "medium_model_num_mods": 32, "layer_dropout": layer_dropout, "recog_num_mods": 48,
                                                "stage_args": stage_args, "sparsity_loss_scale": sparsity_loss_scale}

                                network_args.update({"time_max_mask_per_n_frames": 25,
                                                "freq_max_num_masks": 5,
                                                "vgg_act": "relu",
                                                "dropout": 0.1,
                                                "num_layers": 12})
                                peak_lr_dict = {
                                    "initial_lr": peak_lr/100,
                                    "peak_lr": peak_lr,
                                }
                                str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
                                str_layer_dropout = str(layer_dropout).replace("-", "_").replace(".", "_")
                                system.add_experiment_configs(
                                    f"batch_10k_zeroout_val_{abs(zeroout_val)}_joint_train_after_selection_iter_{len(stage_1_num_zero_per_iter)}_stage_2_small_model_loss_scale_{stage_2_small_model_loss_scale}_loss_scale_{sparsity_loss_scale}_layer_dropout_{layer_dropout}_num_recog_mods_48_peak_lr_{peak_lr}",
                                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                                  batch_size=10000 * 160, network_args=network_args, kwargs={})
                                    )

                                network_args["recog_num_mods"] = 32
                                system.add_experiment_configs(
                                    f"batch_10k_zeroout_val_{abs(zeroout_val)}_joint_train_after_selection_iter_{len(stage_1_num_zero_per_iter)}_stage_2_small_model_loss_scale_{stage_2_small_model_loss_scale}_loss_scale_{sparsity_loss_scale}_layer_dropout_{layer_dropout}_num_recog_mods_32_peak_lr_{peak_lr}",
                                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                                  batch_size=10000 * 160, network_args=network_args, kwargs={})
                                    )

                                network_args["recog_num_mods"] = 16
                                system.add_experiment_configs(
                                    f"batch_10k_zeroout_val_{abs(zeroout_val)}_joint_train_after_selection_iter_{len(stage_1_num_zero_per_iter)}_stage_2_small_model_loss_scale_{stage_2_small_model_loss_scale}_loss_scale_{sparsity_loss_scale}_layer_dropout_{layer_dropout}_num_recog_mods_16_peak_lr_{peak_lr}",
                                    get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                                  batch_size=10000 * 160, network_args=network_args, kwargs={})
                                    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

