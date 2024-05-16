import copy

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.baseline import \
    conformer_ctc_d_model_512_num_layers_12_new_frontend_raw_wave as conformer_ctc
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
num_subepochs = 700

tools = copy.deepcopy(default_tools_v2)
# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int, network_args:dict, kwargs:dict) -> ReturnnConfig:
    network_args["num_inputs"] = 80
    network_args["num_outputs"] = num_outputs
    model_config = conformer_ctc.get_default_config_v1(**network_args)

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
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
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
        prior_scales=[0.5],
        lm_scales=[1.0],
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

    for peak_lr in [3e-4,4e-4,5e-4,6e-4]:
        for time_max_mask_per_n_frames in [25]:
            for freq_max_num_masks in [5]:
                for vgg_act in ["relu"]:
                    for dropout in [0.1]:
                        for num_layers in [12]:
                            if num_layers != 12:
                                dropout = 0.1
                            network_args = {"time_max_mask_per_n_frames": time_max_mask_per_n_frames,
                                            "freq_max_num_masks": freq_max_num_masks,
                                            "vgg_act": vgg_act,
                                            "dropout": dropout,
                                            "num_layers": num_layers}
                            peak_lr_dict = {
                                "initial_lr": peak_lr/100,
                                "peak_lr": peak_lr,
                            }
                            str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
                            str_dropout = str(dropout).replace(".", "_")
                            system.add_experiment_configs(
                                f"num_layers_{num_layers}_subepochs_700_peak_lr_{str_peak_lr}_dropout_{str_dropout}_batch_15000_wei_hyper_new_frontend",
                                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                              batch_size=15000 * 160, network_args=network_args, kwargs={})
                                )

        for time_max_mask_per_n_frames in [25]:
            for freq_max_num_masks in [5]:
                for vgg_act in ["relu"]:
                    for dropout in [0.1]:
                        network_args = {"time_max_mask_per_n_frames": time_max_mask_per_n_frames,
                                        "freq_max_num_masks": freq_max_num_masks,
                                        "vgg_act": vgg_act,
                                        "dropout": dropout}
                        peak_lr_dict = {
                            "initial_lr": 1e-5,
                            "peak_lr": peak_lr,
                            "final_lr": 1e-5
                        }
                        str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
                        str_dropout = str(dropout).replace(".", "_")
                        system.add_experiment_configs(
                            f"wei_lr_scheduler_subepochs_700_peak_lr_{str_peak_lr}_dropout_{str_dropout}_batch_15000_wei_hyper_new_frontend",
                            get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                                          batch_size=15000 * 160, network_args=network_args,kwargs={"cycle_epoch":250})
                        )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

