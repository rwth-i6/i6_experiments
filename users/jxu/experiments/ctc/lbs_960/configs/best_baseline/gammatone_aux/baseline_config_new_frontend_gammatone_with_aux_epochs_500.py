import copy
import os

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.baseline import conformer_ctc_d_model_512_num_layers_12_new_frontend_with_aux_loss as conformer_ctc
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
num_subepochs = 500

tools = copy.deepcopy(default_tools_v2)
# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int, network_args:dict) -> ReturnnConfig:
    network_args["num_inputs"] = 50
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
        num_inputs=50,
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
        final_lr=1e-08,
        batch_size=batch_size,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
        train_data_config: dict,
        dev_data_config: dict,
        lr: dict,
        network_args: dict,
        batch_size: int = 36000
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "network_args": network_args}
    train_generator_kwargs = copy.deepcopy(generator_kwargs)
    train_generator_kwargs['network_args']["recog_num_layer"] = 12
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
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=11)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[num_subepochs],
        prior_scales=[0.5],
        lm_scales=[1.0],
        feature_type=FeatureType.GAMMATONE,
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

    for peak_lr in [4e-4, 5e-4]:
        for aux_losses  in [{"12":1, "6":0.3}]:
            network_args = {
                            "aux_losses": aux_losses,
                            "recog_num_layer": 12}
            peak_lr_dict = {
                "initial_lr": peak_lr/100,
                "peak_lr": peak_lr,
            }
            str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
            system.add_experiment_configs(
                f"recog_num_layers_12_peak_lr_{str_peak_lr}_num_aux_losses_{len(aux_losses)}_batch_15000_wei_hyper_new_frontend",
                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                              batch_size=15000, network_args=network_args)
            )
            network_args["recog_num_layer"] = 6
            system.add_experiment_configs(
                f"recog_num_layers_6_peak_lr_{str_peak_lr}_num_aux_losses_{len(aux_losses)}_batch_15000_wei_hyper_new_frontend",
                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                              batch_size=15000, network_args=network_args)
                )

        for aux_losses  in [{"12":1, "6":0.3}, {"12":1, "8":0.3, "4":0.3}]:
            network_args = {
                            "aux_losses": aux_losses,
                            "recog_num_layer": 12}
            peak_lr_dict = {
                "initial_lr": peak_lr/100,
                "peak_lr": peak_lr,
            }
            str_peak_lr = str(peak_lr).replace("-", "_").replace(".", "_")
            system.add_experiment_configs(
                f"recog_num_layers_12_peak_lr_{str_peak_lr}_num_aux_losses_{len(aux_losses)}_batch_15000_wei_hyper_new_frontend",
                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                              batch_size=15000, network_args=network_args)
            )
            network_args["recog_num_layer"] = 8
            system.add_experiment_configs(
                f"recog_num_layers_8_peak_lr_{str_peak_lr}_num_aux_losses_{len(aux_losses)}_batch_15000_wei_hyper_new_frontend",
                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                              batch_size=15000, network_args=network_args)
            )
            network_args["recog_num_layer"] = 4
            system.add_experiment_configs(
                f"recog_num_layers_4_peak_lr_{str_peak_lr}_num_aux_losses_{len(aux_losses)}_batch_15000_wei_hyper_new_frontend",
                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr_dict,
                                              batch_size=15000, network_args=network_args)
            )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

