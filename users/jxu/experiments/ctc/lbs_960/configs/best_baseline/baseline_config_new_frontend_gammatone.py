import copy
import os

from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.rasr as rasr
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
# from i6_experiments.users.berger.pytorch.models import conformer_ctc
# from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks import conformer_ctc_downsample_4 as conformer_ctc
from i6_experiments.users.jxu.experiments.ctc.lbs_960.pytorch_networks.baseline import conformer_ctc_d_model_512_num_layers_12_new_frontend as conformer_ctc
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
# num_subepochs = 500

tools = copy.deepcopy(default_tools_v2)
# tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
SCTK_BINARY_PATH = compile_sctk()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LBS_DEFAULT_SCTK_BINARY_PATH"


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, lr: dict,
                             batch_size: int, num_subepochs:int) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v1(num_inputs=50, num_outputs=num_outputs)

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
        max_seqs=128,
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
        batch_size: int = 36000,
        num_subepochs: int=500
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config, "lr": lr,
                        "batch_size": batch_size, "num_subepochs": num_subepochs}
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def run_lbs_960_torch_conformer_wei_hyper() -> SummaryReport:
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

    # for num_subepochs in [500]:
    #     for peak_lr in [3e-4, 4e-4, 5e-4, 6e-4]:
    # # for num_subepochs in [700]:
    # #     for peak_lr in [3e-4]:
    #         peak_lr = {
    #             "initial_lr": peak_lr/100,
    #             "peak_lr": peak_lr,
    #         }
    #         exp_name = f"subepochs_{num_subepochs}_peark_lr_{peak_lr['peak_lr']}_batch_15000_wei_hyper_new_frontend"
    #         exp_name=exp_name.replace("-", "_")
    #         exp_name=exp_name.replace(".", "_")
    #         system.add_experiment_configs(
    #             exp_name,
    #             get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr, batch_size=15000, num_subepochs=num_subepochs)
    #         )
    #
    #         train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=11)
    #         recog_args = exp_args.get_ctc_recog_step_args(
    #             num_classes=num_outputs,
    #             epochs=[500],
    #             prior_scales=[0.5, 0.6,0.7],
    #             lm_scales=[0.9, 1],
    #             feature_type=FeatureType.GAMMATONE,
    #         )
    #
    #         system.run_train_step(**train_args:q
    #         )
    #         system.run_dev_recog_step(**recog_args)

    for num_subepochs in [700]:
        for peak_lr in [3e-4, 4e-4, 5e-4, 6e-4]:
    # for num_subepochs in [700]:
    #     for peak_lr in [3e-4]:
            peak_lr = {
                "initial_lr": peak_lr/100,
                "peak_lr": peak_lr,
            }
            exp_name = f"subepochs_{num_subepochs}_peark_lr_{peak_lr['peak_lr']}_batch_15000_wei_hyper_new_frontend"
            exp_name=exp_name.replace("-", "_")
            exp_name=exp_name.replace(".", "_")
            system.add_experiment_configs(
                exp_name,
                get_returnn_config_collection(data.train_data_config, data.cv_data_config, lr=peak_lr, batch_size=15000, num_subepochs=num_subepochs)
            )

            train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=11)
            recog_args = exp_args.get_ctc_recog_step_args(
                num_classes=num_outputs,
                epochs=[700],
                prior_scales=[0.5],
                lm_scales=[1],
                feature_type=FeatureType.GAMMATONE,
            )

            system.run_train_step(**train_args)
            system.run_dev_recog_step(**recog_args)

    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report

