import copy
import os
from pathlib import Path
from typing import List, Union

import i6_core.rasr as rasr
import optuna
import torch
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules
from i6_experiments.users.berger.corpus.tedlium2.ctc_data import get_tedlium2_pytorch_data
from i6_experiments.users.berger.pytorch.models import conformer_ctc
from i6_experiments.users.berger.recipe.returnn.optuna_config import OptunaReturnnConfig
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, CustomStepKwargs, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.optuna_returnn_seq2seq_system import OptunaReturnnSeq2SeqSystem
from i6_experiments.users.berger.util import default_tools_v2
from sisyphus import gs, tk

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

storage_path = Path(__file__).parent.parent / "optuna_studies" / "storage.db"
storage = f"sqlite:///{storage_path.as_posix()}"

num_outputs = 79
num_subepochs = 250

tools = copy.deepcopy(default_tools_v2)

tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/onnx/arch/linux-x86_64-standard")
# tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")


# ********** Return Config generators **********


def tune_specaugment(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    model_config.specaugment_cfg.max_feature_mask_num = trial.suggest_int("max_feature_num", 1, 7, step=2)
    model_config.specaugment_cfg.max_feature_mask_num = trial.suggest_int("max_feature", 5, 10, step=5)
    model_config.specaugment_cfg.max_time_mask_num = trial.suggest_int("max_time_num", 0, 25, step=5)
    model_config.specaugment_cfg.max_time_mask_size = trial.suggest_int("max_time", 10, 20, step=5)

    return {}


def tune_model_size(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    num_heads = trial.suggest_int("num_att_heads", 4, 8, step=2)
    lin_size = num_heads * 64

    model_config.conformer_cfg.frontend.cfg.out_features = lin_size
    model_config.conformer_cfg.block_cfg.ff_cfg.input_dim = lin_size
    model_config.conformer_cfg.block_cfg.ff_cfg.hidden_dim = lin_size * 4
    model_config.conformer_cfg.block_cfg.conv_cfg.channels = lin_size
    model_config.conformer_cfg.block_cfg.conv_cfg.norm = torch.nn.BatchNorm1d(num_features=lin_size, affine=False)
    model_config.conformer_cfg.block_cfg.mhsa_cfg.input_dim = lin_size
    model_config.conformer_cfg.block_cfg.mhsa_cfg.num_att_heads = num_heads

    return {}


def tune_frontend(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    model_config.conformer_cfg.frontend.cfg.conv1_channels = trial.suggest_categorical("conv1_channels", [32, 64])
    model_config.conformer_cfg.frontend.cfg.conv2_channels = trial.suggest_categorical("conv2_channels", [32, 64])
    model_config.conformer_cfg.frontend.cfg.conv3_channels = trial.suggest_categorical("conv3_channels", [32, 64])
    model_config.conformer_cfg.frontend.cfg.conv4_channels = trial.suggest_categorical("conv4_channels", [32, 64])

    model_config.conformer_cfg.frontend.cfg.pool1_kernel_size = (2, trial.suggest_int("pool1_f", 1, 2))
    model_config.conformer_cfg.frontend.cfg.pool2_kernel_size = (2, trial.suggest_int("pool2_f", 1, 2))

    return {}


def tune_oclr_schedule(trial: optuna.Trial, model_config: conformer_ctc.ConformerCTCConfig) -> dict:
    peak_lr = trial.suggest_float("peak_lr", 1e-04, 1e-03, log=True)
    initial_lr = peak_lr / trial.suggest_categorical("init_lr_factor", [5, 10, 20])
    return {
        "initial_lr": initial_lr,
        "peak_lr": peak_lr
    }


tuning_functions = {
    "specaugment": tune_specaugment,
    "model_size": tune_model_size,
    "frontend": tune_frontend,
    "oclr_schedule": tune_oclr_schedule,
}


def returnn_config_generator(
    trial: optuna.Trial,
    tuning_names: List[str],
    variant: ConfigVariant,
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfig:
    model_config = copy.deepcopy(conformer_ctc.get_default_config_v2(num_inputs=50, num_outputs=num_outputs))

    tuning_kwargs = {}
    for tuning_name in tuning_names:
        tuning_kwargs.update(tuning_functions[tuning_name](trial, model_config))

    extra_config = {
        "train": copy.deepcopy(train_data_config),
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {"classes": {"dim": num_outputs}}

    kwargs = {
        "num_epochs": num_subepochs,
        "keep_last_n": 1,
        "keep_best_n": 1,
        "num_inputs": 50,
        "num_outputs": num_outputs,
        "target": "targets",
        "extern_data_config": True,
        "extra_python": [conformer_ctc.get_serializer(model_config, variant)],
        "backend": Backend.PYTORCH,
        "grad_noise": 0.0,
        "grad_clip": 100.0,
        "schedule": LearningRateSchedules.OCLR,
        "initial_lr": 1e-05,
        "peak_lr": 3e-04,
        "final_lr": 1e-06,
        "batch_size": 10000,
        "use_chunking": False,
        "extra_config": extra_config,
    }

    kwargs.update(tuning_kwargs)

    return get_returnn_config(**kwargs)


def get_returnn_config_collection(
    tuning_names: List[str],
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfigs[OptunaReturnnConfig]:
    generator_kwargs = {
        "tuning_names": tuning_names,
        "train_data_config": train_data_config,
        "dev_data_config": dev_data_config,
    }
    return ReturnnConfigs(
        train_config=OptunaReturnnConfig(
            returnn_config_generator, {"variant": ConfigVariant.TRAIN, **generator_kwargs}
        ),
        prior_config=OptunaReturnnConfig(
            returnn_config_generator, {"variant": ConfigVariant.PRIOR, **generator_kwargs}
        ),
        recog_configs={
            "recog": OptunaReturnnConfig(
                returnn_config_generator, {"variant": ConfigVariant.RECOG, **generator_kwargs}
            ),
        },
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_pytorch_data(
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    tools.rasr_binary_path = tk.Path(
        "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    )
    system = OptunaReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC_tune",
        get_returnn_config_collection(
            ["specaugment", "model_size", "frontend", "oclr_schedule"], data.train_data_config, data.cv_data_config
        ),
    )

    # ********** Steps **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
        study_storage=storage,
        score_key="dev_loss_CTC",
        num_trials=150,
        num_parallel=10,
        backend=Backend.PYTORCH,
    )
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[num_subepochs],
        prior_scales=[0.9],
        lm_scales=[1.1],
        #trial_nums=list(range(150)) + ["best"],
        trial_nums=[34, 48, 65, 70, 121, 139, "best"],
        backend=Backend.PYTORCH,
        feature_type=FeatureType.GAMMATONE,
    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
