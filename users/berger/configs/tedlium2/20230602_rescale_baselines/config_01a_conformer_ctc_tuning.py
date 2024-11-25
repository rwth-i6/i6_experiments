import copy
import os
from pathlib import Path
from typing import List

import i6_core.rasr as rasr
import optuna
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import Backend, get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.ctc_data import get_tedlium2_pytorch_data
from i6_experiments.users.berger.pytorch.models import conformer_ctc
from i6_experiments.users.berger.recipe.returnn.optuna_config import OptunaReturnnConfig
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
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
    model_config.specaugment.cfg.time_max_mask_per_n_frames = trial.suggest_int(
        "time_max_mask_per_n_frames", 10, 40, step=10
    )
    model_config.specaugment.cfg.time_mask_max_size = trial.suggest_int("time_mask_max_size", 10, 30, step=5)

    freq_max_num_masks = trial.suggest_int("freq_max_num_masks", 3, 9, step=2)
    model_config.specaugment.cfg.freq_max_num_masks = freq_max_num_masks
    model_config.specaugment.cfg.freq_mask_max_size = 50 // freq_max_num_masks

    return {}


def tune_oclr_schedule(trial: optuna.Trial, _: conformer_ctc.ConformerCTCConfig) -> dict:
    peak_lr = trial.suggest_float("peak_lr", 1e-04, 1e-03, log=True)
    initial_lr = peak_lr / 10
    return {"initial_lr": initial_lr, "peak_lr": peak_lr}


tuning_functions = {
    "specaugment": tune_specaugment,
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
        "extra_python": [conformer_ctc.get_serializer(model_config, variant, in_dim=50)],
        "backend": Backend.PYTORCH,
        "grad_noise": 0.0,
        "grad_clip": 0.0,
        "optimizer": Optimizers.AdamW,
        "schedule": LearningRateSchedules.OCLR,
        "initial_lr": 2.2e-05,
        "peak_lr": 2.2e-04,
        "final_lr": 1e-08,
        "batch_size": 18000,
        "accum_grad": 2,
        "max_seqs": 60,
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
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC_tune",
        get_returnn_config_collection(["specaugment", "oclr_schedule"], data.train_data_config, data.cv_data_config),
    )

    # ********** Steps **********

    train_args = exp_args.get_ctc_train_step_args(
        num_epochs=num_subepochs,
        study_storage=storage,
        score_key="dev_loss_CTC",
        num_trials=20,
        num_parallel=10,
        backend=Backend.PYTORCH,
    )
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[240, num_subepochs],
        prior_scales=[0.5],
        lm_scales=[1.1],
        trial_nums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # ["best"]
        backend=Backend.PYTORCH,
        feature_type=FeatureType.GAMMATONE_16K,
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
