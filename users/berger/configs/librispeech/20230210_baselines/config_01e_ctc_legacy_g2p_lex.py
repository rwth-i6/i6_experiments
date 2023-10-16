import copy
import os
from pathlib import Path

import i6_core.rasr as rasr
import optuna
from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data
from i6_experiments.users.berger.rc_helpers.serializers import (
    get_ctc_rc_network_serializer,
)
from i6_experiments.users.berger.recipe.returnn import OptunaReturnnConfig
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    OptunaReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs
from i6_experiments.users.berger.util import default_tools
from sisyphus import gs, tk

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


storage_path = Path(__file__).parent.parent / "optuna_studies" / "storage.db"

num_classes = 79


# ********** Return Config generators **********


def returnn_config_generator(
    trial: optuna.Trial,
    train: bool,
    *,
    loss_corpus: tk.Path,
    loss_lexicon: tk.Path,
    am_args: dict,
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfig:
    network_serializer = get_ctc_rc_network_serializer(
        num_outputs=num_classes,
        returnn_common_root=default_tools.returnn_common_root,
        network_kwargs={
            "train": train,
            "feature_args": {"sample_rate": 16_000},
            "legacy_specaug": True,
            "specaug_args": {
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 5,
            },
            "loss_args": {
                "loss_corpus_path": loss_corpus,
                "loss_lexicon_path": loss_lexicon,
                "am_args": am_args,
                "rasr_binary_path": default_tools.nn_trainer_rasr_binary_path,
            },
            "encoder_type": CodeWrapper("EncoderType.Blstm"),
            "encoder_args": {
                "dim": 512,
                "num_layers": 6,
                "time_reduction": [1, 2, 2],
                "l2": trial.suggest_float("l2", 1e-05, 1e-03, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            },
        },
    )

    returnn_config = get_returnn_config(
        num_epochs=500,
        extra_python=[network_serializer],
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        initial_lr=1e-05,
        peak_lr=trial.suggest_float("peak_lr", 1e-04, 1e-03, log=True),
        final_lr=1e-05,
        batch_size=1_600_000,
        use_chunking=False,
        extra_config={
            "train": train_data_config,
            "dev": dev_data_config,
        },
    )

    return returnn_config


def run_exp() -> SummaryReport:
    data = get_librispeech_data(
        default_tools.returnn_root,
        default_tools.returnn_python_exe,
        add_unknown=False,
        augmented_lexicon=True,
    )

    # ********** Returnn Configs **********

    config_generator_kwargs = {
        "loss_corpus": data.loss_corpus,
        "loss_lexicon": data.loss_lexicon,
        "am_args": exp_args.ctc_loss_am_args,
        "train_data_config": data.train_data_config,
        "dev_data_config": data.cv_data_config,
    }

    train_config = OptunaReturnnConfig(returnn_config_generator, {"train": True, **config_generator_kwargs})
    recog_config = OptunaReturnnConfig(returnn_config_generator, {"train": False, **config_generator_kwargs})

    returnn_configs = ReturnnConfigs(
        train_config=train_config,
        recog_configs={"recog": recog_config},
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_args(
        study_storage=f"sqlite:///{storage_path.as_posix()}",
        num_trials=20,
        num_parallel=5,
        num_epochs=500,
        mem_rqmt=24,
    )

    recog_args = exp_args.get_ctc_recog_args(num_classes)
    align_args = exp_args.get_ctc_align_args(num_classes)

    intermediate_recog_args = copy.deepcopy(recog_args)
    intermediate_recog_args["trial_nums"] = [0]
    intermediate_recog_args["epochs"] = [300, 400, 500, "best"]

    # ********** System **********

    system = OptunaReturnnSeq2SeqSystem(default_tools)

    system.add_experiment_configs("BLSTM_CTC_legacy-specaug_augment-lex", returnn_configs)
    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    system.run_train_step(**train_args)

    system.run_dev_recog_step(**intermediate_recog_args)
    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)
    system.run_align_step(**align_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
