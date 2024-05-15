import copy
import os
from i6_core.returnn.config import ReturnnConfig

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.tedlium2.ctc_data import get_tedlium2_data_dumped_labels
from i6_experiments.users.berger.pytorch.models import conformer_ctc
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.util import default_tools_v2

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
# num_subepochs = 250
num_subepochs = 1

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def returnn_config_generator(variant: ConfigVariant, train_data_config: dict, dev_data_config: dict) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v2(num_inputs=50, num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.RECOG:
        extra_config["model_outputs"] = {
            "log_probs": {
                "dim": num_outputs,
            }
        }

    extra_config["preload_from_files"] = {
        "base": {
            "init_for_train": True,
            "filename": tk.Path(
                "/work/asr4/berger/sisyphus_work_dirs/tedlium2/20230602_rescale_baselines/i6_core/returnn/training/ReturnnTrainingJob.yEnOhxiP8CgO/output/models/epoch.250.pt"
            ),
        }
    }

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=50,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[conformer_ctc.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=0.0,
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=60,
        # initial_lr=2.2e-05,
        # peak_lr=2.2e-04,
        peak_lr=0.0,
        # final_lr=1e-08,
        batch_size=12000,
        accum_grad=3,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
) -> ReturnnConfigs[ReturnnConfig]:
    generator_kwargs = {"train_data_config": train_data_config, "dev_data_config": dev_data_config}
    return ReturnnConfigs(
        train_config=returnn_config_generator(variant=ConfigVariant.TRAIN, **generator_kwargs),
        prior_config=returnn_config_generator(variant=ConfigVariant.PRIOR, **generator_kwargs),
        recog_configs={"recog": returnn_config_generator(variant=ConfigVariant.RECOG, **generator_kwargs)},
    )


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_data_dumped_labels(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.GAMMATONE_16K,
    )

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        # epochs=[160, num_subepochs],
        epochs=[1],
        prior_scales=[0.5],
        lm_scales=[1.1],
        feature_type=FeatureType.GAMMATONE_16K,
        search_stats=True,
    )

    # ********** System **********

    # tools.returnn_root = tk.Path("/u/berger/repositories/MiniReturnn")
    # tools.rasr_binary_path = tk.Path(
    #     "/u/berger/repositories/rasr_versions/gen_seq2seq_onnx_apptainer/arch/linux-x86_64-standard"
    # )
    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    system.add_experiment_configs(
        "Conformer_CTC", get_returnn_config_collection(data.train_data_config, data.cv_data_config)
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
