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

tools = copy.deepcopy(default_tools_v2)
tools.rasr_binary_path = tk.Path("/u/berger/repositories/rasr_versions/gen_seq2seq_dev/arch/linux-x86_64-standard")


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, num_subepochs: int, **kwargs
) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v3(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "max_seq_length": {"audio_features": 560000},
        "torch_amp": {"dtype": "bfloat16"},
    }
    if variant == ConfigVariant.RECOG:
        extra_config["extern_data"] = {
            "sources": {"dim": 80, "dtype": "float32"},
        }
        extra_config["model_outputs"] = {
            "log_probs": {
                "dim": num_outputs,
            }
        }

    return get_returnn_config(
        num_epochs=num_subepochs,
        num_inputs=1,
        num_outputs=num_outputs,
        target="classes",
        extra_python=[conformer_ctc.get_serializer(model_config, variant=variant)],
        extern_data_config=True,
        backend=Backend.PYTORCH,
        grad_noise=kwargs.get("grad_noise", 0.0),
        grad_clip=0.0,
        optimizer=Optimizers.AdamW,
        schedule=LearningRateSchedules.OCLR,
        max_seqs=60,
        initial_lr=7e-06,
        peak_lr=7e-04,
        decayed_lr=7e-05,
        final_lr=1e-08,
        batch_size=36000 * 160,
        use_chunking=False,
        extra_config=extra_config,
    )


def get_returnn_config_collection(
    train_data_config: dict,
    dev_data_config: dict,
    num_subepochs: int,
    **kwargs,
) -> ReturnnConfigs[ReturnnConfig]:
    return ReturnnConfigs(
        train_config=returnn_config_generator(
            variant=ConfigVariant.TRAIN,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            num_subepochs=num_subepochs,
            **kwargs,
        ),
        prior_config=returnn_config_generator(
            variant=ConfigVariant.PRIOR,
            train_data_config=train_data_config,
            dev_data_config=dev_data_config,
            num_subepochs=num_subepochs,
            **kwargs,
        ),
        recog_configs={
            "recog": returnn_config_generator(
                variant=ConfigVariant.RECOG,
                train_data_config=train_data_config,
                dev_data_config=dev_data_config,
                num_subepochs=num_subepochs,
                **kwargs,
            )
        },
    )


def run_exp(num_subepochs: int = 250) -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_tedlium2_data_dumped_labels(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=[ep for ep in [80, 160, 320, 640, 1280, num_subepochs] if ep <= num_subepochs],
        prior_scales=[0.5],
        lm_scales=[1.1],
        feature_type=FeatureType.SAMPLES,
        search_stats=True,
        seq2seq_v2=True,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.ctc_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    for grad_noise in [0.0, 0.1]:
        system.add_experiment_configs(
            f"Conformer_CTC_{num_subepochs}-epochs_gn-{grad_noise}",
            get_returnn_config_collection(
                train_data_config=data.train_data_config,
                dev_data_config=data.cv_data_config,
                num_subepochs=num_subepochs,
                grad_noise=grad_noise,
            ),
        )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(num_subepochs=250), update_structure=True)
    summary_report.merge_report(run_exp(num_subepochs=500), update_structure=True)
    summary_report.merge_report(run_exp(num_subepochs=1000), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
