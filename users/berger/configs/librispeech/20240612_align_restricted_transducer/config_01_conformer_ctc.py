import copy
import os
from i6_models.config import ModuleFactoryV1
from i6_core.returnn.config import ReturnnConfig

from sisyphus import gs, tk

import i6_core.rasr as rasr
from i6_experiments.users.berger.args.experiments import ctc as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config, Backend
from i6_experiments.users.berger.args.returnn.learning_rates import LearningRateSchedules, Optimizers
from i6_experiments.users.berger.corpus.librispeech.ctc_data import get_librispeech_data_dumped_labels
from i6_experiments.users.berger.pytorch.models import conformer_ctc
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant, FeatureType, ReturnnConfigs
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.util import default_tools_v2
from i6_experiments.users.berger.pytorch.custom_parts.identity import IdentityConfig, IdentityModule

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}

num_outputs = 79
num_subepochs = 1000
sub_checkpoints = [100, 200, 400, 600, 800, 900, 920, 940, 960, 980, 990, 1000]

tools = copy.deepcopy(default_tools_v2)


# ********** Return Config generators **********


def returnn_config_generator(
    variant: ConfigVariant, train_data_config: dict, dev_data_config: dict, num_subepochs: int, **kwargs
) -> ReturnnConfig:
    model_config = conformer_ctc.get_default_config_v4(num_outputs=num_outputs)

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
    }
    if variant == ConfigVariant.TRAIN:
        extra_config["max_seq_length"] = {"audio_features": 560000}
        extra_config["torch_amp"] = {"dtype": "bfloat16"}
    if variant == ConfigVariant.RECOG:
        extra_config["extern_data"] = {
            "data": {"dim": 80, "dtype": "float32"},
        }
        extra_config["model_outputs"] = {
            "log_probs": {
                "dim": num_outputs,
            }
        }
        model_config.feature_extraction = ModuleFactoryV1(IdentityModule, IdentityConfig())

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
        weight_decay=5e-06,
        schedule=LearningRateSchedules.OCLR_STEP_TORCH,
        keep_last_n=1,
        keep_best_n=0,
        keep=sub_checkpoints,
        max_seqs=60,
        initial_lr=kwargs.get("initial_lr", 4e-06),
        peak_lr=kwargs.get("peak_lr", 4e-04),
        decayed_lr=kwargs.get("decay_lr", 4e-05),
        final_lr=1e-08,
        n_steps_per_epoch=1220,
        batch_size=15000 * 160,
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


def run_exp() -> SummaryReport:
    assert tools.returnn_root
    assert tools.returnn_python_exe
    assert tools.rasr_binary_path
    data = get_librispeech_data_dumped_labels(
        num_classes=num_outputs,
        returnn_root=tools.returnn_root,
        returnn_python_exe=tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        add_unknown_phoneme_and_mapping=False,
        use_augmented_lexicon=True,
        feature_type=FeatureType.SAMPLES,
    )

    for data_input in data.data_inputs.values():
        data_input.create_lm_images(tools.rasr_binary_path)

    # ********** Step args **********

    train_args = exp_args.get_ctc_train_step_args(num_epochs=num_subepochs, gpu_mem_rqmt=24)
    recog_args = exp_args.get_ctc_recog_step_args(
        num_classes=num_outputs,
        epochs=sub_checkpoints,
        prior_scales=[0.3],
        lm_scales=[0.9],
        feature_type=FeatureType.LOGMEL_16K,
        search_stats=True,
        seq2seq_v2=True,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        corpus_data=data.data_inputs,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    for initial_lr, peak_lr, decay_lr in [(1e-05, 4e-04, 1e-05), (4e-06, 4e-04, 4e-05)]:
        system.add_experiment_configs(
            f"Conformer_CTC_lr-init-{initial_lr}_peak-{peak_lr}_dec-{decay_lr}",
            get_returnn_config_collection(
                train_data_config=data.train_data_config,
                dev_data_config=data.cv_data_config,
                num_subepochs=num_subepochs,
                initial_lr=initial_lr,
                peak_lr=peak_lr,
                decay_lr=decay_lr,
            ),
        )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)
    # system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    summary_report.merge_report(run_exp(), update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
