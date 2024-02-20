import copy
import os
from typing import Dict, Tuple

import i6_core.rasr as rasr
from i6_core.recognition import Hub5ScoreJob
from i6_core.returnn import Checkpoint
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
import i6_experiments.users.berger.network.models.context_1_transducer_tinaconf as transducer_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, FeatureType
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from i6_experiments.users.berger.corpus.switchboard.viterbi_transducer_data import get_switchboard_data
from .config_01c_ctc_blstm_wei_data import py as py_ctc_blstm
from sisyphus import gs, tk

tools = copy.deepcopy(default_tools)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 88


# ********** Return Config **********


def generate_returnn_config(
    train: bool,
    *,
    train_data_config: dict,
    dev_data_config: dict,
    **kwargs,
) -> ReturnnConfig:
    if train:
        (
            network_dict,
            extra_python,
        ) = transducer_model.make_context_1_conformer_transducer(
            num_inputs=40,
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
            },
            decoder_args={
                "dec_mlp_args": {
                    "num_layers": 2,
                    "size": 640,
                    "activation": "tanh",
                    "dropout": 0.1,
                    "l2": 5e-06,
                },
                "combination_mode": "concat",
                "joint_mlp_args": {
                    "num_layers": 1,
                    "size": 1024,
                    "dropout": 0.1,
                    "l2": 5e-06,
                    "activation": "tanh",
                },
            },
            output_args={
                "label_smoothing": kwargs.get("label_smoothing", None),
            },
            loss_boost_scale=kwargs.get("loss_boost_scale", 5.0),
            loss_boost_v2=kwargs.get("loss_boost_v2", False),
        )
    else:
        (
            network_dict,
            extra_python,
        ) = transducer_model.make_context_1_conformer_transducer_recog(
            num_inputs=40,
            num_outputs=num_classes,
            decoder_args={
                "dec_mlp_args": {
                    "num_layers": 2,
                    "size": 640,
                    "activation": "tanh",
                },
                "combination_mode": "concat",
                "joint_mlp_args": {
                    "num_layers": 1,
                    "size": 1024,
                    "activation": "tanh",
                },
            },
        )

    extra_config = {
        "train": train_data_config,
        "dev": dev_data_config,
        "chunking": (
            {
                "data": 400,
                "classes": 100,
            },
            {
                "data": 200,
                "classes": 50,
            },
        ),
    }

    if kwargs.get("model_preload", None) is not None:
        extra_config["preload_from_files"] = {
            "base": {
                "init_for_train": True,
                "ignore_missing": True,
                "filename": kwargs.get("model_preload", None),
            }
        }

    returnn_config = get_returnn_config(
        network=network_dict,
        target="classes",
        num_epochs=300,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        extra_python=extra_python,
        num_inputs=40,
        num_outputs=num_classes,
        extern_target_kwargs={"dtype": "int8" if train else "int32"},
        extern_data_config=True,
        grad_noise=0.0,
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR_STEP,
        initial_lr=1e-03 / 30,
        peak_lr=1e-03,
        final_lr=1e-06,
        n_steps_per_epoch=2850,
        batch_size=12500,
        extra_config=extra_config,
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp(alignments: Dict[str, AlignmentData], name_suffix: str = "") -> Tuple[SummaryReport, Checkpoint]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_switchboard_data(
        tools.returnn_root,
        tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        alignments=alignments,
        use_wei_data=True,
        test_keys=["hub5e01"],
        feature_type=FeatureType.GAMMATONE_8K,
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=300,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.5, 0.6, 0.7],
        epochs=[300],
        # lookahead_options={"scale": 0.5},
        search_parameters={"label-pruning": 12.0},
        feature_type=FeatureType.GAMMATONE_8K,
        reduction_factor=4,
        reduction_subtrahend=0,
    )
    recog_am_args = copy.deepcopy(exp_args.transducer_recog_am_args)
    recog_am_args.update(
        {
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
        }
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    # ********** Returnn Configs **********

    train_config = generate_returnn_config(
        train=True,
        train_data_config=data.train_data_config,
        dev_data_config=data.cv_data_config,
        label_smoothing=None,
        loss_boost_v2=False,
        loss_boost_scale=0.0,
        model_preload=None,
    )
    recog_config = generate_returnn_config(
        train=False,
        train_data_config=data.train_data_config,
        dev_data_config=data.cv_data_config,
    )

    returnn_configs = ReturnnConfigs(
        train_config=train_config,
        recog_configs={"recog": recog_config},
    )

    system.add_experiment_configs(f"Conformer_Transducer_Viterbi_{name_suffix}", returnn_configs)

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=recog_am_args,
    )
    system.setup_scoring(
        scorer_type=Hub5ScoreJob,
        stm_paths={key: tk.Path("/u/corpora/speech/hub5e_00/xml/hub5e_00.stm") for key in data.dev_keys},
        score_kwargs={
            "glm": tk.Path("/u/corpora/speech/hub5e_00/xml/glm"),
        },
    )

    system.run_train_step(**train_args)

    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    train_job = system.get_train_job(f"Conformer_Transducer_Viterbi_{name_suffix}")
    model = train_job.out_checkpoints[300]
    assert isinstance(model, Checkpoint)

    assert system.summary_report
    return system.summary_report, model


def py() -> Tuple[SummaryReport, Dict[str, Checkpoint]]:
    _, alignments_blstm = py_ctc_blstm()

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()
    models = {}

    for align_model_name, alignments in alignments_blstm.items():
        sub_report, model = run_exp(alignments, name_suffix=f"align-{align_model_name}")
        models[align_model_name] = model
        summary_report.merge_report(sub_report, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, models
