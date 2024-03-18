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
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, FeatureType, SummaryKey
from i6_experiments.users.berger.util import default_tools, recursive_update
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
                "ilm_scale": kwargs.get("ilm_scale", 0.0),
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
        schedule=LearningRateSchedules.OCLR,
        # initial_lr=1e-03 / 30,
        # peak_lr=1e-03,
        initial_lr=1e-05,
        peak_lr=kwargs.get("peak_lr", 4e-04),
        final_lr=1e-06,
        n_steps_per_epoch=3210,
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
        dc_detection=True,
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=300,
        gpu_mem_rqmt=24,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.5],
        epochs=[300],
        # lookahead_options={"scale": 0.5},
        # label_scorer_args={"extra_args": {"blank-label-index": 2}},
        search_parameters={"label-pruning": 14.4},
        feature_type=FeatureType.GAMMATONE_8K,
        reduction_factor=4,
        reduction_subtrahend=0,
        flow_args={"dc_detection": True},
    )
    recog_am_args = copy.deepcopy(exp_args.transducer_recog_am_args)
    recog_am_args.update(
        {
            # "state_tying": "lookup",
            # "state_tying_file": tk.Path("/work/asr4/berger/dependencies/switchboard/state_tying/wei_mono-eow"),
            "tying_type": "global-and-nonword",
            "nonword_phones": ["[NOISE]", "[VOCALIZEDNOISE]", "[LAUGHTER]"],
        }
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.CORPUS,
            SummaryKey.RECOG_NAME,
            SummaryKey.EPOCH,
            SummaryKey.LM,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.INS,
            SummaryKey.DEL,
            SummaryKey.ERR,
        ],
        summary_sort_keys=[SummaryKey.ERR, SummaryKey.CORPUS],
    )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=recog_am_args,
    )
    system.setup_scoring(scorer_type=Hub5ScoreJob)

    # ********** Returnn Configs **********

    for lr in [4e-04, 6e-04, 8e-04]:
        for label_smoothing in [None, 0.2]:
            for loss_boost_scale in [0.0, 5.0]:
                train_config = generate_returnn_config(
                    train=True,
                    train_data_config=data.train_data_config,
                    dev_data_config=data.cv_data_config,
                    peak_lr=lr,
                    label_smoothing=label_smoothing,
                    loss_boost_v2=False,
                    loss_boost_scale=loss_boost_scale,
                    model_preload=None,
                )

                returnn_configs = ReturnnConfigs(
                    train_config=train_config,
                    recog_configs={
                        f"recog_ilm-{ilm_scale}": generate_returnn_config(
                            train=False,
                            ilm_scale=ilm_scale,
                            train_data_config=data.train_data_config,
                            dev_data_config=data.cv_data_config,
                        )
                        for ilm_scale in [0.0, 0.1, 0.2]
                    },
                )
                name = f"Conformer_Transducer_Viterbi_{name_suffix}_lr-{lr}"
                if label_smoothing:
                    name += f"_ls-{label_smoothing}"
                if loss_boost_scale:
                    name += f"_loss-boost"

                system.add_experiment_configs(name, returnn_configs)

    system.run_train_step(**train_args)

    system.run_dev_recog_step(**recog_args)
    # system.run_test_recog_step(**recog_args)

    if "am-1.0" in name_suffix:
        for bp in [0.0, 0.5, 1.0]:
            recursive_update(
                recog_args,
                {
                    "epochs": [300],
                    "lm_scales": [0.4, 0.5, 0.6, 0.7, 0.8],
                    "search_parameters": {"blank-label-penalty": bp} if bp else {},
                },
            )

            system.run_dev_recog_step(
                exp_names=["Conformer_Transducer_Viterbi_align-BLSTM_CTC_am-1.0_lr-0.0008"],
                recog_descriptor=f"bp-{bp}",
                **recog_args,
            )

        for lp in [12.0, 14.0, 16.0, 18.0, 20.0]:
            recursive_update(
                recog_args,
                {
                    "epochs": [300],
                    "lm_scales": [0.8],
                    "search_parameters": {"blank-label-penalty": 1.0, "label-pruning": lp},
                },
            )

            system.run_dev_recog_step(
                exp_names=["Conformer_Transducer_Viterbi_align-BLSTM_CTC_am-1.0_lr-0.0008"],
                recog_exp_names={"Conformer_Transducer_Viterbi_align-BLSTM_CTC_am-1.0_lr-0.0008": ["recog_ilm-0.1"]},
                recog_descriptor=f"lp-{lp}",
                **recog_args,
            )

    recursive_update(
        recog_args, {"epochs": [300], "lm_scales": [0.8], "search_parameters": {"blank-label-penalty": 1.0}}
    )
    system.run_dev_recog_step(recog_exp_names={key: ["recog_ilm-0.1"] for key in system.get_exp_names()}, **recog_args)
    system.run_test_recog_step(recog_exp_names={key: ["recog_ilm-0.1"] for key in system.get_exp_names()}, **recog_args)

    train_job = system.get_train_job(f"Conformer_Transducer_Viterbi_{name_suffix}_lr-0.0008")
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
        if "am-1.0" not in align_model_name:
            continue
        sub_report, model = run_exp(alignments, name_suffix=f"align-{align_model_name}")
        models[align_model_name] = model
        summary_report.merge_report(sub_report, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, models
