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
import i6_experiments.users.berger.network.models.context_1_transducer as transducer_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import (
    ReturnnConfigs,
    FeatureType,
    SummaryKey,
)
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from i6_experiments.users.berger.corpus.switchboard.viterbi_transducer_data import (
    get_switchboard_data,
)
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
    specaug_v2 = kwargs.get("specaug_v2", False)

    if specaug_v2:
        specaug_args = {
            "min_reps_time": 0,
            "max_reps_time": 20,
            "max_len_time": 20,
            "min_reps_feature": 0,
            "max_reps_feature": 1,
            "max_len_feature": 15,
        }
    else:
        specaug_args = {
            "max_time_num": 1,
            "max_time": 15,
            "max_feature_num": 5,
            "max_feature": 4,
        }

    if train:
        (network_dict, extra_python,) = transducer_model.make_context_1_conformer_transducer(
            num_outputs=num_classes,
            specaug_args=specaug_args,
            conformer_args={
                "num_blocks": 12,
                "size": 512,
                "dropout": 0.1,
                "l2": 5e-06,
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
            specaug_v2=specaug_v2,
        )
    else:
        (network_dict, extra_python,) = transducer_model.make_context_1_conformer_transducer_recog(
            num_outputs=num_classes,
            conformer_args={
                "num_blocks": 12,
                "size": 512,
                "dropout": 0.1,
                "l2": 5e-06,
            },
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
                "data": 256,
                "classes": 64,
            },
            {
                "data": 128,
                "classes": 32,
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
        initial_lr=1e-05,
        peak_lr=kwargs.get("peak_lr", 8e-04),
        final_lr=1e-06,
        n_steps_per_epoch=3210,
        batch_size=15000,
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

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=300,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.4, 0.5, 0.6, 0.7, 0.8],
        epochs=[300],
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

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=recog_am_args,
    )
    system.setup_scoring(scorer_type=Hub5ScoreJob)

    # ********** Returnn Configs **********

    for lr in [8e-04]:
        for label_smoothing in [None, 0.2]:
            for loss_boost_scale in [0.0, 5.0]:
                train_config = generate_returnn_config(
                    train=True,
                    train_data_config=data.train_data_config,
                    dev_data_config=data.cv_data_config,
                    peak_lr=lr,
                    label_smoothing=label_smoothing,
                    loss_boost_v2=True,
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
                        for ilm_scale in [0.0, 0.1, 0.2, 0.3]
                    },
                )
                name = f"Conformer_Transducer_Viterbi_wei-data_{name_suffix}_lr-{lr}"
                if label_smoothing:
                    name += f"_ls-{label_smoothing}"
                if loss_boost_scale:
                    name += "_loss-boost"

                system.add_experiment_configs(name, returnn_configs)

    system.add_experiment_configs(
        f"Conformer_Transducer_Viterbi_wei-data_specaug-v2_{name_suffix}",
        ReturnnConfigs(
            train_config=generate_returnn_config(
                train=True,
                train_data_config=data.train_data_config,
                dev_data_config=data.cv_data_config,
                peak_lr=8e-04,
                label_smoothing=None,
                loss_boost_scale=0.0,
                model_preload=None,
                specaug_v2=True,
            ),
            recog_configs={
                f"recog_ilm-{ilm_scale}": generate_returnn_config(
                    train=False,
                    ilm_scale=ilm_scale,
                    train_data_config=data.train_data_config,
                    dev_data_config=data.cv_data_config,
                )
                for ilm_scale in [0.0, 0.2]
            },
        ),
    )

    system.run_train_step(**train_args)
    system.run_dev_recog_step(**recog_args)

    recog_args.update(
        {
            "lm_scales": [0.6, 0.7],
            "epochs": [
                213,
                249,
                261,
                279,
                283,
                283,
                284,
                285,
                286,
                289,
                291,
                297,
                298,
                299,
                300,
            ],
        }
    )
    system.run_dev_recog_step(
        exp_names=[f"Conformer_Transducer_Viterbi_wei-data_{name_suffix}_lr-0.0008"],
        recog_exp_names=["recog_ilm-0.1", "recog_ilm-0.2"],
        **recog_args,
    )

    recog_args.update(
        {
            "epochs": [
                213,
                249,
                261,
                267,
                273,
                276,
                279,
                280,
                281,
                282,
                283,
                284,
                285,
                286,
                289,
                291,
                298,
                298,
                299,
                300,
            ]
        }
    )
    system.run_dev_recog_step(
        exp_names=[f"Conformer_Transducer_Viterbi_wei-data_{name_suffix}_lr-0.0008_loss-boost"],
        recog_exp_names=["recog_ilm-0.1", "recog_ilm-0.2"],
        **recog_args,
    )

    recog_args.update(
        {
            "epochs": [
                213,
                249,
                261,
                267,
                273,
                279,
                284,
                285,
                289,
                291,
                297,
                298,
                299,
                300,
            ]
        }
    )
    system.run_dev_recog_step(
        exp_names=[f"Conformer_Transducer_Viterbi_wei-data_{name_suffix}_lr-0.0008_ls-0.2"],
        recog_exp_names=["recog_ilm-0.1", "recog_ilm-0.2"],
        **recog_args,
    )

    recog_args.update(
        {
            "epochs": [
                213,
                249,
                261,
                267,
                273,
                274,
                279,
                280,
                281,
                282,
                283,
                284,
                285,
                286,
                289,
                291,
                297,
                298,
                299,
                300,
            ]
        }
    )
    system.run_dev_recog_step(
        exp_names=[f"Conformer_Transducer_Viterbi_wei-data_{name_suffix}_lr-0.0008_ls-0.2_loss-boost"],
        recog_exp_names=["recog_ilm-0.1", "recog_ilm-0.2"],
        **recog_args,
    )

    train_job = system.get_train_job(f"Conformer_Transducer_Viterbi_wei-data_{name_suffix}_lr-0.0008")
    model = train_job.out_checkpoints[298]
    assert isinstance(model, Checkpoint)

    assert system.summary_report
    return system.summary_report, model


def py() -> Tuple[SummaryReport, Checkpoint]:
    _, alignments_blstm = py_ctc_blstm()
    alignments_blstm = alignments_blstm["BLSTM_CTC_wei-data_am-1.0"]

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    sub_report, model = run_exp(alignments_blstm, name_suffix="align-blstm")
    summary_report.merge_report(sub_report, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, model
