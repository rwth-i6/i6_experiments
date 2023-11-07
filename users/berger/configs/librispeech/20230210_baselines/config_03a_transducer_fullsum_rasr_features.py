import copy
import os
from typing import Dict, Optional

import i6_core.rasr as rasr
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.users.berger.args.experiments import transducer as exp_args
from i6_experiments.users.berger.args.returnn.config import get_returnn_config
from i6_experiments.users.berger.args.returnn.learning_rates import (
    LearningRateSchedules,
)
from i6_experiments.users.berger.corpus.librispeech.viterbi_transducer_data import (
    get_librispeech_data,
)
import i6_experiments.users.berger.network.models.context_1_transducer as transducer_model
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, FeatureType
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from .config_01b_ctc_conformer import py as py_ctc
from .config_02a_transducer_rasr_features import py as py_transducer
from sisyphus import gs, tk

# ********** Settings **********

tools = copy.deepcopy(default_tools)

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 79


# ********** Return Config **********


def generate_returnn_config(
    train: bool,
    *,
    train_data_config: dict,
    dev_data_config: dict,
    model_preload: tk.Path,
) -> ReturnnConfig:
    if train:
        (network_dict, extra_python,) = transducer_model.make_context_1_conformer_transducer_fullsum(
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 10,
                "max_feature_num": 5,
                "max_feature": 5,
            },
            conformer_args={
                "num_blocks": 12,
                "size": 512,
                "dropout": 0.1,
                "l2": 5e-06,
            },
            compress_joint_input=True,
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
        )
    else:
        (network_dict, extra_python,) = transducer_model.make_context_1_conformer_transducer_recog(
            num_outputs=num_classes,
            conformer_args={
                "num_blocks": 12,
                "size": 512,
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
            },
        )

    returnn_config = get_returnn_config(
        network=network_dict,
        target="classes",
        num_epochs=300,
        extra_python=extra_python,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        num_inputs=50,
        num_outputs=num_classes,
        extern_data_config=True,
        extern_target_kwargs={"dtype": "int8" if train else "int32"},
        grad_noise=0.0,
        grad_clip=20.0,
        schedule=LearningRateSchedules.CONST_DECAY,
        const_lr=5e-05,
        decay_lr=1e-05,
        final_lr=1e-06,
        batch_size=15000,
        accum_grad=2,
        use_chunking=False,
        extra_config={
            "max_seq_length": {"classes": 550},
            "train": train_data_config,
            "dev": dev_data_config,
            "preload_from_files": {
                "base": {
                    "init_for_train": True,
                    "ignore_missing": True,
                    "filename": model_preload,
                }
            },
        },
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp(alignments: Dict[str, AlignmentData], viterbi_model_checkpoint: tk.Path) -> SummaryReport:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None

    data = get_librispeech_data(
        tools.returnn_root,
        tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        alignments=alignments,
        add_unknown=False,
        lm_name="4gram",
        feature_type=FeatureType.GAMMATONE,
        # lm_name="kazuki_transformer",
    )

    # ********** Returnn Configs **********

    config_generator_kwargs = {
        "train_data_config": data.train_data_config,
        "dev_data_config": data.cv_data_config,
        "model_preload": viterbi_model_checkpoint,
    }

    train_config = generate_returnn_config(train=True, **config_generator_kwargs)
    recog_config = generate_returnn_config(train=False, **config_generator_kwargs)

    returnn_configs = ReturnnConfigs(
        train_config=train_config,
        recog_configs={"recog": recog_config},
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=300,
        gpu_mem_rqmt=24,
        mem_rqmt=24,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.6],
        epochs=[80, 160, 240, 300, "best"],
        lookahead_options={"scale": 0.5},
        search_parameters={"label-pruning": 11.2},
        feature_type=FeatureType.GAMMATONE,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    system.add_experiment_configs("Conformer_Transducer_Fullsum", returnn_configs)
    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.transducer_recog_am_args,
    )
    system.setup_scoring()

    system.run_train_step(**train_args)

    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    assert system.summary_report
    return system.summary_report


def py() -> SummaryReport:
    _, alignments = py_ctc()
    _, model = py_transducer()

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = run_exp(alignments, model)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
