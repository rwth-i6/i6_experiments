import copy
import os
from typing import Dict, Tuple, Optional

import i6_core.rasr as rasr
from i6_core.returnn import ReturnnDumpHDFJob
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
from i6_experiments.users.berger.recipe.returnn.training import (
    GetBestCheckpointJob,
)
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from .config_01d_ctc_conformer_rasr_features import py as py_ctc
from .config_test_3 import PeakyAlignmentJob
from sisyphus import gs, tk, Job, Task

from .crnn_wei import network
from .crnn_wei import extra_python as extra_python_wei

tools = copy.deepcopy(default_tools)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 79


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
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 5,
            },
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
            loss_boost_scale=kwargs.get("loss_boost_scale", 0.0),
            loss_boost_v2=kwargs.get("loss_boost_v2", False),
        )
    else:
        network_dict, extra_python = transducer_model.make_context_1_conformer_transducer_recog(
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

    train_feature_hdf = ReturnnDumpHDFJob(
        {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": tk.Path(
                        "/u/berger/repositories/i6_experiments/users/berger/configs/librispeech/20230210_baselines/extern_sprint_data/train_features.cache.bundle"
                    ),
                    "data_type": "feat",
                },
            },
        },
        returnn_python_exe=tools.returnn_python_exe,
        returnn_root=tools.returnn_root,
    ).out_hdf

    train_align_hdf = ReturnnDumpHDFJob(
        {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": tk.Path(
                        "/u/berger/repositories/i6_experiments/users/berger/configs/librispeech/20230210_baselines/extern_sprint_data/train_align.cache.bundle"
                    ),
                    "data_type": "align",
                    "allophone_labeling": {
                        "silence_phone": "[SILENCE]",
                        "allophone_file": tk.Path(
                            "/work/asr4/zhou/asr-exps/librispeech/2021-01-22_phoneme-transducer/work/allophones/StoreAllophones.88co8MfuJDDS/output/allophones"
                        ),
                        "state_tying_file": tk.Path(
                            "/work/asr4/zhou/asr-exps/librispeech/2021-01-22_phoneme-transducer/work/allophones/DumpStateTying.ls3EIU9exf2C/output/state-tying"
                        ),
                    },
                },
            },
        },
        returnn_python_exe=tools.returnn_python_exe,
        returnn_root=tools.returnn_root,
    ).out_hdf

    train_align_hdf = PeakyAlignmentJob(train_align_hdf).out_hdf

    dev_feature_hdf = ReturnnDumpHDFJob(
        {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": tk.Path(
                        "/u/berger/repositories/i6_experiments/users/berger/configs/librispeech/20230210_baselines/extern_sprint_data/dev_features.cache.bundle"
                    ),
                    "data_type": "feat",
                },
            },
        },
        returnn_python_exe=tools.returnn_python_exe,
        returnn_root=tools.returnn_root,
    ).out_hdf

    dev_align_hdf = ReturnnDumpHDFJob(
        {
            "class": "SprintCacheDataset",
            "data": {
                "data": {
                    "filename": tk.Path(
                        "/u/berger/repositories/i6_experiments/users/berger/configs/librispeech/20230210_baselines/extern_sprint_data/dev_align.cache.bundle"
                    ),
                    "data_type": "align",
                    "allophone_labeling": {
                        "silence_phone": "[SILENCE]",
                        "allophone_file": tk.Path(
                            "/work/asr4/zhou/asr-exps/librispeech/2021-01-22_phoneme-transducer/work/allophones/StoreAllophones.88co8MfuJDDS/output/allophones"
                        ),
                        "state_tying_file": tk.Path(
                            "/work/asr4/zhou/asr-exps/librispeech/2021-01-22_phoneme-transducer/work/allophones/DumpStateTying.ls3EIU9exf2C/output/state-tying"
                        ),
                    },
                },
            },
        },
        returnn_python_exe=tools.returnn_python_exe,
        returnn_root=tools.returnn_root,
    ).out_hdf

    extra_config = {
        "train": {
            "class": "MetaDataset",
            "datasets": {
                "data": {
                    "class": "HDFDataset",
                    "files": [train_feature_hdf],
                    "use_cache_manager": True,
                },
                "classes": {
                    "class": "HDFDataset",
                    "files": [train_align_hdf],
                    "use_cache_manager": True,
                    "seq_ordering": "random",
                    "partition_epoch": 20,
                },
            },
            "data_map": {"data": ("data", "data"), "classes": ("classes", "data")},
            "seq_order_control_dataset": "classes",
        },
        "dev": {
            "class": "MetaDataset",
            "datasets": {
                "data": {
                    "class": "HDFDataset",
                    "files": [dev_feature_hdf],
                    "use_cache_manager": True,
                },
                "classes": {
                    "class": "HDFDataset",
                    "files": [dev_align_hdf],
                    "use_cache_manager": True,
                    "seq_ordering": "random",
                    "partition_epoch": 1,
                },
            },
            "data_map": {"data": ("data", "data"), "classes": ("classes", "data")},
            "seq_order_control_dataset": "classes",
        },
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
        num_epochs=400,
        python_prolog=[
            "import sys",
            "sys.setrecursionlimit(10 ** 6)",
        ],
        extra_python=extra_python,
        num_inputs=50,
        num_outputs=num_classes,
        extern_target_kwargs={"dtype": "int8" if train else "int32"},
        extern_data_config=True,
        grad_noise=0.0,
        grad_clip=20.0,
        schedule=LearningRateSchedules.OCLR_STEP,
        initial_lr=8e-05,
        peak_lr=kwargs.get("peak_lr", 8e-04),
        final_lr=1e-06,
        n_steps_per_epoch=2440,
        batch_size=15000,
        extra_config=extra_config,
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def run_exp(alignments: Dict[str, AlignmentData], ctc_model_checkpoint: tk.Path) -> Tuple[SummaryReport, tk.Path]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_librispeech_data(
        tools.returnn_root,
        tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        alignments=alignments,
        add_unknown=False,
        augmented_lexicon=False,
        use_wei_lexicon=True,
        lm_name="4gram",
        # lm_name="kazuki_transformer",
        feature_type=FeatureType.GAMMATONE_16K,
    )

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=400,
        # gpu_mem_rqmt=24,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.6],
        epochs=[80, 160, 240, 320, 400, "best"],
        lookahead_options={"scale": 0.5},
        search_parameters={"label-pruning": 12.0},
        feature_type=FeatureType.GAMMATONE_16K,
        reduction_factor=4,
        reduction_subtrahend=0,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(tools)

    # ********** Returnn Configs **********

    train_config = generate_returnn_config(
        train=True,
        train_data_config=data.train_data_config,
        dev_data_config=data.cv_data_config,
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

    system.add_experiment_configs("Conformer_Transducer_Viterbi_Wei", returnn_configs)

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

    train_job = system.get_train_job(f"Conformer_Transducer_Viterbi_Wei")
    model = GetBestCheckpointJob(
        model_dir=train_job.out_model_dir, learning_rates=train_job.out_learning_rates
    ).out_checkpoint

    assert system.summary_report
    return system.summary_report, model


def py() -> Tuple[SummaryReport, tk.Path]:
    _, ctc_model, alignments = py_ctc()

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report, model = run_exp(alignments, ctc_model)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, model
