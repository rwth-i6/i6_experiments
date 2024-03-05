import copy
import os
from typing import Dict, Tuple, Optional

import i6_core.rasr as rasr
from i6_core.returnn import Checkpoint
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
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, FeatureType, SummaryKey
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from i6_experiments.users.berger.recipe.returnn.training import (
    GetBestCheckpointJob,
)
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from i6_experiments.users.berger.recipe.returnn.hdf import MatchLengthsJob
from .config_01b_ctc_blstm_rasr_features import py as py_ctc_blstm
from .config_01d_ctc_conformer_rasr_features import py as py_ctc_conf
from sisyphus import gs, tk

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
        (network_dict, extra_python,) = transducer_model.make_context_1_conformer_transducer(
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
            loss_boost_scale=kwargs.get("loss_boost_scale", 5.0),
            loss_boost_v2=kwargs.get("loss_boost_v2", False),
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
        n_steps_per_epoch=2450,
        batch_size=15000,
        extra_config=extra_config,
    )
    returnn_config = serialize_dim_tags(returnn_config)

    return returnn_config


def subsample_by_4(orig_len: int) -> int:
    return -(-orig_len // 4)


def run_exp(
    alignments: Dict[str, AlignmentData],
    ctc_model_checkpoint: Optional[Checkpoint] = None,
    name_suffix: str = "",
    data_control_train: bool = False,
    data_control_cv: bool = False,
    match_lengths: bool = False,
) -> Tuple[SummaryReport, Checkpoint]:
    assert tools.returnn_root is not None
    assert tools.returnn_python_exe is not None
    assert tools.rasr_binary_path is not None

    data = get_librispeech_data(
        tools.returnn_root,
        tools.returnn_python_exe,
        rasr_binary_path=tools.rasr_binary_path,
        alignments=alignments,
        add_unknown_phoneme_and_mapping=False,
        # use_augmented_lexicon=False,
        # use_wei_lexicon=True,
        use_augmented_lexicon=True,
        use_wei_lexicon=False,
        feature_type=FeatureType.GAMMATONE_16K,
    )

    changed_data_configs = []
    if data_control_train:
        changed_data_configs.append(data.train_data_config)
    if data_control_cv:
        changed_data_configs.append(data.cv_data_config)
    for data_config in changed_data_configs:
        data_config["datasets"]["data"].update(
            {
                "seq_ordering": data_config["datasets"]["classes"]["seq_ordering"],
                "partition_epoch": data_config["datasets"]["classes"]["partition_epoch"],
            }
        )
        del data_config["datasets"]["classes"]["seq_ordering"]
        del data_config["datasets"]["classes"]["partition_epoch"]
        data_config["seq_order_control_dataset"] = "data"
    if match_lengths:
        for data_config in [data.train_data_config, data.cv_data_config]:
            data_config["datasets"]["classes"]["files"] = [
                MatchLengthsJob(file, data_config["datasets"]["data"]["files"], subsample_by_4).out_hdf
                for file in data_config["datasets"]["classes"]["files"]
            ]

    # ********** Step args **********

    train_args = exp_args.get_transducer_train_step_args(
        num_epochs=400,
        gpu_mem_rqmt=24,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.5, 0.6, 0.7],
        epochs=[320, 400, "best"],
        # lookahead_options={"scale": 0.5},
        search_parameters={"label-pruning": 12.0},
        feature_type=FeatureType.GAMMATONE_16K,
        reduction_factor=4,
        reduction_subtrahend=0,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.CORPUS,
            SummaryKey.EPOCH,
            SummaryKey.PRIOR,
            SummaryKey.LM,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.INS,
            SummaryKey.DEL,
            SummaryKey.ERR,
        ],
    )

    # ********** Returnn Configs **********

    for label_smoothing, loss_boost_v2, peak_lr, loss_boost_scale, ctc_init in [
        #        (None, True, 4e-04, 5.0, False),
        #        (None, True, 8e-04, 5.0, False),
        #        # (None, False, 4e-04, 5.0, False),
        #        # (None, False, 8e-04, 5.0, False),
        #        # (None, False, 4e-04, 5.0, True),
        #        # (None, False, 8e-04, 5.0, True),
        #        (0.2, True, 4e-04, 5.0, False),
        #        (0.2, True, 8e-04, 5.0, False),
        #        # (0.2, False, 4e-04, 5.0, False),
        #        # (0.2, False, 8e-04, 5.0, False),
        (None, False, 8e-04, 0.0, False),
        #        (None, False, 8e-04, 0.0, True),
    ]:
        ctc_init = ctc_init and (ctc_model_checkpoint is not None)
        train_config = generate_returnn_config(
            train=True,
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            label_smoothing=label_smoothing,
            loss_boost_v2=loss_boost_v2,
            loss_boost_scale=loss_boost_scale,
            peak_lr=peak_lr,
            model_preload=ctc_model_checkpoint if ctc_init else None,
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

        suffix = f"lr-{peak_lr}"
        if loss_boost_scale:
            if loss_boost_v2:
                suffix += "_loss-boost-v2"
            else:
                suffix += "_loss-boost"
        if label_smoothing:
            suffix += f"_ls-{label_smoothing}"

        if ctc_init:
            suffix += "_ctc-init"

        system.add_experiment_configs(f"Conformer_Transducer_Viterbi_{suffix}_{name_suffix}", returnn_configs)

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

    train_job = system.get_train_job(f"Conformer_Transducer_Viterbi_lr-0.0008_{name_suffix}")
    # model = GetBestCheckpointJob(
    #     model_dir=train_job.out_model_dir, learning_rates=train_job.out_learning_rates
    # ).out_checkpoint
    model = train_job.out_checkpoints[400]
    assert isinstance(model, Checkpoint)

    assert system.summary_report
    return system.summary_report, model


def py() -> Tuple[SummaryReport, Checkpoint]:
    _, _, alignments_blstm = py_ctc_blstm()
    _, ctc_model, alignments_conf = py_ctc_conf()

    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report, model = run_exp(alignments_blstm, ctc_model, name_suffix="blstm-align")
    # summary_report.merge_report(run_exp(alignments_conf, name_suffix="conf-align")[0])

    alignments_nour = {}

    for key, path in [
        (
            "train-other-960_align",
            tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.0a7MCFFN37Bg/output/alignment.cache.bundle"
            ),
        ),
        (
            "dev-clean_align",
            tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.HjJgbxdZhWZj/output/alignment.cache.bundle"
            ),
        ),
        (
            "dev-other_align",
            tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.UatqVP2YM55f/output/alignment.cache.bundle"
            ),
        ),
    ]:
        alignments_nour[key] = AlignmentData(
            alignment_cache_bundle=path,
            allophone_file=tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_core/lexicon/allophones/StoreAllophonesJob.8Nygr67IZfVG/output/allophones"
            ),
            state_tying_file=tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_core/lexicon/allophones/DumpStateTyingJob.6w7HRWTGkgEd/output/state-tying"
            ),
            silence_phone="<blank>",
        )
    # report, model = run_exp(
    #     alignments_nour, name_suffix="nour-align", data_control_train=True, data_control_cv=False, match_lengths=True
    # )
    # summary_report.merge_report(report)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report, model
