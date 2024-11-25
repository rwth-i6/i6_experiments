import copy
import os
from typing import Dict, Tuple, Optional

import i6_core.rasr as rasr
import i6_experiments.users.berger.network.models.context_1_transducer_tinaconf as transducer_model
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
from i6_experiments.users.berger.recipe.mm import ComputeTSEJob
from i6_experiments.users.berger.recipe.returnn.hdf import MatchLengthsJob
from i6_experiments.users.berger.recipe.summary.report import SummaryReport
from i6_experiments.users.berger.systems.dataclasses import AlignmentData
from i6_experiments.users.berger.systems.dataclasses import ReturnnConfigs, FeatureType, SummaryKey
from i6_experiments.users.berger.systems.returnn_seq2seq_system import (
    ReturnnSeq2SeqSystem,
)
from i6_experiments.users.berger.util import default_tools
from i6_private.users.vieting.helpers.returnn import serialize_dim_tags
from sisyphus import gs, tk

tools = copy.deepcopy(default_tools)

# ********** Settings **********

rasr.flow.FlowNetwork.default_flags = {"cache_mode": "task_dependent"}


num_classes = 79
num_epochs = 600


# ********** Return Config **********


def generate_returnn_config(
    train: bool,
    *,
    train_data_config: dict,
    dev_data_config: dict,
    **kwargs,
) -> ReturnnConfig:
    if train:
        network_dict, extra_python = transducer_model.make_context_1_conformer_transducer(
            num_inputs=50,
            num_outputs=num_classes,
            specaug_args={
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 5,
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
        network_dict, extra_python = transducer_model.make_context_1_conformer_transducer_recog(
            num_inputs=50,
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
        num_epochs=num_epochs,
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
        grad_clip=0.0,
        schedule=LearningRateSchedules.OCLR,
        # initial_lr=1e-03 / 30,
        # peak_lr=1e-03,
        initial_lr=1e-05,
        peak_lr=kwargs.get("peak_lr", 4e-04),
        final_lr=1e-06,
        batch_size=12500,
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
        use_augmented_lexicon=False,
        use_wei_lexicon=True,
        # use_augmented_lexicon=True,
        # use_wei_lexicon=False,
        feature_type=FeatureType.GAMMATONE_16K,
    )

    changed_data_configs = []
    if data_control_train:
        changed_data_configs.append(data.train_data_config)
    if data_control_cv:
        changed_data_configs.append(data.cv_data_config)

    data.train_data_config["datasets"]["classes"]["seq_ordering"] = "laplace:.384"
    data.train_data_config["datasets"]["classes"]["partition_epoch"] = 40

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
        num_epochs=num_epochs,
        gpu_mem_rqmt=11,
    )

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.8],
        epochs=[600],
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
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    for lr in [4e-04, 8e-04, 1e-03]:
        train_config = generate_returnn_config(
            train=True,
            train_data_config=data.train_data_config,
            dev_data_config=data.cv_data_config,
            peak_lr=lr,
            label_smoothing=None,
            loss_boost_v2=False,
            loss_boost_scale=0.0,
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
                for ilm_scale in [0.2]
            },
        )

        system.add_experiment_configs(f"Conformer_Transducer_Viterbi_{name_suffix}_lr-{lr}", returnn_configs)

    system.run_train_step(**train_args)

    system.run_dev_recog_step(**recog_args)
    system.run_test_recog_step(**recog_args)

    train_job = system.get_train_job(f"Conformer_Transducer_Viterbi_{name_suffix}_lr-0.0004")
    # model = GetBestCheckpointJob(
    #     model_dir=train_job.out_model_dir, learning_rates=train_job.out_learning_rates
    # ).out_checkpoint
    model = train_job.out_checkpoints[600]
    assert isinstance(model, Checkpoint)

    assert system.summary_report
    return system.summary_report, model


def py() -> SummaryReport:
    filename_handle = os.path.splitext(os.path.basename(__file__))[0][len("config_") :]
    gs.ALIAS_AND_OUTPUT_SUBDIR = f"{filename_handle}/"

    summary_report = SummaryReport()

    alignments_nour = {}

    alignment_paths_nour = {
        0.1: {
            "train-other-960_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.waHWItDFeH4p/output/alignment.cache.bundle"
            ),
            "dev-clean_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.39RvKswiwE5X/output/alignment.cache.bundle"
            ),
            "dev-other_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.UQcQtRgFJtri/output/alignment.cache.bundle"
            ),
        },
        0.3: {
            "train-other-960_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.4bWrFMO9rBP7/output/alignment.cache.bundle"
            ),
            "dev-clean_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.WAPZqf6YGRqV/output/alignment.cache.bundle"
            ),
            "dev-other_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.8e6a0qmzOKPS/output/alignment.cache.bundle"
            ),
        },
        0.5: {
            "train-other-960_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.9F7XAOE5SW6a/output/alignment.cache.bundle"
            ),
            "dev-clean_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.NZ9KCbM3iaUM/output/alignment.cache.bundle"
            ),
            "dev-other_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.NrLiIv3mx2Mi/output/alignment.cache.bundle"
            ),
        },
        0.7: {
            "train-other-960_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.nOX1kOQx5Txi/output/alignment.cache.bundle"
            ),
            "dev-clean_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.WhaHQ8VtCQWb/output/alignment.cache.bundle"
            ),
            "dev-other_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.Z7Yc9kH2BYOc/output/alignment.cache.bundle"
            ),
        },
        1.0: {
            "train-other-960_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.0a7MCFFN37Bg/output/alignment.cache.bundle"
            ),
            "dev-clean_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.HjJgbxdZhWZj/output/alignment.cache.bundle"
            ),
            "dev-other_align": tk.Path(
                "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_experiments/users/berger/recipe/mm/alignment/Seq2SeqAlignmentJob.UatqVP2YM55f/output/alignment.cache.bundle"
            ),
        },
    }

    for am_scale, alignment_paths in alignment_paths_nour.items():
        for key, path in alignment_paths.items():
            align_data = AlignmentData(
                alignment_cache_bundle=path,
                allophone_file=tk.Path(
                    "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_core/lexicon/allophones/StoreAllophonesJob.8Nygr67IZfVG/output/allophones"
                ),
                state_tying_file=tk.Path(
                    "/work/asr3/raissi/shared_workspaces/bayoumi/sisyphus_work/i6_core/lexicon/allophones/DumpStateTyingJob.6w7HRWTGkgEd/output/state-tying"
                ),
                silence_phone="<blank>",
            )
            alignments_nour[key] = align_data

            if "train" in key:  # So far no reference alignment on dev-clean/dev-other
                compute_tse_job = ComputeTSEJob(
                    alignment_cache=align_data.alignment_cache_bundle,
                    allophone_file=align_data.allophone_file,
                    silence_phone=align_data.silence_phone,
                    upsample_factor=4,
                    ref_alignment_cache=tk.Path(
                        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob"
                        ".oyZ7O0XJcO20/output/alignment.cache.bundle"
                    ),
                    ref_allophone_file=tk.Path(
                        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
                    ),
                    ref_silence_phone="[SILENCE]",
                    ref_upsample_factor=1,
                    # remove_outlier_limit=50,
                )
                tk.register_output(f"tse/nour-align_{key}_am-{am_scale}", compute_tse_job.out_tse_frames)
        report, model = run_exp(
            alignments_nour,
            name_suffix=f"nour-align-am-{am_scale}",
            data_control_train=True,
            data_control_cv=False,
            match_lengths=True,
        )
        summary_report.merge_report(report, update_structure=True)

    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/summary.report", summary_report)

    return summary_report
