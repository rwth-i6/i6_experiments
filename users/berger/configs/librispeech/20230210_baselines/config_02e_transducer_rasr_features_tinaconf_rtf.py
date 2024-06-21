import copy
import os
import numpy as np
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
from i6_experiments.users.berger.args.jobs.recognition_args import get_seq2seq_search_parameters
from .config_02e_transducer_rasr_features_tinaconf import subsample_by_4
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
    precompute: bool = False,
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
        if precompute:
            network_dict, extra_python = transducer_model.make_context_1_conformer_transducer_precomputed_recog(
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
        ]
        + (["from returnn.tf.util.data import FeatureDim"] if precompute else []),
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
    data.data_inputs["dev-other_4gram"].corpus_object.duration = 5.120

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

    session_threads_config = rasr.RasrConfig()
    session_threads_config["*"].session.inter_op_parallelism_threads = 1
    session_threads_config["*"].session.intra_op_parallelism_threads = 2

    recog_args = exp_args.get_transducer_recog_step_args(
        num_classes,
        lm_scales=[0.8],
        epochs=[600],
        # lookahead_options={"scale": 0.5},
        search_parameters={"label-pruning": 12.0},
        feature_type=FeatureType.GAMMATONE_16K,
        reduction_factor=4,
        reduction_subtrahend=0,
        extra_config=session_threads_config,
        mem=8,
        rqmt_update={"sbatch_args": ["-A", "rescale_speed", "-p", "rescale_amd"], "cpu": 2},
        search_stats=True,
        num_threads=1,
    )

    # ********** System **********

    system = ReturnnSeq2SeqSystem(
        tools,
        summary_keys=[
            SummaryKey.TRAIN_NAME,
            SummaryKey.CORPUS,
            SummaryKey.RECOG_NAME,
            # SummaryKey.EPOCH,
            # SummaryKey.LM,
            SummaryKey.RTF,
            SummaryKey.WER,
            SummaryKey.SUB,
            SummaryKey.INS,
            SummaryKey.DEL,
            SummaryKey.ERR,
        ],
        summary_sort_keys=[SummaryKey.ERR, SummaryKey.RTF],
    )

    system.init_corpora(
        dev_keys=data.dev_keys,
        test_keys=data.test_keys,
        align_keys=data.align_keys,
        corpus_data=data.data_inputs,
        am_args=exp_args.transducer_recog_am_args,
    )
    system.setup_scoring()

    # ********** Returnn Configs **********

    for lr in [4e-04]:
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
                "recog_ilm-0.2": generate_returnn_config(
                    train=False,
                    ilm_scale=0.2,
                    train_data_config=data.train_data_config,
                    dev_data_config=data.cv_data_config,
                ),
                "recog_ilm-0.2_precompute": generate_returnn_config(
                    train=False,
                    precompute=True,
                    ilm_scale=0.2,
                    train_data_config=data.train_data_config,
                    dev_data_config=data.cv_data_config,
                ),
            },
        )

        system.add_experiment_configs(f"Conformer_Transducer_Viterbi_{name_suffix}_lr-{lr}", returnn_configs)

    system.run_train_step(**train_args)

    np.random.seed(0)
    for _ in range(20):
        lp = np.random.uniform(8.0, 18.0)
        lpl = int(np.round(np.exp(np.random.uniform(np.log(50), np.log(1000)))))
        wep = np.random.uniform(0.3, 0.7)
        wepl = int(np.round(np.exp(np.random.uniform(np.log(10), np.log(1000)))))
        search_params = get_seq2seq_search_parameters(
            lp=lp,
            lpl=lpl,
            wep=wep,
            wepl=wepl,
            allow_blank=True,
            allow_loop=False,
        )
        recog_args["search_parameters"] = search_params
        descr = f"lp-{lp:.2f}_lpl-{lpl}_wep-{wep:.2f}_wepl-{wepl}"
        # system.run_recog_step_for_corpora(recog_exp_names=["recog_ilm-0.2"], corpora=["dev-other_4gram"], recog_descriptor=descr, **recog_args)

    for _ in range(10):
        lp = np.random.uniform(10.0, 13.0)
        lpl = np.random.randint(50, 250)
        wep = 0.5
        wepl = np.random.randint(20, 100)
        search_params = get_seq2seq_search_parameters(
            lp=lp,
            lpl=lpl,
            wep=wep,
            wepl=wepl,
            allow_blank=True,
            allow_loop=False,
        )
        recog_args["search_parameters"] = search_params
        descr = f"lp-{lp:.2f}_lpl-{lpl}_wep-{wep:.2f}_wepl-{wepl}"
        # system.run_recog_step_for_corpora(recog_exp_names=["recog_ilm-0.2"], corpora=["dev-other_4gram"], recog_descriptor=descr, **recog_args)

    for _ in range(20):
        lp = 12.0
        lpl = np.random.randint(10, 250)
        wep = 0.5
        wepl = np.random.randint(10, 100)
        search_params = get_seq2seq_search_parameters(
            lp=lp,
            lpl=lpl,
            wep=wep,
            wepl=wepl,
            allow_blank=True,
            allow_loop=False,
        )
        recog_args["search_parameters"] = search_params
        descr = f"lp-{lp:.2f}_lpl-{lpl}_wep-{wep:.2f}_wepl-{wepl}"
        # system.run_recog_step_for_corpora(recog_exp_names=["recog_ilm-0.2"], corpora=["dev-other_4gram"], recog_descriptor=descr, **recog_args)

    for lp in [
        2.0,
        4.0,
        6.0,
        8.0,
        9.0,
        10.0,
        10.5,
        11.0,
        11.5,
        12.0,
        12.5,
        13.0,
        13.5,
        13.6,
        13.7,
        13.8,
        13.9,
        14.0,
        15.0,
        16.0,
        18.0,
        20.0,
    ]:
        for wep in [0.3, 0.5, 0.7]:
            search_params = get_seq2seq_search_parameters(
                lp=lp,
                lpl=300,
                wep=wep,
                wepl=200,
                allow_blank=True,
                allow_loop=False,
            )
            recog_args["search_parameters"] = search_params
            descr = f"lp-{lp:.2f}_lpl-300_wep-{wep:.2f}_wepl-200"
            system.run_recog_step_for_corpora(
                recog_exp_names=["recog_ilm-0.2"], corpora=["dev-other_4gram"], recog_descriptor=descr, **recog_args
            )

    for lp in [
        2.0,
        4.0,
        6.0,
        8.0,
        9.0,
        10.0,
        10.5,
        11.0,
        11.5,
        12.0,
        12.5,
        13.0,
        13.5,
        13.6,
        13.7,
        13.8,
        13.9,
        14.0,
        15.0,
        16.0,
        18.0,
        20.0,
    ]:
        for wep in [0.5, 0.6]:
            search_params = get_seq2seq_search_parameters(
                lp=lp,
                lpl=1000,
                wep=wep,
                wepl=500,
                allow_blank=True,
                allow_loop=False,
            )
            recog_args["search_parameters"] = search_params
            descr = f"lp-{lp:.2f}_lpl-1000_wep-{wep:.2f}_wepl-500"
            system.run_recog_step_for_corpora(
                recog_exp_names=["recog_ilm-0.2"], corpora=["dev-other_4gram"], recog_descriptor=descr, **recog_args
            )

    recog_args.update(
        {
            "seq2seq_v2": True,
            "label_scorer_type": "precomputed-log-posterior",
            "model_flow_args": {"output_layer_name": "output_precompute"},
        }
    )
    for lp in [
        9.0,
        10.0,
        10.5,
        11.0,
        11.5,
        12.0,
        12.5,
        13.0,
    ]:
        # for wep in [0.3, 0.5, 0.7]:
        for wep in [0.5]:
            search_params = get_seq2seq_search_parameters(
                lp=lp,
                lpl=300,
                wep=wep,
                wepl=200,
                allow_blank=True,
                allow_loop=False,
            )
            recog_args["search_parameters"] = search_params
            descr = f"lp-{lp:.2f}_lpl-300_wep-{wep:.2f}_wepl-200"
            system.run_recog_step_for_corpora(
                recog_exp_names=["recog_ilm-0.2_precompute"],
                corpora=["dev-other_4gram"],
                recog_descriptor=descr,
                **recog_args,
            )

    for lp in [
        9.0,
        10.0,
        10.5,
        11.0,
        11.5,
        12.0,
        12.5,
        13.0,
    ]:
        for wep in [0.5, 0.6]:
            search_params = get_seq2seq_search_parameters(
                lp=lp,
                lpl=1000,
                wep=wep,
                wepl=500,
                allow_blank=True,
                allow_loop=False,
            )
            recog_args["search_parameters"] = search_params
            descr = f"lp-{lp:.2f}_lpl-1000_wep-{wep:.2f}_wepl-500"
            # system.run_recog_step_for_corpora(
            #     recog_exp_names=["recog_ilm-0.2_precompute"],
            #     corpora=["dev-other_4gram"],
            #     recog_descriptor=descr,
            #     **recog_args,
            # )

    # system.run_dev_recog_step(**recog_args)
    # system.run_test_recog_step(**recog_args)

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
