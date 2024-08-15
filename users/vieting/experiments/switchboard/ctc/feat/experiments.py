import copy
import os.path
from sisyphus import tk, gs
from typing import Any, Dict, Tuple

from i6_core.returnn.config import CodeWrapper
from i6_core.recognition import Hub5ScoreJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.users.vieting.tools.report import Report
from i6_experiments.users.vieting.experiments.switchboard.transducer.feat.data import (
    get_switchboard_data,
    get_returnn_ogg_datasets,
)
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import (
    TransducerSystem,
    ReturnnConfigs,
    ScorerInfo,
    SearchTypes,
)
from .baseline_args import get_nn_args as get_nn_args_baseline
from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE, SCTK_BINARY_PATH


def get_datasets(**kwargs):
    returnn_datasets = get_returnn_ogg_datasets(**kwargs)
    train_corpus, dev_corpora, segments = get_switchboard_data()
    rasr_loss_lexicon = train_corpus.lexicon["filename"]
    rasr_loss_corpus = train_corpus.corpus_object.corpus_file
    rasr_loss_segments = segments["all_filtered"]
    return returnn_datasets, rasr_loss_corpus, rasr_loss_segments, rasr_loss_lexicon, dev_corpora["ctc"]


def args_to_key_and_report_strings(args: Dict[str, Any]) -> Tuple[str, str]:
    """
    Process the argument dictionary to generate a key string and a report string.

    Returns:
        tuple: A tuple containing the key string and the report string.
    """

    key_string = ""
    report_dict = {}

    for key, value in args.items():
        if key in ["speed", "tempo", "preemphasis", "non_linearity"]:
            key_component = f"{key}_{value['prob']}_{value['minimum']}_{value['maximum']}"
            if "default" in value:
                key_component += f"_{value['default']}"
            key_string += key_component
            report_dict[key] = f"{value['prob']}_{value['minimum']}_{value['maximum']}"
        elif key == "codecs":
            codecs_str = "_".join([f"{codec['encoding']}_{codec['prob']}" for codec in value])
            key_string += f"{key}_{codecs_str}_"
            report_dict[key] = codecs_str
        else:
            raise ValueError(f"Unknown argument name: {key}")

    return key_string, report_dict


def run_nn_args(nn_args, report_args_collection, dev_corpora, report_name="", returnn_root=None, recog_args=None):
    returnn_configs = {}
    for exp in nn_args.returnn_training_configs:
        prior_config = copy.deepcopy(nn_args.returnn_training_configs[exp])
        prior_config.config["batch_size"] = prior_config.config["batch_size"]["data"]
        assert isinstance(prior_config.config["batch_size"], int)
        returnn_configs[exp] = ReturnnConfigs(
            train_config=nn_args.returnn_training_configs[exp],
            prior_config=prior_config,
            recog_configs={"recog": nn_args.returnn_recognition_configs[exp]},
        )

    recog_args = {
        **{
            "lm_scales": [0.7],
            "prior_scales": [0.3, 0.5],
            "epochs": [300, 400, 450, "best"],
            "lookahead_options": {"lm_lookahead_scale": 0.7},
            "label_scorer_args": {
                "use_prior": True,
                "extra_args": {"blank_label_index": 0},
            },
            "label_tree_args": {"skip_silence": True},
            "search_parameters": {
                "allow-blank-label": True,
                "allow-label-loop": True,
                "allow-label-recombination": True,
                "allow-word-end-recombination": True,
                "create-lattice": True,
                "label-pruning": 11.2,
                "label-pruning-limit": 100000,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 10000,
            },
        },
        **(recog_args or {}),
    }
    score_info = ScorerInfo()
    score_info.ref_file = dev_corpora["hub5e00"].stm
    score_info.job_type = Hub5ScoreJob
    score_info.score_kwargs = {"glm": dev_corpora["hub5e00"].glm, "sctk_binary_path": SCTK_BINARY_PATH}

    ctc_nn_system = TransducerSystem(
        returnn_root=returnn_root or RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
        require_native_lstm=False,
    )
    ctc_nn_system.init_system(
        returnn_configs=returnn_configs,
        dev_keys=["hub5e00"],
        corpus_data=dev_corpora,
        am_args={
            "state_tying": "monophone",
            "states_per_phone": 1,
            "tdp_transition": (0, 0, 0, 0),
            "tdp_silence": (0, 0, 0, 0),
            "phon_history_length": 0,
            "phon_future_length": 0,
        },
        scorer_info=score_info,
        report=Report(
            columns_start=["train_name"],
            columns_end=["lm_scale", "prior_scale", "sub", "del", "ins", "wer"],
        ),
    )
    ctc_nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_from_lexicon = False
    ctc_nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_all = True
    ctc_nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_from_file = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/allophones_blank",
        hash_overwrite="SWB_ALLOPHONE_FILE_WEI_BLANK",
        cached=True,
    )
    ctc_nn_system.run_train_step(nn_args.training_args)
    ctc_nn_system.run_dev_recog_step(recog_args=recog_args, report_args=report_args_collection)

    assert ctc_nn_system.report is not None
    report = ctc_nn_system.report
    report.delete_redundant_columns()
    report.delete_redundant_rows()
    if report_name:
        tk.register_report(
            os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, report_name),
            values=report.get_values(),
            template=report.get_template(),
        )
    return report, ctc_nn_system


def run_mel_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()
    returnn_args = {
        "batch_size": 10000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "conformer_type": "wei",
        "specaug_old": {"max_feature": 8},
    }
    feature_args = {"class": "LogMelNetwork", "wave_norm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256}
    lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }
    recog_args = {"epochs": [350, 390, 400, 410, 450]}

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs10k_lgm80_baseline": dict(
                returnn_args=returnn_args,
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
            "bs5k_lgm80_baseline": dict(
                returnn_args={**returnn_args, "batch_size": 5000},
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "5k"},
            ),
            "bs2x5k_lgm80_baseline": dict(
                returnn_args={
                    **returnn_args,
                    "batch_size": 5000,
                    "extra_args": {
                        "accum_grad_multiple_step": 2,
                        "watch_memory": True,
                    },
                },
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
        },
        num_epochs=450,
        evaluation_epochs=[6, 12, 24, 350, 390, 400, 410, 450],
        prefix="conformer_",
    )
    report, ctc_nn_system = run_nn_args(nn_args, report_args_collection, dev_corpora, recog_args=recog_args)
    return report, ctc_nn_system


def run_scf_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()
    returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "extra_args": {
            "accum_grad_multiple_step": 2,
            "watch_memory": True,
            "conv_pad_seq_len_to_power": 1.5,
        },
        "conformer_type": "wei",
        "specaug_old": {"max_feature": 15},
    }
    feature_args = {"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}
    lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs2x5k_scf_baseline": dict(
                returnn_args=returnn_args,
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_baseline_wn": dict(
                returnn_args=returnn_args,
                feature_args={**feature_args, "wave_norm": True},
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_baseline_preemphasis97_wn": dict(
                returnn_args=returnn_args,
                feature_args={**feature_args, "preemphasis": 0.97, "wave_norm": True},
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_baseline_preemphasis97": dict(
                returnn_args=returnn_args,
                feature_args={**feature_args, "preemphasis": 0.97},
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs7k_scf_baseline": dict(
                returnn_args={
                    **returnn_args,
                    "batch_size": 7000,
                    "extra_args": {"watch_memory": True, "conv_pad_seq_len_to_power": 1.5},
                },
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "7k"},
            ),
        },
        num_epochs=450,
        evaluation_epochs=[6, 12, 24, 350, 390, 400, 410, 450],
        prefix="conformer_",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report, ctc_nn_system = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        returnn_root=returnn_root,
        recog_args={"epochs": [350, 390, 400, 410, 450]},
    )
    return report, ctc_nn_system


def run_scf_baseline_lr_reset():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"
    _, nn_system = run_scf_baseline()
    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()
    returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "extra_args": {
            "accum_grad_multiple_step": 2,
            "watch_memory": True,
            "conv_pad_seq_len_to_power": 1.5,
            "preload_from_files": {
                "existing-model": {
                    "filename": nn_system.train_jobs["conformer_bs2x5k_scf_baseline_preemphasis97_wn"].out_checkpoints[
                        400
                    ],
                    "init_for_train": True,
                },
            },
        },
        "conformer_type": "wei",
        "specaug_old": {"max_feature": 15},
    }
    feature_args = {
        "class": "ScfNetwork",
        "size_tf": 256 // 2,
        "stride_tf": 10 // 2,
        "preemphasis": 0.97,
        "wave_norm": True,
    }
    lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs2x5k_scf_baseline_preemphasis97_wn_lr_reset": dict(
                returnn_args=returnn_args,
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
        },
        num_epochs=426,
        evaluation_epochs=[326, 376, 386, 396, 406, 426],
        prefix="conformer_",
    )
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report, ctc_nn_system = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        returnn_root=returnn_root,
        recog_args={"epochs": [326, 376, 386, 396, 406, 426]},
    )
    return report, ctc_nn_system


def run_scf_frozen_features():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    _, nn_system = run_scf_baseline()

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()

    # Common configurations
    common_returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "conformer_type": "wei",
        "specaug_old": {"max_feature": 15},
        "extra_args": {
            "accum_grad_multiple_step": 2,
            "conv_pad_seq_len_to_power": 1.5,
        },
    }

    common_feature_args = {
        "class": "ScfNetwork",
        "size_tf": 256 // 2,
        "stride_tf": 10 // 2,
        "preemphasis": 0.97,
        "wave_norm": True,
    }

    common_lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }

    common_report_args = {
        "batch_size": "2x5k",
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "epoch_256_freezing": {
                "returnn_args": {
                    "staged_opts": {
                        256: "freeze_features",
                    },
                    **common_returnn_args,
                },
                "feature_args": common_feature_args,
                "lr_args": common_lr_args,
                "report_args": common_report_args,
            },
            "fixed_features": {
                "returnn_args": {
                    **common_returnn_args,
                    "staged_opts": {
                        1: "freeze_features",
                    },
                    "extra_args": {
                        **common_returnn_args["extra_args"],
                        "preload_from_files": {
                            "existing-model": {
                                "filename": nn_system.train_jobs[
                                    "conformer_bs2x5k_scf_baseline_preemphasis97_wn"
                                ].out_checkpoints[400],
                                "init_for_train": True,
                                "prefix": "features",
                                "var_name_mapping": {
                                    "/conv_h_filter/conv_h_filter": "features/conv_h_filter/conv_h_filter",
                                    "/conv_l/W": "features/conv_l/W",
                                    "/conv_l_act/bias": "features/conv_l_act/bias",
                                    "/conv_l_act/scale": "features/conv_l_act/scale",
                                    "/wave_norm/bias": "features/wave_norm/bias",
                                    "/wave_norm/scale": "features/wave_norm/scale",
                                },
                            },
                        },
                    },
                },
                "feature_args": common_feature_args,
                "lr_args": common_lr_args,
                "report_args": common_report_args,
            },
        },
        num_epochs=450,
        evaluation_epochs=[350, 390, 400, 410, 450],
        prefix="conformer_bs2x5k_frozen_features_",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report, _ = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        "report_scf_frozen_features.csv",
        returnn_root=returnn_root,
        recog_args={"epochs": [350, 390, 400, 410, 450]},
    )
    return report


def run_scf_audio_perturbation_from_checkpoint():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    _, nn_system = run_scf_baseline()

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets(use_multi_proc_dataset=True, pre_process=CodeWrapper("audio_perturb_runner.run"))
    returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "conformer_type": "wei",
        "audio_perturbation": True,
        "use_multi_proc_dataset": True,
        "specaug_old": {"max_feature": 15},
        "extra_args": {
            "accum_grad_multiple_step": 2,
            "watch_memory": True,
            "conv_pad_seq_len_to_power": 1.5,
            "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
        },
    }

    # usually the lr args would need to be adapted to fit with the checkpoint,
    # but experiments showed that restarting the scheduling is better
    lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }

    perturbation_args = [
        {"codecs": [{"encoding": "ULAW", "prob": 0.3, "minimum": 250, "maximum": 260}]},
        {"codecs": [{"encoding": "ULAW", "prob": 0.7, "minimum": 250, "maximum": 260}]},
        {"codecs": [{"encoding": "ULAW", "prob": 0.7, "minimum": 240, "maximum": 270}]},
        {"codecs": [{"encoding": "ULAW", "prob": 1, "minimum": 250, "maximum": 260}]},
        {"tempo": {"prob": 0.3, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.8, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.3, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 0.8, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 1, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3}},
        {"speed": {"prob": 0.5, "minimum": 0.88, "maximum": 1.12}},
        {"speed": {"prob": 0.3, "minimum": 0.88, "maximum": 1.12}},
        {"speed": {"prob": 0.7, "minimum": 0.88, "maximum": 1.12}},
        {"speed": {"prob": 0.6, "minimum": 0.8, "maximum": 1.2}},
        {"speed": {"prob": 0.6, "minimum": 0.9, "maximum": 1.1}},
        {"speed": {"prob": 1, "minimum": 0.9, "maximum": 1.1}},
        {"speed": {"prob": 1, "minimum": 0.8, "maximum": 1.2}},
        {"non_linearity": {"prob": 0.7, "minimum": 0.9, "maximum": 1.1}},
        {"non_linearity": {"prob": 0.3, "minimum": 0.9, "maximum": 1.1}},
        {"non_linearity": {"prob": 1, "minimum": 0.9, "maximum": 1.1}},
        # v2
        {"tempo": {"prob": 0.8, "minimum": 0.65, "maximum": 1.35}},
        {
            "tempo": {"prob": 0.8, "minimum": 0.7, "maximum": 1.3},
            "non_linearity": {"prob": 0.3, "minimum": 0.9, "maximum": 1.1},
        },
        {
            "tempo": {"prob": 0.3, "minimum": 0.7, "maximum": 1.3},
            "non_linearity": {"prob": 0.3, "minimum": 0.9, "maximum": 1.1},
        },
        {"non_linearity": {"prob": 0.3, "minimum": 0.85, "maximum": 1.15}},
    ]

    # seperated because the training needs a different network without preemphasis
    perturbation_args_preemphasis = [
        {"preemphasis": {"prob": 0.3, "minimum": 0.94, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.94, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.90, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.90, "maximum": 1.0, "default": 0.95}},
        {"preemphasis": {"prob": 1.0, "minimum": 0.94, "maximum": 1.0, "default": 0.97}},
    ]

    nn_base_args = {}

    for args in perturbation_args:
        exp_name_suffix, report_args = args_to_key_and_report_strings(args)

        # Construct the exp_name and report_args
        exp_name = f"scf_bs2x5k_perturb_from_checkpoint_24_{exp_name_suffix}"
        nn_base_args[exp_name] = dict(
            returnn_args={
                "extra_args": {
                    **returnn_args["extra_args"],
                    "audio_perturb_args": args,
                    "preload_from_files": {
                        "existing-model": {
                            "filename": nn_system.train_jobs[
                                "conformer_bs2x5k_scf_baseline_preemphasis97_wn"
                            ].out_checkpoints[24],
                            "init_for_train": True,
                        }
                    },
                },
                **returnn_args,
            },
            feature_args={
                "class": "ScfNetwork",
                "size_tf": 256 // 2,
                "stride_tf": 10 // 2,
                "preemphasis": 0.97,
                "wave_norm": True,
            },
            lr_args=lr_args,
            report_args=report_args,
        )
    for args in perturbation_args_preemphasis:
        exp_name_suffix, report_args = args_to_key_and_report_strings(args)

        # Construct the exp_name and report_args
        exp_name = f"scf_bs2x5k_perturb_from_checkpoint_24_{exp_name_suffix}"
        nn_base_args[exp_name] = dict(
            returnn_args={
                "extra_args": {
                    **returnn_args["extra_args"],
                    "audio_perturb_args": args,
                    "preload_from_files": {
                        "existing-model": {
                            "filename": nn_system.train_jobs["conformer_bs2x5k_scf_baseline"].out_checkpoints[24],
                            "init_for_train": True,
                        }
                    },
                },
                **returnn_args,
            },
            feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2},
            lr_args=lr_args,
            report_args=report_args,
        )
    # comparison with wave norm
    for args in perturbation_args_preemphasis:
        exp_name_suffix, report_args = args_to_key_and_report_strings(args)

        # Construct the exp_name and report_args
        exp_name = f"scf_bs2x5k_perturb_from_checkpoint_24_wn_{exp_name_suffix}"
        nn_base_args[exp_name] = dict(
            returnn_args={
                "extra_args": {
                    **returnn_args["extra_args"],
                    "audio_perturb_args": args,
                    "preload_from_files": {
                        "existing-model": {
                            "filename": nn_system.train_jobs["conformer_bs2x5k_scf_baseline_wn"].out_checkpoints[24],
                            "init_for_train": True,
                        }
                    },
                },
                **returnn_args,
            },
            feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2, "wave_norm": True},
            lr_args=lr_args,
            report_args=report_args,
        )

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args=nn_base_args,
        num_epochs=426,
        evaluation_epochs=[[376, 386, 396, 406, 426]],
        prefix="conformer_",
    )
    report, _ = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        "report_scf_audio_perturbation_from_checkpoint24.csv",
        recog_args={"epochs": [[376, 386, 396, 406, 426]]},
    )
    return report


def run_scf_specaug_sort():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()
    returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "extra_args": {
            "accum_grad_multiple_step": 2,
            "watch_memory": True,
            "conv_pad_seq_len_to_power": 1.5,
        },
        "conformer_type": "wei",
        "specaug_old": {"max_feature": 15},
    }
    feature_args = {"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2}
    lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs2x5k_scf_specaugsortlayer2": dict(
                returnn_args={**returnn_args, "specaug_old": {"max_feature": 15, "sort_layer2": True}},
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_specaugsortlayer2frome210": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_old": {"max_feature": 15, "sort_layer2": True},
                    "extra_args": {
                        "watch_memory": True,
                        "conv_pad_seq_len_to_power": 1.5,
                        "preload_from_files": {
                            "existing-model": {
                                "filename": (
                                    "/work/asr4/vieting/setups/swb/work/20230406_feat/i6_core/returnn/training/"
                                    "ReturnnTrainingJob.y9otnVMrBAWw/output/models/backup.epoch.210"
                                ),
                                "init_for_train": True,
                            },
                        },
                        **returnn_args["extra_args"],
                    },
                },
                feature_args=feature_args,
                lr_args={**lr_args, "peak_lr": 3.35e-4, "increase_epochs": 0, "decrease_epochs": 150},
                report_args={"batch_size": "2x5k"},
            ),
        },
        num_epochs=450,
        evaluation_epochs=[350, 390, 400, 410, 450],
        prefix="conformer_",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report, ctc_nn_system = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        returnn_root=returnn_root,
        recog_args={"epochs": [350, 390, 400, 410, 450]},
    )
    return report


def run_scf_baseline_decaying_batchsize():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    _, nn_system = run_scf_baseline()

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()
    returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "extra_args": {
            "conv_pad_seq_len_to_power": 1.5,
            "preload_from_files": {
                "existing-model": {
                    "filename": nn_system.train_jobs["conformer_bs2x5k_scf_baseline_preemphasis97_wn"].out_checkpoints[
                        24
                    ],
                    "init_for_train": True,
                }
            },
        },
        "conformer_type": "wei",
        "specaug_old": {"max_feature": 15},
    }
    feature_args = {
        "class": "ScfNetwork",
        "size_tf": 256 // 2,
        "stride_tf": 10 // 2,
        "preemphasis": 0.97,
        "wave_norm": True,
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "scf_baseline": dict(
                returnn_args=returnn_args,
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 6.2668056e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 156,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "deacy_epoch": 24,
                    "start_batch_size": 5000,
                    "end_batch_size": 10000,
                },
            ),
        },
        num_epochs=426,
        evaluation_epochs=[366, 376, 386, 426],
        prefix="conformer_bs5k_decay_",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report, _ = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        "report_scf_baseline_bs_decay.csv",
        returnn_root=returnn_root,
        recog_args={"epochs": [366, 376, 386, 426]},
    )
    return report


def run_mel_audio_perturbation_from_checkpoint():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    _, nn_system = run_mel_baseline()

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets(use_multi_proc_dataset=True, pre_process=CodeWrapper("audio_perturb_runner.run"))
    returnn_args = {
        "batch_size": 5000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
        "conformer_type": "wei",
        "audio_perturbation": True,
        "use_multi_proc_dataset": True,
        "specaug_old": {"max_feature": 8},
        "extra_args": {
            "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
            "conv_pad_seq_len_to_power": 1.5,
            "accum_grad_multiple_step": 2,
        },
    }
    # usually the lr args would need to be adapted to fit with the checkpoint,
    # but experiments showed that restarting the scheduling is better
    lr_args = {
        "peak_lr": 4e-4,
        "start_lr": 1.325e-05,
        "end_lr": 1e-5,
        "increase_epochs": 180,
        "decrease_epochs": 180,
        "final_epochs": 0,
    }

    perturbation_args = [
        {"codecs": [{"encoding": "ULAW", "prob": 0.3, "minimum": 250, "maximum": 260}]},
        {"codecs": [{"encoding": "ULAW", "prob": 0.7, "minimum": 250, "maximum": 260}]},
        {"codecs": [{"encoding": "ULAW", "prob": 0.7, "minimum": 240, "maximum": 270}]},
        {"codecs": [{"encoding": "ULAW", "prob": 1, "minimum": 250, "maximum": 260}]},
        {"tempo": {"prob": 0.3, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.8, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.3, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 0.8, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 1, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3}},
        {"speed": {"prob": 0.5, "minimum": 0.88, "maximum": 1.12}},
        {"speed": {"prob": 0.3, "minimum": 0.88, "maximum": 1.12}},
        {"speed": {"prob": 0.7, "minimum": 0.88, "maximum": 1.12}},
        {"speed": {"prob": 0.6, "minimum": 0.8, "maximum": 1.2}},
        {"speed": {"prob": 0.6, "minimum": 0.9, "maximum": 1.1}},
        {"speed": {"prob": 1, "minimum": 0.9, "maximum": 1.1}},
        {"speed": {"prob": 1, "minimum": 0.8, "maximum": 1.2}},
        {"non_linearity": {"prob": 0.7, "minimum": 0.9, "maximum": 1.1}},
        {"non_linearity": {"prob": 0.3, "minimum": 0.9, "maximum": 1.1}},
        {"non_linearity": {"prob": 1, "minimum": 0.9, "maximum": 1.1}},
        {"preemphasis": {"prob": 0.3, "minimum": 0.94, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.94, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 1.0, "minimum": 0.94, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.90, "maximum": 1.0, "default": 0.97}},
        {"preemphasis": {"prob": 0.7, "minimum": 0.90, "maximum": 1.0, "default": 0.95}},
    ]

    nn_base_args = {}

    for args in perturbation_args:
        exp_name_suffix, report_args = args_to_key_and_report_strings(args)

        # Construct the exp_name and report_args
        exp_name = f"mel_bs2x5k_perturb_from_checkpoint_24_{exp_name_suffix}"
        nn_base_args[exp_name] = dict(
            returnn_args={
                "extra_args": {
                    **returnn_args["extra_args"],
                    "audio_perturb_args": args,
                    "preload_from_files": {
                        "existing-model": {
                            "filename": nn_system.train_jobs["conformer_bs2x5k_lgm80_baseline"].out_checkpoints[24],
                            "init_for_train": True,
                        }
                    },
                },
                **returnn_args,
            },
            feature_args={
                "class": "LogMelNetwork",
                "wave_norm": True,
                "frame_size": 200,
                "frame_shift": 80,
                "fft_size": 256,
            },
            lr_args=lr_args,
            report_args=report_args,
        )

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args=nn_base_args,
        num_epochs=426,
        evaluation_epochs=[376, 386, 396, 406, 426],
        prefix="conformer_",
    )
    report, _ = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        "report_mel_audio_perturbation_from_checkpoint24.csv",
        recog_args={"epochs": [376, 386, 396, 406, 426]},
    )
    return report


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    report_mel, _ = run_mel_baseline()
    report_scf, _ = run_scf_baseline()
    report_scf_specaug_sort = run_scf_specaug_sort()
    report_scf_audio_perturbation_from_checkpoint = run_scf_audio_perturbation_from_checkpoint()
    report_mel_audio_perturbation_from_checkpoint = run_mel_audio_perturbation_from_checkpoint()

    report_base = Report(
        columns_start=["train_name", "batch_size"],
        columns_end=["epoch", "lm_scale", "prior_scale"],
    )
    report = Report.merge_reports(
        [
            report_base,
            report_mel,
            report_scf,
            report_scf_specaug_sort,
            report_scf_audio_perturbation_from_checkpoint,
            report_mel_audio_perturbation_from_checkpoint,
        ]
    )
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
