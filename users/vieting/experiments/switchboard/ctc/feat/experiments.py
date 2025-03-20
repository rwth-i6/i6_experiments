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
        if key in ["speed", "tempo", "pitch", "preemphasis", "non_linearity"]:
            key_component = f"{key}_{value['prob']}_{value['minimum']}_{value['maximum']}"
            if "default" in value:
                key_component += f"_{value['default']}"
            key_string += key_component
            report_dict[key] = f"{value['prob']}_{value['minimum']}_{value['maximum']}"
        elif key == "codecs":
            codecs_str_list = []
            for codec in value:
                codec_str = f"{codec['encoding']}_{codec['prob']}_{codec['minimum']}_{codec['maximum']}"
                codecs_str_list.append(codec_str)
            codecs_str = "_".join(codecs_str_list)
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
            "bs2x5k_lgm80_baseline_preemphasis97": dict(
                returnn_args={
                    **returnn_args,
                    "batch_size": 5000,
                    "extra_args": {
                        "accum_grad_multiple_step": 2,
                        "watch_memory": True,
                    },
                },
                feature_args={**feature_args, "preemphasis": 0.97},
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
                        24
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


def run_mel_baseline_lr_reset():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"
    _, nn_system = run_mel_baseline()
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
            "preload_from_files": {
                "existing-model": {
                    "filename": nn_system.train_jobs["conformer_bs2x5k_lgm80_baseline"].out_checkpoints[24],
                    "init_for_train": True,
                },
            },
        },
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

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs2x5k_lgm80_baseline_lr_reset": dict(
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
    report, ctc_nn_system = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
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
        {"codecs": [{"encoding": "ULAW", "prob": 0.3, "minimum": 1, "maximum": 10, "default": None}]},
        {"codecs": [{"encoding": "ULAW", "prob": 0.7, "minimum": 1, "maximum": 10, "default": None}]},
        {"codecs": [{"encoding": "ULAW", "prob": 0.7, "minimum": 1, "maximum": 20, "default": None}]},
        {"codecs": [{"encoding": "ULAW", "prob": 1, "minimum": 1, "maximum": 10, "default": None}]},
        {"tempo": {"prob": 0.3, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.8, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.3, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 0.8, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 1, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3}},
        {"pitch": {"prob": 0.3, "minimum": -2, "maximum": 2}},
        {"pitch": {"prob": 0.3, "minimum": -3, "maximum": 3}},
        {"pitch": {"prob": 0.7, "minimum": -2, "maximum": 2}},
        {"pitch": {"prob": 0.7, "minimum": -3, "maximum": 3}},
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
                **returnn_args,
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
                **returnn_args,
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
                **returnn_args,
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
            },
            feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2, "wave_norm": True},
            lr_args=lr_args,
            report_args=report_args,
        )

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args=nn_base_args,
        num_epochs=426,
        evaluation_epochs=[376, 386, 396, 406, 426],
        prefix="conformer_",
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
        "report_scf_audio_perturbation_from_checkpoint24.csv",
        returnn_root=returnn_root,
        recog_args={"epochs": [376, 386, 396, 406, 426]},
    )
    return report


def run_scf_specaug():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets()
    base_returnn_args = {
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
    }
    feature_args = {
        "class": "ScfNetwork",
        "size_tf": 256 // 2,
        "stride_tf": 10 // 2,
        "wave_norm": True,
        "preemphasis": 0.97,
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
            "baseline": dict(
                returnn_args={
                    **base_returnn_args,
                    "specaug_config": {"enable_sorting": False, "max_feature": 15, "steps_per_epoch": 4100},
                },
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k", "enable_sorting": False, "max_feature": 15},
            ),
            # note this only reduces the masking in frequency dimension, not in time.
            "baseline_increase_flag": dict(
                returnn_args={
                    **base_returnn_args,
                    "specaug_config": {
                        "enable_sorting": False,
                        "max_feature": 15,
                        "max_feature_num": 2,
                        "steps_per_epoch": 4100,
                        "freq_mask_num_schedule": {0: 1, 1: 2.5},
                    },
                },
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k", "enable_sorting": False, "max_feature": 15},
            ),
            "variance_based_specaug": dict(
                returnn_args={
                    **base_returnn_args,
                    "specaug_config": {
                        "steps_per_epoch": 4100,
                        "enable_sorting": False,
                        "filter_based_masking_strategy": "variance",
                        "enable_logging": True,
                        "filter_factor": 0.5,
                        "max_number_masks_for_filter_based_specaug": 75,
                    },
                },
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={
                    "batch_size": "2x5k",
                    "filter_based_masking_strategy": "variance",
                    "filter_factor": 0.5,
                    "max_number_masks_for_filter_based_specaug": 75,
                },
            ),
            "peakToAverage_based_specaug": dict(
                returnn_args={
                    **base_returnn_args,
                    "specaug_config": {
                        "steps_per_epoch": 4100,
                        "enable_sorting": False,
                        "filter_based_masking_strategy": "peakToAverage",
                        "enable_logging": True,
                        "filter_factor": 0.5,
                        "max_number_masks_for_filter_based_specaug": 75,
                    },
                },
                feature_args=feature_args,
                lr_args=lr_args,
                report_args={
                    "batch_size": "2x5k",
                    "filter_based_masking_strategy": "peakToAverage",
                    "filter_factor": 0.5,
                    "max_number_masks_for_filter_based_specaug": 75,
                },
            ),
        },
        num_epochs=450,
        evaluation_epochs=[350, 390, 400, 410, 450],
        prefix="conformer_bs2x5k_scf_specaug_",
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
        "report_scf_specaug.csv",
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
                    "batch_size_decay_epoch": 24,
                    "batch_size_start": 5000,
                    "batch_size_end": 10000,
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

    feature_args = {"class": "LogMelNetwork", "wave_norm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256}

    perturbation_args = [
        {"tempo": {"prob": 0.3, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.8, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 0.3, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 0.8, "minimum": 0.7, "maximum": 1.3}},
        {"tempo": {"prob": 1, "minimum": 0.83, "maximum": 1.17}},
        {"tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3}},
        {"pitch": {"prob": 0.3, "minimum": -2, "maximum": 2}},
        {"pitch": {"prob": 0.3, "minimum": -3, "maximum": 3}},
        {"pitch": {"prob": 0.7, "minimum": -2, "maximum": 2}},
        {"pitch": {"prob": 0.7, "minimum": -3, "maximum": 3}},
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
            feature_args=feature_args,
            lr_args=lr_args,
            report_args=report_args,
        )

    for args in perturbation_args_preemphasis:
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
                            "filename": nn_system.train_jobs["bs2x5k_lgm80_baseline_preemphasis97"].out_checkpoints[24],
                            "init_for_train": True,
                        }
                    },
                },
                **returnn_args,
            },
            feature_args=feature_args,
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


def run_specaug_stft_experiments():
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
            "conv_pad_seq_len_to_power": 1.5,
        },
        "conformer_type": "wei",
    }
    feature_args_scf = {"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2, "preemphasis": 0.97}
    feature_args_lgm = {
        "class": "LogMelNetwork",
        "wave_norm": True,
        "frame_size": 200,
        "frame_shift": 80,
        "fft_size": 256,
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
            "bs2x5k_scf_stft20ms_time_only": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {
                        "max_feature": 0,
                        "max_feature_num": 0,
                        "frame_size": 400,
                        "frame_shift": 160,
                        "fft_size": 512,
                    },
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_stft20ms_fmask_1_1of512": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {
                        "max_feature": 1,
                        "max_feature_num": 1,
                        "frame_size": 400,
                        "frame_shift": 160,
                        "fft_size": 512,
                    },
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_stft20ms_fmask_2_4of512": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {
                        "max_feature": 4,
                        "max_feature_num": 2,
                        "frame_size": 400,
                        "frame_shift": 160,
                        "fft_size": 512,
                    },
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_stft20ms_fmask_5_8of512": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 8, "frame_size": 400, "frame_shift": 160, "fft_size": 512},
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_scf_stft20ms_fmask_5_15of512": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 15, "frame_size": 400, "frame_shift": 160, "fft_size": 512},
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "2x5k"},
            ),
            "bs2x5k_lgm_stft20ms_fmask_5_8of512": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 8, "frame_size": 400, "frame_shift": 160, "fft_size": 512},
                    "extra_args": {"accum_grad_multiple_step": 2},
                },
                feature_args=feature_args_lgm,
                lr_args=lr_args,
                report_args={
                    "batch_size": "2x5k",
                },
            ),
            "bs10k_lgm_stft20ms_fmask_5_8of512": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 8, "frame_size": 400, "frame_shift": 160, "fft_size": 512},
                    "batch_size": 10000,
                    "extra_args": {},
                },
                feature_args=feature_args_lgm,
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
            "bs10k_scf_stft10ms_fmask_5_8of256": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 8},
                    "batch_size": 10000,
                    "extra_args": {
                        "conv_pad_seq_len_to_power": 1.5,
                    },
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
            "bs10k_lgm_stft10ms_fmask_5_8of256": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 8},
                    "batch_size": 10000,
                    "extra_args": {},
                },
                feature_args=feature_args_lgm,
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
            "bs10k_scf_stft10ms_fmask_5_4of256": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 4},
                    "batch_size": 10000,
                    "extra_args": {
                        "conv_pad_seq_len_to_power": 1.5,
                    },
                },
                feature_args=feature_args_scf,
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
            "bs10k_lgm_stft10ms_fmask_5_4of256": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 4},
                    "batch_size": 10000,
                    "extra_args": {},
                },
                feature_args=feature_args_lgm,
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
            # tested for paper since perturbation exps used alpha=1, result same as with 0.97
            "bs10k_scf_stft10ms_fmask_5_8of256_pre1": dict(
                returnn_args={
                    **returnn_args,
                    "specaug_stft": {"max_feature": 8},
                    "batch_size": 10000,
                    "extra_args": {"conv_pad_seq_len_to_power": 1.5},
                },
                feature_args={**feature_args_scf, "preemphasis": 1.0},
                lr_args=lr_args,
                report_args={"batch_size": "10k"},
            ),
        },
        num_epochs=450,
        evaluation_epochs=[24, 350, 390, 400, 410, 420, 430, 440, 450],
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
        "report_specaug_stft.csv",
        returnn_root=returnn_root,
        recog_args={"epochs": [24, 350, 390, 400, 410, 420, 430, 440, 450]},
    )
    return report, ctc_nn_system


def run_scf_combination_experiments():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    _, nn_system_stft_specaug = run_specaug_stft_experiments()

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

    feature_args_scf = {"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2, "preemphasis": 0.97}

    def _preload_dict_helper(train_name, epoch):
        return {
            "existing-model": {
                "filename": nn_system_stft_specaug.train_jobs[train_name].out_checkpoints[epoch],
                "init_for_train": True,
            },
        }

    nn_base_args = {
        # Final result in Max' Thesis 
        "scf_bs2x5k_stft20ms_fmask_5_8of512_tempo": dict(
            returnn_args={
                **returnn_args,
                "specaug_stft": {
                    "max_feature": 8,
                    "frame_size": 400,
                    "frame_shift": 160,
                    "fft_size": 512,
                },
                "extra_args": {
                    **returnn_args["extra_args"],
                    "audio_perturb_args": {"tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3}},
                    "preload_from_files": _preload_dict_helper("conformer_bs2x5k_scf_stft20ms_fmask_5_8of512", 24),
                },
            },
            feature_args=feature_args_scf,
            lr_args=lr_args,
        ),
        # tested for paper, but performed worse than just tempo
        "scf_bs10k_stft10ms_fmask_5_8of256_tempo_nonlinear_preemphasis": dict(
            returnn_args={
                **returnn_args,
                "specaug_stft": {"max_feature": 8},
                "batch_size": 10000,
                "extra_args": {
                    "conv_pad_seq_len_to_power": 1.5,
                    "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    "audio_perturb_args": {
                        "tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3},
                        "preemphasis": {"prob": 0.7, "minimum": -0.05, "maximum": 0.05, "default": 0.0},
                        "non_linearity": {"prob": 1, "minimum": 0.9, "maximum": 1.1, "default": 1},
                    },
                    "preload_from_files": _preload_dict_helper("conformer_bs10k_scf_stft10ms_fmask_5_8of256", 24),
                },
            },
            feature_args=feature_args_scf,
            lr_args=lr_args,
        ),
        # Used in paper as combined result
        "scf_bs10k_stft10ms_fmask_5_8of256_tempo": dict(
            returnn_args={
                **returnn_args,
                "specaug_stft": {"max_feature": 8},
                "batch_size": 10000,
                "extra_args": {
                    "conv_pad_seq_len_to_power": 1.5,
                    "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    "audio_perturb_args": {
                        "tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3},
                    },
                    "preload_from_files": _preload_dict_helper("conformer_bs10k_scf_stft10ms_fmask_5_8of256", 24),
                },
            },
            feature_args=feature_args_scf,
            lr_args=lr_args,
        ),
        # Tested since other audio perturbation results are unsing preemphasis 1 
        "scf_bs10k_stft10ms_fmask_5_8of256_tempo_pre1": dict(
            returnn_args={
                **returnn_args,
                "specaug_stft": {"max_feature": 8},
                "batch_size": 10000,
                "extra_args": {
                    "conv_pad_seq_len_to_power": 1.5,
                    "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    "audio_perturb_args": {
                        "tempo": {"prob": 1, "minimum": 0.7, "maximum": 1.3},
                    },
                    "preload_from_files": _preload_dict_helper("conformer_bs10k_scf_stft10ms_fmask_5_8of256_pre1", 24),
                },
            },
            feature_args={**feature_args_scf, "preemphasis": 1.0},
            lr_args=lr_args,
        ),
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args=nn_base_args,
        num_epochs=426,
        evaluation_epochs=[24, 376, 386, 396, 406, 426],
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
        recog_args={"epochs": [376, 386, 396, 406, 426]},
    )
    return report, ctc_nn_system


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    report_mel, _ = run_mel_baseline()
    report_scf, _ = run_scf_baseline()
    report_scf_specaug_sort = run_scf_specaug_sort()
    report_scf_audio_perturbation_from_checkpoint = run_scf_audio_perturbation_from_checkpoint()
    report_mel_audio_perturbation_from_checkpoint = run_mel_audio_perturbation_from_checkpoint()
    report_specaug_stft = run_specaug_stft_experiments()

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
            report_specaug_stft,
        ]
    )
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
