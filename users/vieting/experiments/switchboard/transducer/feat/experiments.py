import copy
import os.path
from typing import List
from sisyphus import tk, gs

from i6_core.lexicon import DumpStateTyingJob, StoreAllophonesJob
from i6_core.recognition import Hub5ScoreJob
from i6_experiments.users.vieting.tools.report import Report
from i6_experiments.users.berger.helpers.hdf import build_hdf_from_alignment
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.experiments import (
    run_mel_baseline as run_mel_baseline_ctc,
)
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import (
    TransducerSystem,
    ReturnnConfigs,
    ScorerInfo,
)
from .data import get_switchboard_data, get_returnn_datasets_transducer_viterbi
from .baseline_args import get_nn_args as get_nn_args_baseline
from .helpers.lr.oclr import dynamic_learning_rate
from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE, SCTK_BINARY_PATH


def get_ctc_alignment() -> List[tk.Path]:
    train_corpus, dev_corpora, _ = get_switchboard_data()
    _, ctc_nn_system = run_mel_baseline_ctc()
    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 1,
        "tdp_transition": (0, 0, "infinity", 0),
        "tdp_silence": (0, 0, "infinity", 0),
        "phon_history_length": 0,
        "phon_future_length": 0,
    }
    align_args = {
        "epochs": ["best"],
        "lm_scales": [0.7],
        "prior_scales": [0.3],
        "use_gpu": False,
        "alignment_options": {
            "label-pruning": 50,
            "label-pruning-limit": 100000,
        },
        "align_node_options": {
            "allophone-state-graph-builder.topology": "rna",  # No label loop for transducer
        },
        "label_scorer_args": {
            "use_prior": True,
            "extra_args": {
                "blank_label_index": 0,
            },
        },
        "rtf": 5,
    }
    train_corpus.concurrent = 100
    train_corpus.lexicon = dev_corpora["ctc"]["hub5e00"].lexicon
    ctc_nn_system.corpus_data["train"] = train_corpus
    ctc_nn_system.crp["train"] = ctc_nn_system.get_crp(train_corpus, am_args)
    allophone_file = StoreAllophonesJob(crp=ctc_nn_system.crp["train"]).out_allophone_file
    state_tying_job = DumpStateTyingJob(ctc_nn_system.crp["train"]).out_state_tying
    ctc_nn_system.crp["train"].acoustic_model_config.allophones.add_from_file = allophone_file
    ctc_nn_system.align_corpora = ["train"]
    ctc_nn_system.run_align_step(align_args)
    alignment = build_hdf_from_alignment(
        alignment_cache=ctc_nn_system.alignments["train"].alternatives["bundle"],
        allophone_file=allophone_file,
        state_tying_file=state_tying_job,
        silence_phone="<blank>",
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
    )
    return [alignment]


def run_nn_args(nn_args, report_args_collection, dev_corpora, report_name="", returnn_root=None, recog_args=None):
    returnn_configs = {}
    for exp in nn_args.returnn_training_configs:
        prior_config = copy.deepcopy(nn_args.returnn_training_configs[exp])
        if isinstance(prior_config.config["batch_size"], dict):
            prior_config.config["batch_size"] = prior_config.config["batch_size"]["data"]
        assert isinstance(prior_config.config["batch_size"], int)
        returnn_configs[exp] = ReturnnConfigs(
            train_config=nn_args.returnn_training_configs[exp],
            prior_config=prior_config,
            recog_configs={"recog": nn_args.returnn_recognition_configs[exp]},
        )

    recog_args = {
        **{
            "lm_scales": [0.55],
            "epochs": [270, 280, 290, 300, "best"],
            "lookahead_options": {"lm_lookahead_scale": 0.55},
            "label_scorer_args": {
                "extra_args": {
                    "blank-label-index": 0,
                    "context-size": 1,
                    "label-scorer-type": "tf-ffnn-transducer",
                    "max-batch-size": 256,
                    "reduction-factors": 80 * 4,
                    "reduction-subtrahend": 200 - 1,  # STFT window size - 1
                    "start-label-index": 89,
                    "transform-output-negate": True,
                    "use-start-label": True,
                },
            },
            "label_tree_args": {"skip_silence": True},
            "search_parameters": {
                "allow-blank-label": True,
                "allow-label-recombination": True,
                "allow-word-end-recombination": True,
                "create-lattice": True,
                "full-sum-decoding": True,
                "label-pruning": 8.8,
                "label-pruning-limit": 50000,
                "recombination-lm.type": "simple-history",
                "separate-recombination-lm": True,
                "word-end-pruning": 0.5,
                "word-end-pruning-limit": 5000,
            },
            "label_scorer_type": "tf-ffnn-transducer",
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
        "/u/vieting/setups/swb/20230406_feat/dependencies/allophones",
        hash_overwrite="SWB_ALLOPHONE_FILE_WEI",
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
    return report


def run_rasr_gt_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/transducer/feat/"

    _, dev_corpora, _ = get_switchboard_data()
    returnn_datasets = get_returnn_datasets_transducer_viterbi(
        features="wei",
        alignment="wei",
        context_window={"classes": 1, "data": 121},
    )
    returnn_args = {
        "batch_size": 15000,
        "datasets": returnn_datasets,
        "extra_args": {
            # data sequence is longer by factor 4 because of subsampling
            "chunking": ({"classes": 64, "data": 64 * 4}, {"classes": 32, "data": 32 * 4}),
            "gradient_clip": 20.0,
            "learning_rate_control_error_measure": "sum_dev_score",
            "min_learning_rate": 1e-6,
        },
    }
    recog_args = {
        "flow_args": {
            "type": "gammatone",
            "channels": 40,
            "maxfreq": 3800,
            "warp_freqbreak":3700,
            "do_specint": False,
            "add_features_output": True,
        },
        "label_scorer_args": {
            "extra_args": {
                "blank-label-index": 0,
                "context-size": 1,
                "label-scorer-type": "tf-ffnn-transducer",
                "max-batch-size": 256,
                "reduction-factors": "2,2",
                "start-label-index": 89,
                "transform-output-negate": True,
                "use-start-label": True,
            },
        },
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs15k_v0": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 4}, **returnn_args},
                num_inputs=40,
                lr_args={"dynamic_learning_rate": dynamic_learning_rate},
                report_args={
                    "architecture": "conf-wei",
                    "lr": "1e-3",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                },
            ),
        },
        num_epochs=300,
        evaluation_epochs=[270, 280, 290, 300],
        prefix="viterbi_rasrgt_",
    )
    report = run_nn_args(nn_args, report_args_collection, dev_corpora["transducer"], recog_args=recog_args)
    return report


def run_mel_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/transducer/feat/"
    reports = []

    _, dev_corpora, _ = get_switchboard_data()
    for alignment in ["wei", "ctc"]:
        prefix = "viterbi_lgm80_"
        if alignment == "ctc":
            prefix += f"align-ctc_"
            alignment = get_ctc_alignment()
        returnn_datasets = get_returnn_datasets_transducer_viterbi(
            alignment=alignment,
            context_window={"classes": 1, "data": 121},
        )
        returnn_args = {
            "batch_size": 15000,
            "datasets": returnn_datasets,
            "extra_args": {
                # data sequence is longer by factor 4 because of subsampling and 80 because of feature extraction vs. wave
                "chunking": ({"classes": 64, "data": 64 * 4 * 80}, {"classes": 32, "data": 32 * 4 * 80}),
                "gradient_clip": 20.0,
                "learning_rate_control_error_measure": "sum_dev_score",
                "min_learning_rate": 1e-6,
            },
        }
        feature_args = {"class": "LogMelNetwork", "wave_norm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256}

        nn_args, report_args_collection = get_nn_args_baseline(
            nn_base_args={
                "bs15k_v0": dict(
                    returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                    feature_args=feature_args,
                    lr_args={"dynamic_learning_rate": dynamic_learning_rate},
                    report_args={
                        "architecture": "conf-wei",
                        "lr": "1e-3",
                        "specaug": "wei_adapt_80dim",
                        "wave_norm": "True",
                    },
                ),
            },
            num_epochs=300,
            evaluation_epochs=[270, 280, 290, 300],
            prefix=prefix,
        )
        report = run_nn_args(nn_args, report_args_collection, dev_corpora["transducer"])
        reports.append(report)
    return Report.merge_reports(reports)


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    report_rasr_gt = run_rasr_gt_baseline()
    report_mel = run_mel_baseline()

    report_base = Report(
        columns_start=["train_name", "batch_size"],
        columns_end=["epoch", "recog_name", "lm", "optlm", "lm_scale", "prior_scale"],
    )
    report = Report.merge_reports([
        report_base,
        report_rasr_gt,
        report_mel,
    ])
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report_swb_transducer.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
