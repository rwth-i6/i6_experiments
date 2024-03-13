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
from .helpers.lr.fullsum import dynamic_learning_rate as dynamic_learning_rate_fullsum
from .default_tools import RASR_BINARY_PATH, RASR_BINARY_PATH_PRECISION, RETURNN_ROOT, RETURNN_ROOT_FULLSUM, RETURNN_EXE, SCTK_BINARY_PATH


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
        "epochs": [401],
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
    ctc_nn_system.rasr_binary_path = RASR_BINARY_PATH_PRECISION
    ctc_nn_system.base_crp.set_executables(rasr_binary_path=ctc_nn_system.rasr_binary_path)
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

    nn_system = TransducerSystem(
        returnn_root=returnn_root or RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
        require_native_lstm=False,
    )
    nn_system.init_system(
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
        report=Report(),
    )
    nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_from_lexicon = False
    nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_all = True
    nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_from_file = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/allophones",
        hash_overwrite="SWB_ALLOPHONE_FILE_WEI",
        cached=True,
    )
    nn_system.run_train_step(nn_args.training_args)
    nn_system.run_dev_recog_step(recog_args=recog_args, report_args=report_args_collection)

    assert nn_system.report is not None
    report = nn_system.report
    report.delete_redundant_rows()
    if report_name:
        tk.register_report(
            os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, report_name),
            values=report.get_values(),
            template=report.get_template(),
        )
    return nn_system, report


def run_rasr_gt_stage1():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/transducer/feat/"

    _, dev_corpora, _ = get_switchboard_data()
    returnn_datasets = get_returnn_datasets_transducer_viterbi(
        features="wei",
        alignment="wei",
        context_window={"classes": 1, "data": 121},
    )
    returnn_datasets_legacy = get_returnn_datasets_transducer_viterbi(
        features="wei",
        alignment="wei",
        context_window={"classes": 1, "data": 121},
        legacy_feature_dump=True,
    )
    returnn_datasets_hash_break = get_returnn_datasets_transducer_viterbi(
        features="wei",
        alignment="wei",
        context_window={"classes": 1, "data": 121},
        keep_hashes=False,
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
            "warp_freqbreak": 3700,
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
    common_args = {
        "num_inputs": 40,
        "lr_args": {"dynamic_learning_rate": dynamic_learning_rate},
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs15k_v0": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {"max_feature": 4},
                    **returnn_args,
                    **{"datasets": returnn_datasets_legacy},
                },
                report_args={"alignment": "wei", "dataset_hash": "legacy-0"},
                **common_args,
            ),
            "bs15k_v0.1": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 4}, **returnn_args},
                report_args={"alignment": "wei", "dataset_hash": "legacy-1"},
                **common_args,
            ),
            "bs15k_v1": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {"max_feature": 4},
                    **returnn_args,
                    **{"datasets": returnn_datasets_hash_break},
                },
                report_args={"alignment": "wei", "dataset_hash": "legacy-0"},
                **common_args,
            ),
        },
        num_epochs=300,
        evaluation_epochs=[270, 280, 290, 300],
        prefix="viterbi_rasrgt_",
    )
    nn_system, report = run_nn_args(nn_args, report_args_collection, dev_corpora["transducer"], recog_args=recog_args)
    return nn_system, report


def run_mel_stage1():
    ctc_alignment = get_ctc_alignment()

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/transducer/feat/"
    _, dev_corpora, _ = get_switchboard_data()
    returnn_datasets = get_returnn_datasets_transducer_viterbi(context_window={"classes": 1, "data": 121})
    returnn_datasets_hash_break = get_returnn_datasets_transducer_viterbi(
        context_window={"classes": 1, "data": 121},
        keep_hashes=False,
    )
    returnn_datasets_align_ctc = get_returnn_datasets_transducer_viterbi(
        alignment=ctc_alignment,
        features="waveform_pcm",
    )
    returnn_args = {
        "batch_size": 15000,
        "datasets": returnn_datasets,
        "extra_args": {
            # data sequence is longer by factor 4 because of subsampling and 80 because of feature extraction vs.
            # raw waveform
            "chunking": ({"classes": 64, "data": 64 * 4 * 80}, {"classes": 32, "data": 32 * 4 * 80}),
            "gradient_clip": 20.0,
            "learning_rate_control_error_measure": "sum_dev_score",
            "min_learning_rate": 1e-6,
        },
        "specaug_old": {"max_feature": 8},
    }
    returnn_args_ctc_align = copy.deepcopy(returnn_args)
    returnn_args_ctc_align["datasets"] = returnn_datasets_align_ctc
    returnn_args_ctc_align["extra_args"]["extern_data"] = {
        "data": {"dim": 1, "dtype": "int16"},
        "classes": {"dim": 88, "dtype": "int8", "sparse": True},
    }
    returnn_args_ctc_align["extra_args"]["min_chunk_size"] = {"classes": 1, "data": 200}
    # Data sequence is longer by factor 4 because of subsampling and 80 because of feature extraction vs.
    # raw waveform. Also, there are frame size - frame shift more samples at the end. This should be more correct than
    # the version above.
    returnn_args_ctc_align["extra_args"]["chunking"] = (
        {"classes": 64, "data": 64 * 4 * 80 + 200 - 80},
        {"classes": 32, "data": 32 * 4 * 80},
    )
    feature_args = {
        "class": "LogMelNetwork",
        "wave_norm": True,
        "frame_size": 200,
        "frame_shift": 80,
        "fft_size": 256,
    }
    common_args = {
        "feature_args": feature_args,
        "lr_args": {"dynamic_learning_rate": dynamic_learning_rate},
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs15k_v0": dict(
                returnn_args=returnn_args,
                report_args={"alignment": "wei", "dataset_hash": "legacy-1"},
                **common_args,
            ),
            "bs15k_v1": dict(
                returnn_args={**returnn_args, **{"datasets": returnn_datasets_hash_break}},
                report_args={"alignment": "wei"},
                **common_args,
            ),
            "bs15k_v1_align-ctc-conf-e401": dict(
                returnn_args=returnn_args_ctc_align,
                report_args={"alignment": "ctc-conf-e401"},
                lr_args={"dynamic_learning_rate": dynamic_learning_rate},
                feature_args={"wave_cast": True, **feature_args}
            ),
        },
        num_epochs=300,
        evaluation_epochs=[270, 280, 290, 300],
        prefix="viterbi_lgm80_",
    )
    config = copy.deepcopy(nn_args.returnn_recognition_configs["viterbi_lgm80_bs15k_v1_align-ctc-conf-e401"].config)
    config["extern_data"]["data"]["dtype"] = "float32"
    config["extern_data"]["classes"]["dtype"] = "int32"
    nn_args.returnn_recognition_configs["viterbi_lgm80_bs15k_v1_align-ctc-conf-e401"].config = config
    nn_system, report = run_nn_args(nn_args, report_args_collection, dev_corpora["transducer"])
    return nn_system, report


def run_mel_stage2():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/transducer/feat/"

    nn_system_stage1, _ = run_mel_stage1()
    _, dev_corpora, _ = get_switchboard_data()
    returnn_datasets = get_returnn_datasets_transducer_viterbi(keep_hashes=False)
    returnn_args = {
        "batch_size": 3000,
        "datasets": returnn_datasets,
        "extra_args": {
            "accum_grad_multiple_step": 3,
            "gradient_clip": 0.0,
            "min_learning_rate": 1e-6,
            "max_seq_length": {'classes': 600},
            "preload_from_files": {
                "viterbi": {
                    "filename": nn_system_stage1.train_jobs["viterbi_lgm80_bs15k_v1"].out_checkpoints[280],
                    "ignore_missing": True,
                    "init_for_train": True,
                },
            },
        },
        "specaug_old": {"max_feature": 8},
        "rasr_loss_args": {"transducer_training_stage": "fullsum"},
        "conformer_args": {"dropout": 0.25, "batch_norm_freeze": True},
    }
    returnn_args_keep_hash = copy.deepcopy(returnn_args)
    returnn_args_keep_hash["extra_args"]["learning_rate_control_error_measure"] = "sum_dev_score"
    feature_args = {
        "class": "LogMelNetwork",
        "wave_norm": True,
        "frame_size": 200,
        "frame_shift": 80,
        "fft_size": 256,
    }
    recog_args = {
        "lm_scales": [0.25, 0.3, 0.35, 0.4, 0.45],
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
        "epochs": [200, 210, 220, 230, 240, "best"],
    }
    common_args = {
        "feature_args": feature_args,
        "lr_args": {"dynamic_learning_rate": dynamic_learning_rate_fullsum},
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "bs15k_v1": dict(
                returnn_args=returnn_args_keep_hash,
                report_args={"stage": "fullsum"},
                **common_args,
            ),
        },
        num_epochs=240,
        evaluation_epochs=[200, 210, 220, 230, 240],
        prefix="fullsum_lgm80_",
    )
    nn_system, report = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora["transducer"],
        returnn_root=RETURNN_ROOT_FULLSUM,
        recog_args=recog_args,
    )
    return nn_system, report


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    _, report_rasr_gt_stage1 = run_rasr_gt_stage1()
    _, report_mel_stage1 = run_mel_stage1()
    _, report_mel_stage2 = run_mel_stage2()

    report_base = Report(
        columns_start=["train_name", "features", "alignment"],
        columns_end=["epoch", "recog_name", "lm_scale", "ins", "del", "sub", "wer"],
    )
    report = Report.merge_reports([
        report_base,
        report_rasr_gt_stage1,
        report_mel_stage1,
        report_mel_stage2,
    ])
    report.delete_redundant_columns()
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report_swb_transducer.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
