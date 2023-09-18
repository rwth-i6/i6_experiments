import copy
import os.path
from sisyphus import tk, gs

from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.vieting.tools.report import Report
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import (
    ScorerInfo,
    TransducerSystem,
    ReturnnConfigs,
)

from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.baseline_args import (
    get_nn_args as get_nn_args_baseline
)
from .ctc_data import get_librispeech_data
from .default_tools import RASR_BINARY_PATH_APPTAINER, RETURNN_ROOT, RETURNN_EXE_APPTAINER


def run_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_960_ctc/feat/"

    # TODO: create in graph: allophone file, lexicon
    (
        returnn_datasets, rasr_loss_corpus_path, rasr_loss_lexicon_path, dev_corpora
    ) = get_librispeech_data(augmented_lexicon=True, cv_segments_wei=True)
    (
        returnn_datasets_new, rasr_loss_corpus_path_new, rasr_loss_lexicon_path_new, _
    ) = get_librispeech_data(augmented_lexicon=True, ogg_no_conversion=False)
    allophone_file = tk.Path(
        "/work/asr4/vieting/setups/librispeech/work/allophones/StoreAllophones.NRKTz6cpvJMU/output/allophones",
        hash_overwrite="wei_librispeech_allophone_file",
    )
    rasr_loss_lexicon_path = tk.Path(
        "/work/asr4/vieting/setups/librispeech/dependencies/lexicon/align-dev.blank-rep.lexicon.blank.manualg2p.xml",
        hash_overwrite="wei_librispeech_loss_lexicon_g2p"
    )
    returnn_args = {
        "batch_size": {"data": 640000},
        "rasr_binary_path": RASR_BINARY_PATH_APPTAINER,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": None,
        "rasr_loss_corpus_prefix": "loss-corpus/",
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "am_args": {
            "state_tying": "monophone-eow",
            "states_per_phone": 1,
            "tdp_transition": (0.0, 0.0, "infinity", 0.0),
            "tdp_silence": (0.0, 0.0, "infinity", 0.0),
            "phon_history_length": 0,
            "phon_future_length": 0,
            "allophone_file": allophone_file,
        },
        "datasets": returnn_datasets,
        "extra_args": {
            "accum_grad_multiple_step": 3,
        },
        "specaug_old": {"max_feature": 8},  # use old SpecAugment implementation
    }
    returnn_args_new = copy.deepcopy(returnn_args)
    returnn_args_new.update({
        "rasr_loss_corpus_path": rasr_loss_corpus_path_new,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path_new,
        "datasets": returnn_datasets_new,
    })
    returnn_datasets_old = copy.deepcopy(returnn_datasets)
    returnn_datasets_old["train"]["path"] = tk.Path(
        "/work/asr4/vieting/setups/librispeech/work/crnn/oggzip/BlissToOggZipJob.PMoMRrqXpZWl/output/out.ogg.zip",
        hash_overwrite="librispeech_960_train_old.ogg.zip",
    )
    returnn_datasets_old["dev"]["path"] = tk.Path(
        "/work/asr4/vieting/setups/librispeech/work/crnn/oggzip/BlissToOggZipJob.eyYjJHZFdOwV/output/out.ogg.zip",
        hash_overwrite="librispeech_960_dev_old.ogg.zip",
    )
    returnn_args_old = copy.deepcopy(returnn_args)
    returnn_args_old.update({
        "rasr_loss_corpus_path": tk.Path(
            "/work/asr4/zhou/data/librispeech/am-data/corpus/train-dev.corpus.xml",
            hash_overwrite="librispeech_960_train_dev_corpus.xml",
        ),
        "datasets": returnn_datasets_old,
    })
    feature_args = {
        "log_mel": {
            "class": "LogMelNetwork",
            "wave_norm": True,
        },
        "gammatone": {
            "class": "GammatoneNetwork",
            "wave_norm": True,
            "preemphasis": 1.0,
        },
        "scf": {
            "class": "ScfNetwork",
            "wave_norm": True,
        },
    }
    lr_args = {
        "peak_lr": 3e-4, "start_lr": 1e-3 * (0.01 * 224 + 0.3) / 225, "end_lr": 1e-5,
        "increase_epochs": 224, "peak_epochs": 1, "decrease_epochs": 225, "final_epochs": 0,
    }
    training_args = {
        "log_verbosity": 4,
        "num_epochs": 500,
        "save_interval": 1,
        "keep_epochs": None,
        "time_rqmt": 168,
        "mem_rqmt": 16,
        "cpu_rqmt": 3,
    }

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "lgm80": dict(
                returnn_args=returnn_args,
                feature_args=feature_args["log_mel"],
                lr_args=lr_args,
                num_outputs=79,
                report_args={},
            ),
            # "newsegments_lgm80": dict(
            #     returnn_args=returnn_args_new,
            #     feature_args=feature_args["log_mel"],
            #     lr_args=lr_args,
            #     num_outputs=79,
            #     report_args={},
            # ),
            "gt50": dict(
                returnn_args=returnn_args,
                feature_args=feature_args["gammatone"],
                lr_args=lr_args,
                num_outputs=79,
                report_args={},
            ),
            "scf": dict(
                returnn_args=returnn_args,
                feature_args=feature_args["scf"],
                lr_args=lr_args,
                num_outputs=79,
                report_args={},
            ),
            "old_data_scf": dict(
                returnn_args=returnn_args_old,
                feature_args=feature_args["scf"],
                lr_args=lr_args,
                num_outputs=79,
                report_args={},
            ),
        },
        num_epochs=500,
        evaluation_epochs=[490, 500],
        prefix="conformer_bs10k_",
        training_args=training_args,
    )

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
        "lm_scales": [1.1],
        "prior_scales": [0.3],
        "epochs": [490, 500],
        "lookahead_options": {"lm_lookahead_scale": 1.1},
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
            "optimize-lattice": False,
            "label-pruning": 17.6,
            "label-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        },
    }

    ctc_nn_system = TransducerSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE_APPTAINER,
        rasr_binary_path=RASR_BINARY_PATH_APPTAINER,
        require_native_lstm=False,
    )
    ctc_nn_system.init_system(
        returnn_configs=returnn_configs,
        dev_keys=["dev-other"],
        corpus_data=dev_corpora,
        am_args={
            "state_tying": "monophone",
            "states_per_phone": 1,
            "tdp_transition": (0, 0, 0, 0),
            "tdp_silence": (0, 0, 0, 0),
            "phon_history_length": 0,
            "phon_future_length": 0,
        },
        scorer_info=ScorerInfo(score_kwargs={"sctk_binary_path": compile_sctk()}),
        report=Report(
            columns_start=["train_name"],
            columns_end=["lm_scale", "prior_scale", "sub", "del", "ins", "wer"],
        ),
    )
    ctc_nn_system.crp["dev-other"].acoustic_model_config.allophones.add_from_file = allophone_file
    ctc_nn_system.run_train_step(nn_args.training_args)
    ctc_nn_system.run_dev_recog_step(recog_args=recog_args, report_args=report_args_collection)

    report = Report.merge_reports([
        ctc_nn_system.report,
    ])
    report.delete_redundant_columns()
    report.delete_redundant_rows()
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report.csv"),
        values=report.get_values(),
        template=report.get_template())


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    run_baseline()
