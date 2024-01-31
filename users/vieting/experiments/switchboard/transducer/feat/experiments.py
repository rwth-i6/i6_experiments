import copy
import os.path
from sisyphus import tk, gs

from i6_core import corpus
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.recognition import Hub5ScoreJob
from i6_core.returnn import RasrFeatureDumpHDFJob, RasrAlignmentDumpHDFJob, BlissToOggZipJob
from i6_core.text.processing import ConcatenateJob
from i6_experiments.users.vieting.tools.report import Report

from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.experiments import get_datasets as get_datasets_ctc
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import (
    TransducerSystem,
    ReturnnConfigs,
    ScorerInfo,
)
from .baseline_args import get_nn_args as get_nn_args_baseline
from .helpers.lr.oclr import dynamic_learning_rate
from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE, SCTK_BINARY_PATH


def get_datasets_transducer(waveform=True):
    returnn_datasets, rasr_loss_corpus, rasr_loss_segments, rasr_loss_lexicon, dev_corpora = get_datasets_ctc()
    alignment_caches_train = [tk.Path(
        "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/mm/alignment/AlignmentJob.fWmd1ZVWfcFA/output/"
        f"alignment.cache.{idx}",
        hash_overwrite=f"wei_ctc_blstm_ss4_alignment_train_{idx}"
    ) for idx in range(1, 101)]
    alignment_caches_dev = [tk.Path(
        "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/mm/alignment/AlignmentJob.ETS2qXk7kdOY/output/"
        f"alignment.cache.{idx}",
        hash_overwrite=f"wei_ctc_blstm_ss4_alignment_dev_{idx}"
    ) for idx in range(1, 11)]
    allophone_file = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/allophones",
        hash_overwrite="SWB_ALLOPHONE_FILE_WEI"
    )
    state_tying_file = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/state-tying",
        hash_overwrite="SWB_STATE_TYING_FILE_MONO_EOW_NOCTX_WEI"
    )
    targets = RasrAlignmentDumpHDFJob(
        alignment_caches=alignment_caches_train + alignment_caches_dev,
        allophone_file=allophone_file,
        state_tying_file=state_tying_file,
        sparse=True,
        returnn_root=RETURNN_ROOT,
    )
    feature_caches_train = [tk.Path(
        "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/features/extraction/"
        f"FeatureExtraction.Gammatone.OKQT9hEV3Zgd/output/gt.cache.{idx}",
        hash_overwrite=f"wei_ls960_gammatone_train_{idx}"
    ) for idx in range(1, 101)]
    feature_cache_bundle_train = tk.Path(
        "/u/zhou/asr-exps/swb1/2021-12-09_phoneme-transducer/work/features/extraction/"
        "FeatureExtraction.Gammatone.OKQT9hEV3Zgd/output/gt.cache.bundle",
        hash_overwrite="wei_ls960_gammatone_train_bundle",
        cached=False,
    )
    feature_caches_dev = [tk.Path(
        "/u/zhou/asr-exps/swb1/2020-07-27_neural_transducer/work/features/extraction/"
        f"FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.{idx}",
        hash_overwrite=f"wei_ls960_gammatone_dev_{idx}"
    ) for idx in range(1, 11)]
    feature_cache_bundle_dev = tk.Path(
        "/u/zhou/asr-exps/swb1/2020-07-27_neural_transducer/work/features/extraction/"
        "FeatureExtraction.Gammatone.pp9W8m2Z8mHU/output/gt.cache.bundle",
        hash_overwrite="wei_ls960_gammatone_dev_bundle",
        cached=False,
    )

    returnn_datasets["train"]["segment_file"] = corpus.FilterSegmentsByListJob(
        {1: returnn_datasets["train"]["segment_file"]},
        targets.out_excluded_segments,
    ).out_single_segment_files[1]
    returnn_datasets["dev"]["segment_file"] = corpus.FilterSegmentsByListJob(
        {1: returnn_datasets["dev"]["segment_file"]},
        targets.out_excluded_segments,
    ).out_single_segment_files[1]
    segment_files = {
        "train": returnn_datasets["train"]["segment_file"],
        "dev": returnn_datasets["dev"]["segment_file"],
        "devtrain": returnn_datasets["eval_datasets"]["devtrain"]["segment_file"],
        "dev.wei": tk.Path(
            "/u/vieting/setups/swb/20230406_feat/dependencies/segments.wei.dev",
            hash_overwrite="swb_segments_dev_wei",
        ),
    }

    def _add_targets_to_dataset(dataset):
        ogg_zip_job = dataset["path"][0].creator
        feature_bundle = ConcatenateJob(
            [feature_cache_bundle_train, feature_cache_bundle_dev],
            zip_out=False,
            out_name="gt.cache.bundle",
        ).out
        synced_ogg_zip_job = BlissToOggZipJob(
            bliss_corpus=ogg_zip_job.bliss_corpus,
            segments=ogg_zip_job.segments,
            rasr_cache=feature_bundle,
            raw_sample_rate=ogg_zip_job.raw_sample_rate,
            feat_sample_rate=ogg_zip_job.feat_sample_rate,
            returnn_python_exe=ogg_zip_job.returnn_python_exe,
            returnn_root=ogg_zip_job.returnn_root,
        )
        synced_ogg_zip_job.rqmt = {"time": 8.0, "cpu": 2}
        dataset["path"] = [synced_ogg_zip_job.out_ogg_zip]
        dataset = {
            "class": "MetaDataset",
            "data_map": {"classes": ("alignment", "data"), "data": ("ogg", "data")},
            "datasets": {
                "ogg": dataset,
                "alignment": {
                    "class": "HDFDataset",
                    "files": targets.out_hdf_files,
                    "use_cache_manager": True,
                },
            },
            "seq_order_control_dataset": "ogg",
            "partition_epoch": dataset.get("partition_epoch", 1),
            "context_window": {"classes": 1, "data": 121},
        }
        return dataset

    if waveform:
        returnn_datasets["train"] = _add_targets_to_dataset(returnn_datasets["train"])
        returnn_datasets["dev"] = _add_targets_to_dataset(returnn_datasets["dev"])
        returnn_datasets["eval_datasets"]["devtrain"] = _add_targets_to_dataset(
            returnn_datasets["eval_datasets"]["devtrain"])
    else:
        features = RasrFeatureDumpHDFJob(feature_caches_train + feature_caches_dev, returnn_root=RETURNN_ROOT)
        returnn_datasets = {
            "train": {
                "class": "MetaDataset",
                "data_map": {"classes": ("alignment", "data"), "data": ("features", "data")},
                "datasets": {
                    "features": {
                        "class": "HDFDataset",
                        "files": features.out_hdf_files,
                        "use_cache_manager": True,
                    },
                    "alignment": {
                        "class": "HDFDataset",
                        "files": targets.out_hdf_files,
                        "use_cache_manager": True,
                        "seq_ordering": "laplace:.384",
                        "seq_list_filter_file": segment_files["train"],
                        "partition_epoch": 6,
                    },
                },
                "seq_order_control_dataset": "alignment",
                "partition_epoch": 6,
            },
            "dev": {
                "class": "MetaDataset",
                "data_map": {"classes": ("alignment", "data"), "data": ("features", "data")},
                "datasets": {
                    "features": {
                        "class": "HDFDataset",
                        "files": features.out_hdf_files,
                        "use_cache_manager": True,
                    },
                    "alignment": {
                        "class": "HDFDataset",
                        "files": targets.out_hdf_files,
                        "use_cache_manager": True,
                        "seq_ordering": "sorted_reverse",
                        "seq_list_filter_file": segment_files["dev"],
                    },
                },
                "seq_order_control_dataset": "alignment",
            },
        }
        returnn_datasets["eval_datasets"] = {
            "devtrain": copy.deepcopy(returnn_datasets["dev"]),
            "dev.wei": copy.deepcopy(returnn_datasets["dev"]),
        }
        returnn_datasets["eval_datasets"]["devtrain"]["datasets"]["alignment"]["seq_list_filter_file"] = segment_files["devtrain"]
        returnn_datasets["eval_datasets"]["dev.wei"]["datasets"]["alignment"]["seq_list_filter_file"] = segment_files["dev.wei"]

    # retrieve silence lexicon
    nonword_phones = ["[LAUGHTER]", "[NOISE]", "[VOCALIZEDNOISE]"]
    recog_lexicon = AddEowPhonemesToLexiconJob(
        rasr_loss_lexicon.creator.bliss_lexicon, nonword_phones=nonword_phones
    ).out_lexicon
    dev_corpora["hub5e00"].lexicon["filename"] = recog_lexicon

    return returnn_datasets, rasr_loss_corpus, rasr_loss_segments, rasr_loss_lexicon, dev_corpora


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
            "prior_scales": [0.5],
            "epochs": [290],  # [130, 140, 150, "best"],
            "lookahead_options": {"lm_lookahead_scale": 0.55},
            "label_scorer_args": {
                "use_prior": True,
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

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets_transducer(waveform=False)
    returnn_args = {
        "batch_size": 15000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
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
    report = run_nn_args(nn_args, report_args_collection, dev_corpora, recog_args=recog_args)
    return report


def run_mel_baseline():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/transducer/feat/"

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets_transducer()
    returnn_args = {
        "batch_size": 15000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
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
            "lgm80_baseline": dict(
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
        prefix="viterbi_bs10k_",
    )
    report = run_nn_args(nn_args, report_args_collection, dev_corpora)
    return report


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    report_rasr_gt = run_rasr_gt_baseline()
    report_mel = run_mel_baseline()

    # report_base = Report(
    #     columns_start=["train_name", "wave_norm", "specaug", "lr", "batch_size"],
    #     columns_end=["epoch", "recog_name", "lm", "optlm", "lm_scale", "prior_scale"],
    # )
    # report = Report.merge_reports([
    #     report_base,
    #     report_mel,
    # ])
    # tk.register_report(
    #     os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report.csv"),
    #     values=report.get_values(),
    #     template=report.get_template(),
    # )
