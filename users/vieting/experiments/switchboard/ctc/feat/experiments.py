import copy
import os.path
from sisyphus import tk, gs
from sisyphus.delayed_ops import DelayedFormat

from i6_core.meta.system import CorpusObject
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.returnn.config import CodeWrapper
from i6_core.recognition import Hub5ScoreJob
from i6_core.tools import CloneGitRepositoryJob
from i6_experiments.common.datasets.switchboard.corpus_eval import get_hub5e00
from i6_experiments.common.setups.rasr.util import RasrDataInput
from i6_experiments.users.berger.recipe.lexicon.modification import DeleteEmptyOrthJob, MakeBlankLexiconJob
from i6_experiments.users.vieting.tools.report import Report

# TODO: run_gmm_system_from_common might be copied here for stability
from i6_experiments.users.vieting.experiments.switchboard.hybrid.feat.experiments import run_gmm_system_from_common
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import (
    TransducerSystem,
    ReturnnConfigs,
    ScorerInfo,
    SearchTypes,
)

from .baseline_args import get_nn_args as get_nn_args_baseline
from .data import get_corpus_data_inputs_oggzip  # TODO: might be copied here for stability
from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE, SCTK_BINARY_PATH


def get_datasets(use_multi_proc_dataset=False, **kwargs):
    gmm_system = run_gmm_system_from_common()

    # TODO: get oggzip independent of GMM system
    # noinspection PyTypeChecker
    (
        nn_train_data_inputs,
        nn_cv_data_inputs,
        nn_devtrain_data_inputs,
        nn_dev_data_inputs,
        nn_test_data_inputs,
        train_corpus_path,
        traincv_segments,
    ) = get_corpus_data_inputs_oggzip(
        gmm_system,
        partition_epoch={"train": 6, "dev": 1},
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        **kwargs,
    )

    returnn_datasets = {
        "train": nn_train_data_inputs["switchboard.train"].get_data_dict()["datasets"]["ogg"],
        "dev": nn_cv_data_inputs["switchboard.cv"].get_data_dict()["datasets"]["ogg"],
        "eval_datasets": {
            "devtrain": nn_devtrain_data_inputs["switchboard.devtrain"].get_data_dict()["datasets"]["ogg"],
        },
    }

    if use_multi_proc_dataset:
        returnn_datasets["train"] = {
            "class": "MultiProcDataset",
            "dataset": nn_train_data_inputs["switchboard.train"].get_data_dict()["datasets"]["ogg"],
            "num_workers": 2,
            "buffer_size": 5,
        }

    lexicon = gmm_system.crp["switchboard"].lexicon_config.file
    lexicon = DeleteEmptyOrthJob(lexicon).out_lexicon
    rasr_loss_lexicon = MakeBlankLexiconJob(lexicon).out_lexicon
    nonword_phones = ["[LAUGHTER]", "[NOISE]", "[VOCALIZEDNOISE]"]
    recog_lexicon = AddEowPhonemesToLexiconJob(rasr_loss_lexicon, nonword_phones=nonword_phones).out_lexicon

    rasr_loss_corpus = train_corpus_path
    rasr_loss_segments = traincv_segments

    hub5e00 = get_hub5e00()
    corpus_object = CorpusObject()
    corpus_object.corpus_file = hub5e00.bliss_corpus
    corpus_object.audio_format = nn_dev_data_inputs["hub5e00"].crp.audio_format
    corpus_object.duration = nn_dev_data_inputs["hub5e00"].crp.corpus_duration
    dev_corpora = {
        "hub5e00": RasrDataInput(
            corpus_object=corpus_object,
            lexicon={
                "filename": recog_lexicon,
                "normalize_pronunciation": False,
                "add_all": True,
                "add_from_lexicon": False,
            },
            lm={"filename": nn_dev_data_inputs["hub5e00"].crp.language_model_config.file, "type": "ARPA"},
            stm=hub5e00.stm,
            glm=hub5e00.glm,
        )
    }

    return returnn_datasets, rasr_loss_corpus, rasr_loss_segments, rasr_loss_lexicon, dev_corpora


def run_test_mel():
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
    }
    feature_args = {"class": "LogMelNetwork", "wavenorm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256}
    returnn_datasets_laplace25 = copy.deepcopy(returnn_datasets)
    returnn_datasets_laplace25["train"]["seq_ordering"] = "laplace:.25"

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            # "lgm80_conf-simon": dict(
            #     returnn_args={"conformer_type": "simon", **returnn_args},
            #     feature_args=feature_args,
            # ),
            "lgm80_conf-wei_old-lr": dict(
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args=feature_args,
                report_args={"architecture": "conf-wei", "lr": "default"},
            ),
            "lgm80_conf-wei_old-lr-4e-4": dict(
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args=feature_args,
                lr_args={"peak_lr": 4e-4},
                report_args={"architecture": "conf-wei", "lr": "default_peak_4e-4"},
            ),
            "lgm80_conf-wei": dict(  # matches original lr schedule from wei
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 1,
                    "decrease_epochs": 120,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4"},
            ),
            "lgm80_conf-wei2": dict(  # almost matches original lr schedule from wei
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4"},
            ),
            "lgm80_conf-wei-oldspecaug": dict(  # specaugment as in wei's setup
                returnn_args={"conformer_type": "wei", "specaug_old": {}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4", "specaug": "wei"},
            ),
            "lgm80_conf-wei-oldspecaug2": dict(  # specaugment as in wei's setup but double feature dim due to log Mel
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4", "specaug": "wei_adapt_80dim"},
            ),
            "lgm80_conf-wei-oldspecaug-bs3200step": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {},
                    **returnn_args,
                    "batch_size": {"data": 514635},
                },
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4", "specaug": "wei"},
            ),
            "lgm80_conf-wei-oldspecaug2-bs3200step": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {"max_feature": 8},
                    **returnn_args,
                    "batch_size": {"data": 514635},
                },
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4", "specaug": "wei_adapt_80dim"},
            ),
            "lgm80_conf-wei-oldspecaug-laplace25": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {},
                    **returnn_args,
                    "datasets": returnn_datasets_laplace25,
                },
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4", "specaug": "wei"},
            ),
            "lgm80_conf-wei-oldspecaug2-lrv1": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 2 * 4e-4,
                    "start_lr": 2 * 1.325e-05,
                    "end_lr": 2 * 1e-5,
                    "increase_epochs": 119,
                    "peak_epochs": 2,
                    "decrease_epochs": 119,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_8e-4", "specaug": "wei_adapt_80dim"},
            ),
            # "lgm80_conf-wei2-nadam": dict(  # does not work well
            #     returnn_args={
            #         "conformer_type": "wei",
            #         "extra_args": {"optimizer": None, "optimizer_epsilon": 1e-8, "nadam": True},
            #         **returnn_args,
            #     },
            #     feature_args=feature_args,
            #     lr_args={
            #         "peak_lr": 4e-4, "start_lr": 1.325e-05, "end_lr": 1e-5,
            #         "increase_epochs": 119, "peak_epochs": 2, "decrease_epochs": 119, "final_epochs": 0,
            #     },
            # ),
            # "gt40_pe_conf-wei_old-lr": dict(  # does not converge
            #     returnn_args={"conformer_type": "wei", **returnn_args},
            #     feature_args={
            #         "class": "GammatoneNetwork", "sample_rate": 8000, "freq_max": 3800., "output_dim": 40,
            #         "preemphasis": 1.0,
            #     },
            # ),
            # "acc2_scf750_conf-wei_old-lr": dict(  # does not converge
            #     returnn_args={
            #     "conformer_type": "wei",
            #     **returnn_args,
            #     "batch_size": 5000,
            #     "extra_args": {"accum_grad_multiple_step": 2},
            #     },
            #     feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2},
            # ),
        },
        num_epochs=300,
        prefix="conformer_bs10k_",
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
        "lm_scales": [0.7],
        "prior_scales": [0.3, 0.5],
        "epochs": [300],
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
    }
    score_info = ScorerInfo()
    score_info.ref_file = dev_corpora["hub5e00"].stm
    score_info.job_type = Hub5ScoreJob
    score_info.score_kwargs = {"glm": dev_corpora["hub5e00"].glm, "sctk_binary_path": SCTK_BINARY_PATH}

    ctc_nn_system = TransducerSystem(
        returnn_root=RETURNN_ROOT,
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
    ctc_nn_system.run_recogs_for_corpora(  # test for single model with larger prior_scales
        ["hub5e00"],
        "conformer_bs10k_lgm80_conf-wei_old-lr-4e-4",
        search_type=SearchTypes.GenericSeq2SeqSearchJob,
        report_args=report_args_collection,
        **{**recog_args, "prior_scales": [0.4, 0.6, 0.7, 0.9]},
    )
    ctc_nn_system.run_recogs_for_corpora(  # test for single model with larger prior_scales
        ["hub5e00"],
        "conformer_bs10k_lgm80_conf-wei_old-lr",
        search_type=SearchTypes.GenericSeq2SeqSearchJob,
        report_args=report_args_collection,
        **{**recog_args, "prior_scales": [0.5], "epochs": [260]},
    )

    # test blank penalty as we have more deletions than insertions
    ctc_nn_system_blank_penalty = copy.deepcopy(ctc_nn_system)
    exp_name = "conformer_bs10k_lgm80_conf-wei_old-lr-4e-4"
    recog_config = ctc_nn_system_blank_penalty.returnn_configs[exp_name].recog_configs.pop("recog")
    for blank_penalty in [0.2, 0.3, 0.5, 0.8]:
        config = copy.deepcopy(recog_config)
        blank_index = 0
        num_outputs = 88
        config.config["network"]["output_blank_penalty"] = {
            "class": "eval",
            "from": "output",
            "is_output_layer": True,
            "eval": f"source(0) - tf.expand_dims("
            f"tf.one_hot([{blank_index}], {num_outputs}, on_value={blank_penalty}, dtype=tf.float32), axis=0)",
        }
        ctc_nn_system_blank_penalty.returnn_configs[exp_name].recog_configs[
            f"recog_blank-penalty-{blank_penalty}"
        ] = config
    ctc_nn_system_blank_penalty.run_recogs_for_corpora(
        ["hub5e00"],
        exp_name,
        search_type=SearchTypes.GenericSeq2SeqSearchJob,
        report_args=report_args_collection,
        extra_name=f"_blank-pen-{blank_penalty}",
        tf_flow_args={"output_layer_name": "output_blank_penalty"},
        **{**recog_args, "prior_scales": [0.5]},
    )

    # same lm as in wei's setup, results indicate that this is not better (if any, slightly worse)
    ctc_nn_system_wei_lm = copy.deepcopy(ctc_nn_system)
    for train_name in list(ctc_nn_system_wei_lm.returnn_configs.keys()):
        if train_name not in ["conformer_bs10k_lgm80_conf-wei2", "conformer_bs10k_lgm80_conf-wei_old-lr-4e-4"]:
            # only use some trainings
            ctc_nn_system_wei_lm.returnn_configs.pop(train_name)
    report_args_wei_lm = copy.deepcopy(report_args_collection)
    for name in report_args_wei_lm:
        report_args_wei_lm[name]["lm"] = "wei"
    wei_lm = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/zoltan_4gram.gz",
        hash_overwrite="ZOLTAN_SWB_LM_4GRAM",
        cached=True,
    )
    ctc_nn_system_wei_lm.corpus_data["hub5e00"].lm["filename"] = wei_lm
    ctc_nn_system_wei_lm.crp["hub5e00"].language_model_config.file = wei_lm
    ctc_nn_system_wei_lm.run_dev_recog_step(recog_args=recog_args, extra_name="_lm-wei", report_args=report_args_wei_lm)

    # same lexicon as in wei's setup, results indicate that this is not better (less del, but overall slightly worse)
    ctc_nn_system_wei_lex = copy.deepcopy(ctc_nn_system)
    for train_name in list(ctc_nn_system_wei_lex.returnn_configs.keys()):
        if train_name not in ["conformer_bs10k_lgm80_conf-wei2", "conformer_bs10k_lgm80_conf-wei_old-lr-4e-4"]:
            # only use some trainings
            ctc_nn_system_wei_lex.returnn_configs.pop(train_name)
    report_args_wei_lex = copy.deepcopy(report_args_collection)
    for name in report_args_wei_lex:
        report_args_wei_lex[name]["lex"] = "wei"
    wei_lex = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/lexicon_wei_blank.xml",
        hash_overwrite="WEI_SWB_LEX",
        cached=True,
    )
    ctc_nn_system_wei_lex.corpus_data["hub5e00"].lexicon["filename"] = wei_lex
    ctc_nn_system_wei_lex.run_dev_recog_step(
        recog_args=recog_args, extra_name="_lex-wei", report_args=report_args_wei_lex
    )

    # longer training to compensate for fewer steps per epoch
    feature_args_wave_norm = {
        "class": "LogMelNetwork",
        "wave_norm": True,
        "frame_size": 200,
        "frame_shift": 80,
        "fft_size": 256,
    }
    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "lgm80_conf-wei-oldspecaug-e450v1": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={"architecture": "conf-wei", "lr": "wei_peak_4e-4_e450_cycle360", "specaug": "wei"},
            ),
            "lgm80_conf-wei-oldspecaug2-e450v1": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "specaug": "wei_adapt_80dim",
                },
            ),
            "lgm80_conf-wei-oldspecaug2-e450v2": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 160,
                    "decrease_epochs": 160,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_320cycle",
                    "specaug": "wei_adapt_80dim",
                },
            ),
            "lgm80_conf-wei-oldspecaug2-e450v3": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 200,
                    "decrease_epochs": 200,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_400_cycle",
                    "specaug": "wei_adapt_80dim",
                },
            ),
            "lgm80_conf-wei-oldspecaug2-e450v1-wavenorm": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args_wave_norm,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                },
            ),
        },
        num_epochs=450,
        prefix="conformer_bs10k_",
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

    recog_args_e450 = copy.deepcopy(recog_args)
    recog_args_e450["epochs"] = [300, 400, 450, "best"]
    ctc_nn_system_e450 = copy.deepcopy(ctc_nn_system)
    ctc_nn_system_e450.returnn_configs = returnn_configs
    ctc_nn_system_e450.run_train_step(nn_args.training_args)
    ctc_nn_system_e450.run_dev_recog_step(recog_args=recog_args_e450, report_args=report_args_collection)

    report = Report.merge_reports(
        [
            ctc_nn_system.report,
            ctc_nn_system_e450.report,
            ctc_nn_system_blank_penalty.report,
            ctc_nn_system_wei_lm.report,
            ctc_nn_system_wei_lex.report,
        ]
    )
    report.delete_redundant_columns()
    report.delete_redundant_rows()
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report_test_mel.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )


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
    return report


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
    }
    feature_args = {"class": "LogMelNetwork", "wave_norm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256}

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "lgm80_baseline": dict(
                returnn_args={"conformer_type": "wei", "specaug_old": {"max_feature": 8}, **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                },
            ),
        },
        num_epochs=450,
        prefix="conformer_bs10k_",
    )
    report = run_nn_args(nn_args, report_args_collection, dev_corpora)
    return report


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

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "scf_baseline": dict(
                returnn_args=returnn_args,
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                },
            ),
        },
        num_epochs=450,
        evaluation_epochs=[350, 400, 450],
        prefix="conformer_bs2x5k_",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        returnn_root=returnn_root,
        recog_args={"epochs": [350, 400, 450, "best"]},
    )
    return report


def run_scf_audio_perturbation_gridsearch():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

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
        "specaug_old": {"max_feature": 15},
        "audio_perturbation": True,
        "use_multi_proc_dataset": True,
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

    perturbation_args = [
        {'speeds': {'prob': 0.6, 'minimum': 0.8, 'maximum': 1.2}},
        {'speeds': {'prob': 0.6, 'minimum': 0.7, 'maximum': 1.3}},
        {'speeds': {'prob': 0.5, 'minimum': 0.9, 'maximum': 1.1}},
        {'speeds': {'prob': 0.5, 'minimum': 0.8, 'maximum': 1.2}},
        {'speeds': {'prob': 0.5, 'minimum': 0.7, 'maximum': 1.3}},
        {'speeds': {'prob': 0.4, 'minimum': 0.9, 'maximum': 1.1}},
        {'speeds': {'prob': 0.4, 'minimum': 0.8, 'maximum': 1.2}},
        {'speeds': {'prob': 0.4, 'minimum': 0.7, 'maximum': 1.3}},
        {'tempos': {'prob': 0.4, 'minimum': 0.9, 'maximum': 1.1}},
        {'tempos': {'prob': 0.4, 'minimum': 0.8, 'maximum': 1.2}},
        {'tempos': {'prob': 0.4, 'minimum': 0.7, 'maximum': 1.3}},
        {'tempos': {'prob': 0.5, 'minimum': 0.9, 'maximum': 1.1}},
        {'tempos': {'prob': 0.5, 'minimum': 0.8, 'maximum': 1.2}},
        {'tempos': {'prob': 0.5, 'minimum': 0.7, 'maximum': 1.3}},
        {'tempos': {'prob': 0.6, 'minimum': 0.9, 'maximum': 1.1}},
        {'tempos': {'prob': 0.6, 'minimum': 0.8, 'maximum': 1.2}},
        {'tempos': {'prob': 0.6, 'minimum': 0.7, 'maximum': 1.3}},
        {'preemphases': {'prob': 0.9, 'minimum': 0.9, 'maximum': 1.0}},
        {'preemphases': {'prob': 0.9, 'minimum': 0.8, 'maximum': 1.0}},
        {'preemphases': {'prob': 0.8, 'minimum': 0.9, 'maximum': 1.0}},
        {'preemphases': {'prob': 0.8, 'minimum': 0.8, 'maximum': 1.0}},
        {'codecs': {'encoding': 'ULAW', 'prob': 0.4}},
        {'codecs': {'encoding': 'ULAW', 'prob': 0.6}},
        {'non_linearities': {'prob': 0.4,  'minimum': 0.1, 'maximum': 0.3}},
        {'non_linearities': {'prob': 0.4,  'minimum': 0.2, 'maximum': 0.4}},
        {'non_linearities': {'prob': 0.6,  'minimum': 0.1, 'maximum': 0.3}},
        {'non_linearities': {'prob': 0.6,  'minimum': 0.2, 'maximum': 0.4}},
    ]

    nn_base_args = {}

    for args in perturbation_args:
        arg_key = list(args.keys())[0]
        arg_values = list(args.values())[0]

        # Construct key_suffix and report_values
        if 'minimum' in arg_values and 'maximum' in arg_values:
            key_suffix = f"{arg_key}_{arg_values['prob']}_{arg_values['minimum']}_{arg_values['maximum']}_"
            report_values = f"{arg_key}: '{arg_values['prob']}_{arg_values['minimum']}_{arg_values['maximum']}'"
        else:
            key_suffix = f"{arg_key}_{arg_values['prob']}_"
            report_values = f"{arg_key}: '{arg_values['prob']}'"

        # Construct the key and report_args
        key = f"scf_bs2x5k_perturb_{key_suffix}"
        report_args = {key: report_values}
        nn_base_args[key] = dict(
            returnn_args={
                "extra_args": {
                    "audio_perturb_args": args,
                    "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    "conv_pad_seq_len_to_power": 1.5,
                    "watch_memory": True,
                    "accum_grad_multiple_step": 2,
                },
                **returnn_args,
            },
            feature_args=feature_args,
            lr_args=lr_args,
            report_args=report_args,
        )    

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args=nn_base_args,
        num_epochs=450,
        evaluation_epochs=[350, 400, 450],
        prefix="conformer_",
    )

    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="c4d36d06f6465e82a50d400d114259e07b8b0709",
    ).out_repository
    returnn_root.hash_overwrite = "returnn_conv_padding"
    report = run_nn_args(
        nn_args,
        report_args_collection,
        dev_corpora,
        returnn_root=returnn_root,
        recog_args={"epochs": [350, 400, 450, "best"]},
    )
    return report


def run_mel_audio_perturbation():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    (
        returnn_datasets,
        rasr_loss_corpus_path,
        rasr_loss_corpus_segments,
        rasr_loss_lexicon_path,
        dev_corpora,
    ) = get_datasets(pre_process=CodeWrapper("audio_perturb_runner.run"))
    returnn_args = {
        "batch_size": 10000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
    }
    feature_args = {"class": "LogMelNetwork", "wave_norm": True, "frame_size": 200, "frame_shift": 80, "fft_size": 256}

    nn_args, report_args_collection = get_nn_args_baseline(
        nn_base_args={
            "lgm80_conf-wei-oldspecaug-audio_perturbation": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {"max_feature": 8},
                    "audio_perturbation": True,
                    "extra_args": {
                        "audio_perturb_args": {
                            "speed": {"prob": 0.6, "minimum": 0.88, "maximum": 1.12},
                            "tempo": {"prob": 0.6, "minimum": 0.83, "maximum": 1.17},
                        },
                        "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    },
                    **returnn_args,
                },
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "speed": "0.6_0.88_1.12",
                    "tempo": "0.6_0.83_1.17",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                },
            ),
            "lgm80_conf-wei-oldspecaug-audio_perturbation_v1": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {"max_feature": 8},
                    "audio_perturbation": True,
                    "extra_args": {
                        "audio_perturb_args": {  # v1
                            "speed": {"prob": 0.6, "minimum": 0.88, "maximum": 1.12},
                            "tempo": {"prob": 0.6, "minimum": 0.83, "maximum": 1.17},
                            "preemphasis": {"prob": 0.9, "minimum": 0.9, "maximum": 1.0},
                        },
                        "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    },
                    **returnn_args,
                },
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "speed": "0.6_0.88_1.12",
                    "tempo": "0.6_0.83_1.17",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                    "preemphasis": "0.9_0.9_1.0",
                },
            ),
            "lgm80_conf-wei-oldspecaug-audio_perturbation_v2": dict(
                returnn_args={
                    "conformer_type": "wei",
                    "specaug_old": {"max_feature": 8},
                    "audio_perturbation": True,
                    "extra_args": {
                        "audio_perturb_args": {  # v2
                            "speed": {"prob": 0.6, "minimum": 0.88, "maximum": 1.12},
                            "tempo": {"prob": 0.6, "minimum": 0.83, "maximum": 1.17},
                            "preemphasis": {"prob": 0.9, "minimum": 0.9, "maximum": 1.0},
                            "codecs": [{"encoding": "ULAW", "prob": 0.4}],
                        },
                        "audio_perturb_runner": CodeWrapper("WaveformPerturbation(**audio_perturb_args)"),
                    },
                    **returnn_args,
                },
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4,
                    "start_lr": 1.325e-05,
                    "end_lr": 1e-5,
                    "increase_epochs": 180,
                    "decrease_epochs": 180,
                    "final_epochs": 0,
                },
                report_args={
                    "architecture": "conf-wei",
                    "lr": "wei_peak_4e-4_e450_cycle360",
                    "speed": "0.6_0.88_1.12",
                    "tempo": "0.6_0.83_1.17",
                    "specaug": "wei_adapt_80dim",
                    "wave_norm": "True",
                    "preemphasis": "0.9_0.9_1.0",
                    "codec": "wav_ulaw_0.4",
                },
            ),
        },
        num_epochs=450,
        prefix="conformer_bs10k_",
    )
    run_nn_args(nn_args, report_args_collection, dev_corpora, "report_mel_audio_perturbation.csv")


def py():
    """
    called if the file is passed to sis manager, used to run all experiments (replacement for main)
    """
    report_mel = run_mel_baseline()
    report_scf = run_scf_baseline()

    report_base = Report(
        columns_start=["train_name", "wave_norm", "specaug", "lr", "batch_size"],
        columns_end=["epoch", "recog_name", "lm", "optlm", "lm_scale", "prior_scale"],
    )
    report = Report.merge_reports(
        [
            report_base,
            report_mel,
            report_scf,
        ]
    )
    tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, "report.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
