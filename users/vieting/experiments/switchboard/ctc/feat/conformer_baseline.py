import copy
from sisyphus import tk, gs

from i6_core.meta.system import CorpusObject
from i6_core.lexicon.modification import AddEowPhonemesToLexiconJob
from i6_core.recognition import Hub5ScoreJob
from i6_experiments.common.datasets.switchboard.corpus_eval import get_hub5e00
from i6_experiments.common.setups.rasr.util import RasrDataInput
from i6_experiments.users.berger.recipe.lexicon.modification import DeleteEmptyOrthJob, MakeBlankLexiconJob
from i6_experiments.users.vieting.experiments.switchboard.hybrid.feat.experiments import run_gmm_system_from_common  # TODO: might be copied here for stability
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import (
    TransducerSystem,
    ReturnnConfigs,
    ScorerInfo,
)

from .baseline_args import get_nn_args as get_nn_args_baseline
from .data import get_corpus_data_inputs_oggzip  # TODO: might be copied here for stability
from .default_tools import RASR_BINARY_PATH, RETURNN_ROOT, RETURNN_EXE


def get_datasets():
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
    )

    returnn_datasets = {
        "train": nn_train_data_inputs["switchboard.train"].get_data_dict()["datasets"]["ogg"],
        "dev": nn_cv_data_inputs["switchboard.cv"].get_data_dict()["datasets"]["ogg"],
        "eval_datasets": {
            "devtrain": nn_devtrain_data_inputs["switchboard.devtrain"].get_data_dict()["datasets"]["ogg"],
        },
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
    dev_corpora = {"hub5e00": RasrDataInput(
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
    )}

    return returnn_datasets, rasr_loss_corpus, rasr_loss_segments, rasr_loss_lexicon, dev_corpora


def run_test_mel():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    (
        returnn_datasets, rasr_loss_corpus_path, rasr_loss_corpus_segments, rasr_loss_lexicon_path, dev_corpora
    ) = get_datasets()
    returnn_args = {
        "batch_size": 10000,
        "rasr_binary_path": RASR_BINARY_PATH,
        "rasr_loss_corpus_path": rasr_loss_corpus_path,
        "rasr_loss_corpus_segments": rasr_loss_corpus_segments,
        "rasr_loss_lexicon_path": rasr_loss_lexicon_path,
        "datasets": returnn_datasets,
    }
    feature_args = {
        "class": "LogMelNetwork",
        "wavenorm": True,
        "frame_size": 200,
        "frame_shift": 80,
        "fft_size": 256
    }

    nn_args = get_nn_args_baseline(
        nn_base_args={
            # "lgm80_conf-simon": dict(
            #     returnn_args={"conformer_type": "simon", **returnn_args},
            #     feature_args=feature_args,
            # ),
            "lgm80_conf-wei_old-lr": dict(
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args=feature_args,
            ),
            "lgm80_conf-wei": dict(
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args=feature_args,
                lr_args={
                    "peak_lr": 4e-4, "start_lr": 1.325e-05, "end_lr": 1e-5,
                    "increase_epochs": 119, "peak_epochs": 1, "decrease_epochs": 120, "final_epochs": 0,
                },
            ),
            "gt40_pe_conf-wei_old-lr": dict(
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args={
                    "class": "GammatoneNetwork", "sample_rate": 8000, "freq_max": 3800., "output_dim": 40,
                    "preemphasis": 1.0,
                },
            ),
            "scf750_conf-wei_old-lr": dict(
                returnn_args={"conformer_type": "wei", **returnn_args},
                feature_args={"class": "ScfNetwork", "size_tf": 256 // 2, "stride_tf": 10 // 2},
            ),
        },
        num_epochs=300,
        prefix="conformer_bs10k_"
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
        "prior_scales": [0.5],
        "epochs": [260],
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
    score_info.score_kwargs = {"glm": dev_corpora["hub5e00"].glm}

    ctc_nn_system = TransducerSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
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
    )
    ctc_nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_from_lexicon = False
    ctc_nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_all = True
    ctc_nn_system.crp["hub5e00"].acoustic_model_config.allophones.add_from_file = tk.Path(
        "/u/vieting/setups/swb/20230406_feat/dependencies/allophones_blank",
        hash_overwrite="SWB_ALLOPHONE_FILE_WEI_BLANK",
        cached=True
    )
    ctc_nn_system.run_train_step(nn_args.training_args)
    ctc_nn_system.run_dev_recog_step(recog_args=recog_args)
