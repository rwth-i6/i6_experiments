import copy
from sisyphus import tk, gs

from .baseline_args import get_nn_args as get_nn_args_baseline

from i6_experiments.users.berger.recipe.lexicon.modification import DeleteEmptyOrthJob, MakeBlankLexiconJob
from i6_experiments.users.vieting.experiments.switchboard.hybrid.feat.experiments import run_gmm_system_from_common  # TODO: might be copied here for stability
from i6_experiments.users.vieting.experiments.switchboard.ctc.feat.transducer_system_v2 import TransducerSystem, ReturnnConfigs

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
    lexicon = MakeBlankLexiconJob(lexicon).out_lexicon

    rasr_loss_corpus = train_corpus_path
    rasr_loss_segments = traincv_segments
    rasr_loss_lexicon = lexicon

    return returnn_datasets, rasr_loss_corpus, rasr_loss_segments, rasr_loss_lexicon


def run_test_mel():
    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/switchboard/ctc/feat/"

    returnn_datasets, rasr_loss_corpus_path, rasr_loss_corpus_segments, rasr_loss_lexicon_path = get_datasets()
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
        },
        num_epochs=300,
        prefix="conformer_bs10k_"
    )

    returnn_configs = {}
    for exp in nn_args.returnn_training_configs:
        returnn_configs[exp] = ReturnnConfigs(
            train_config=nn_args.returnn_training_configs[exp],
            recog_configs={"recog": nn_args.returnn_recognition_configs[exp]},
        )

    ctc_nn_system = TransducerSystem(
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE,
        rasr_binary_path=RASR_BINARY_PATH,
    )
    ctc_nn_system.init_system(returnn_configs=returnn_configs)
    for exp in returnn_configs:
        ctc_nn_system.returnn_training(exp, **nn_args.training_args)