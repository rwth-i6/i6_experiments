import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint


from .config import create_config, NetworkOptions

def baseline():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    returnn_root_search = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/contrastive_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root, prefix_name, bpe_size=2000)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root, output_path=prefix_name)

    # Initial experiment
    exp_prefix = prefix_name + "/test_baseline"
    network_options = NetworkOptions()
    returnn_config = create_config(training_datasets=training_datasets, network_options=network_options)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root_search)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)


def continue_from_old():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    returnn_root_search = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/contrastive_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root, prefix_name, bpe_size=2000, use_curicculum=False)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root, output_path=prefix_name)

    # Initial experiment
    exp_prefix = prefix_name + "/test_baseline_from_old"
    network_options = NetworkOptions()
    retrain_opts = {
        'model': "/work/asr4/rossenbach/sisyphus_work_folders/librispeech_tts_work/returnn/training/RETURNNTrainingFromFile.dZJ0CQR0dfXS/output/models/epoch.080"
    }
    returnn_config = create_config(training_datasets=training_datasets, network_options=network_options, retrain_opts=retrain_opts)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root, num_epochs=170)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[170], test_dataset_tuples, returnn_exe, returnn_root_search)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

