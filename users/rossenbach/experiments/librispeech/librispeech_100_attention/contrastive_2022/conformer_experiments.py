import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint


from .conformer_config import create_config, EncoderOptions, DecoderOptions

def conformer_baseline():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                commit="50c0cb8ef6d0c3bf26dd81fb4cb9014a6fa10937").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/contrastive_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name)

    # Initial experiment
    exp_prefix = prefix_name + "/conformer_baseline"
    encoder_options = EncoderOptions()
    decoder_options = DecoderOptions()

    # TODO: use behavior version = 3 and fix recursion limit if you ever run this again!!!!!!!!!!!

    returnn_config = create_config(training_datasets=training_datasets, encoder_options=encoder_options, decoder_options=decoder_options, behavior_version=1)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)
