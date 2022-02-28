import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint


from .config import create_config, BLSTMNetworkOptions

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
    network_options = BLSTMNetworkOptions()
    returnn_config = create_config(training_datasets=training_datasets, network_options=network_options)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root_search)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)





base_contrastive_loss_opts = {
    'softmax_temp': 10.0,
    'num_neg_samples': 100,
    'masked_input_layer_name': 'lstm2_pool',
    'next_layer_names': ['lstm3_fw', 'lstm3_bw'],
    'masked_input_dim': 2048,
    'upsample': False
}


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
    network_options = BLSTMNetworkOptions()
    retrain_opts = {
        'model': "/work/asr4/rossenbach/sisyphus_work_folders/librispeech_tts_work/returnn/training/RETURNNTrainingFromFile.dZJ0CQR0dfXS/output/models/epoch.080"
    }
    returnn_config = create_config(training_datasets=training_datasets, network_options=network_options, retrain_opts=retrain_opts)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root, num_epochs=170)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[170], test_dataset_tuples, returnn_exe, returnn_root_search)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

    # Initial experiment
    exp_prefix = prefix_name + "/test_baseline_from_old_lr3e-4"
    network_options = BLSTMNetworkOptions()
    retrain_opts = {
        'model': "/work/asr4/rossenbach/sisyphus_work_folders/librispeech_tts_work/returnn/training/RETURNNTrainingFromFile.dZJ0CQR0dfXS/output/models/epoch.080"
    }
    returnn_config = create_config(training_datasets=training_datasets, network_options=network_options, retrain_opts=retrain_opts)
    returnn_config.config['learning_rates'] = [0.0003]
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root, num_epochs=170)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[170], test_dataset_tuples, returnn_exe, returnn_root_search)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)



    # contrastive test
    # Initial experiment
    for variant in [1,2,3,4]:
        exp_prefix = prefix_name + "/test_contrastive_variant_%i" % variant
        network_options = BLSTMNetworkOptions()
        retrain_opts = {
            'model': "/work/asr4/rossenbach/sisyphus_work_folders/librispeech_tts_work/returnn/training/RETURNNTrainingFromFile.dZJ0CQR0dfXS/output/models/epoch.080"
        }

        contrastive_loss_opts = copy.deepcopy(base_contrastive_loss_opts)

        contrastive_loss_opts['variant'] = variant
        contrastive_loss_opts['loss_scale'] = 0.3
        contrastive_loss_opts['project_dim'] = 256
        contrastive_loss_opts['num_neg_samples'] = 10
        contrastive_loss_opts['masking_method'] = 'specaug'

        contrastive_loss_opts['masked_input_layer_name'] = 'source'
        contrastive_loss_opts['next_layer_names'] = 'source0'

        if variant != 4:
            contrastive_loss_opts['masked_input_dim'] = 40  # 40-dim MFCC
        else:
            # needed only for variant 4
            contrastive_loss_opts['layer_after_downsample'] = 'lstm1_pool'
            contrastive_loss_opts['next_layers_after_downsample'] = ['lstm2_fw', 'lstm2_bw']
            contrastive_loss_opts['masked_input_dim'] = 2048

        contrastive_loss_opts['l2'] = 1e-4

        returnn_config = create_config(training_datasets=training_datasets, network_options=network_options, retrain_opts=retrain_opts, contrastive_loss_opts=contrastive_loss_opts, behavior_version=3)
        train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root, num_epochs=170)
        search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[170], test_dataset_tuples, returnn_exe, returnn_root_search)
        #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

    for variant in [1, 3]:
        exp_prefix = prefix_name + "/test_contrastive_variant_%i_lr3e-4_neg15_scale0.1" % variant
        network_options = BLSTMNetworkOptions()
        retrain_opts = {
            'model': "/work/asr4/rossenbach/sisyphus_work_folders/librispeech_tts_work/returnn/training/RETURNNTrainingFromFile.dZJ0CQR0dfXS/output/models/epoch.080"
        }

        contrastive_loss_opts = copy.deepcopy(base_contrastive_loss_opts)

        contrastive_loss_opts['variant'] = variant
        contrastive_loss_opts['loss_scale'] = 0.1
        contrastive_loss_opts['project_dim'] = 256
        contrastive_loss_opts['num_neg_samples'] = 15
        contrastive_loss_opts['masking_method'] = 'specaug'

        contrastive_loss_opts['masked_input_layer_name'] = 'source'
        contrastive_loss_opts['next_layer_names'] = 'source0'

        if variant != 4:
            contrastive_loss_opts['masked_input_dim'] = 40  # 40-dim MFCC
        else:
            # needed only for variant 4
            contrastive_loss_opts['layer_after_downsample'] = 'lstm1_pool'
            contrastive_loss_opts['next_layers_after_downsample'] = ['lstm2_fw', 'lstm2_bw']
            contrastive_loss_opts['masked_input_dim'] = 2048

        contrastive_loss_opts['l2'] = 1e-4

        returnn_config = create_config(training_datasets=training_datasets, network_options=network_options, retrain_opts=retrain_opts, contrastive_loss_opts=contrastive_loss_opts, behavior_version=3)
        returnn_config.config['learning_rates'] = [0.0003]
        train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root, num_epochs=170)
        search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[170], test_dataset_tuples, returnn_exe, returnn_root_search)
        #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

