import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .prototype_pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, TrainingDatasets

Path = tk.setup_path(__package__)


RECURSION_LIMIT_CODE = """\
import resource
import sys
try:
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
sys.setrecursionlimit(10 ** 6)
"""

def get_config(
        returnn_common_root,
        training_datasets,
        **kwargs):
    """

    :param prefix_name:
    :param TrainingDatasets training_datasets:
    :param returnn_exe:
    :param returnn_root:
    :param kwargs:
    :return:
    """

    # changing these does not change the hash
    post_config = {
        'use_tensorflow': True,
        'tf_log_memory_usage': True,
        'cleanup_old_models': True,
        'log_batch_size': True,
        'debug_print_layer_output_template': True,
    }

    wup_start_lr = 0.0003
    initial_lr = 0.0008

    learning_rates = [wup_start_lr] * 10 + list(numpy.linspace(wup_start_lr, initial_lr, num=10))

    config = {
        'gradient_clip': 0,
        'optimizer': {'class': 'Adam', 'epsilon': 1e-8},
        'accum_grad_multiple_step': 2,
        'gradient_noise': 0.0,
        'learning_rates': learning_rates,
        'min_learning_rate': 0.00001,
        'learning_rate_control': "newbob_multi_epoch",
        'learning_rate_control_relative_error_relative_lr': True,
        'learning_rate_control_min_num_epochs_per_new_lr': 3,
        'use_learning_rate_control_always': True,
        'newbob_multi_num_epochs': 3,
        'newbob_multi_update_interval': 1,
        'newbob_learning_rate_decay': 0.9,
        'batch_size': 10000,
        'max_seqs': 200,
    }

    from i6_experiments.users.rossenbach.returnn.nnet_constructor import ReturnnCommonSerializer,\
        ReturnnCommonExternData, ReturnnCommonDynamicNetwork, NonhashedCode, ReturnnCommonImport

    extern_data = [
        datastream.as_nnet_constructor_data(key) for key, datastream in training_datasets.datastreams.items()]

    config["train"] = training_datasets.train.as_returnn_opts()
    config["dev"] = training_datasets.cv.as_returnn_opts()
    #config["eval_datasets"] =  {'devtrain': training_datasets.devtrain.as_returnn_opts()}

    rc_recursionlimit = NonhashedCode(code=RECURSION_LIMIT_CODE)
    rc_extern_data = ReturnnCommonExternData(extern_data=extern_data)
    rc_model = ReturnnCommonImport(
        "i6_experiments.users.rossenbach.returnn.common_modules.asr_transformer.BLSTMDownsamplingTransformerASR")
    rc_construction_code = ReturnnCommonImport(
        "i6_experiments.users.rossenbach.returnn.common_modules.simple_asr_constructor.construct_network")

    rc_network = ReturnnCommonDynamicNetwork(
        net_func_name=rc_construction_code.get_name(),
        net_func_map={"net_module": rc_model.get_name(),
                      "audio_data": "audio_features",
                      "label_data": "bpe_labels",
                      "audio_feature_dim": "audio_features_feature",
                      "audio_time_dim": "audio_features_time",
                      "label_time_dim": "bpe_labels_time",
                      "label_dim": "bpe_labels_indices"
                     },
        net_kwargs={'weight_decay': 0.1}
    )

    serializer = ReturnnCommonSerializer(
        serializer_objects=[rc_recursionlimit,
                            rc_extern_data,
                            rc_model,
                            rc_construction_code,
                            rc_network],
        returnn_common_root=returnn_common_root,
        make_local_package_copy=True,
    )
    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[serializer],
    )
    return returnn_config


def transformer():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="0c045ec027bd32ce279a91632a8c758d1900d0dd").out_repository
    returnn_common_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common",
                                           commit="8f4378a88c6f4aeea14882b9010498558d19544f",
                                           checkout_folder_name="returnn_common").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/lstm_encdec_2022/transformer"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name)

    # Initial experiment
    exp_prefix = prefix_name + "/test_baseline"
    returnn_config = get_config(
        returnn_common_root=returnn_common_root,
        training_datasets=training_datasets
    )

    train_job = training(exp_prefix, returnn_config, num_epochs=250, returnn_exe=returnn_exe, returnn_root=returnn_root)
    tk.register_output("test_model", train_job.out_checkpoints[250].index_path)
    #search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root)


