from sisyphus import tk

import copy
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, RETURNN_ROOT

from .pipeline import training, get_best_checkpoint, get_average_checkpoint, search

from .config import get_training_config

def conformer_baseline():
    BPE_SIZE = 2000
    prefix_name = "experiments/librispeech/librispeech_100_attention/conformer_rc_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function="speed_perturbation",
        partition_epoch=3,
        epoch_wise_filters=[(1, 5, 1000)],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings
    )
    train_data_retrain = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings_retrain
    )

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None
        )


        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function

    def run_exp(ft_name, datasets, train_args, search_args=None):
        search_args = search_args if search_args is not None else train_args

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        #returnn_search_config = create_config(training_datasets=datasets, **search_args, is_recog=True)

        #from i6_core.returnn.compile import CompileTFGraphJob
        #search_compile = CompileTFGraphJob(
        #    returnn_config=returnn_search_config,
        #    returnn_python_exe=RETURNN_EXE,
        #    returnn_root=RETURNN_ROOT,
        #    search=1
        #)
        #tk.register_output(ft_name + "/attention_compile_graph", search_compile.out_graph)

        train_job = training(ft_name, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=250)

        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        #best_checkpoint = get_best_checkpoint(train_job)

        #search(ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    train_args = {
        "network_args":{
            "model_type": "conformer_aed_trial"
        },
        "debug": True
    }

    run_exp(prefix_name + "/test", datasets=train_data, train_args=train_args)
