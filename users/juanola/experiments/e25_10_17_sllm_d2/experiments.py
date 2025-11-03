import copy
from functools import partial
from typing import Any

from sisyphus import tk

from i6_core.tools.download import DownloadJob
from .configurations.training_configs import training_configs
from .default_tools import RETURNN_ROOT, MINI_RETURNN_ROOT
from .experiments_core.data.dataset_commons import DatasetSettings, build_test_dataset, TrainingDatasets
from .experiments_core.data.spm_utils import build_spm_training_datasets
from .experiments_core.model_creation.training_job_builder import create_training_job
from .experiments_core.reporting.report import create_report_job, build_base_report
from .experiments_core.tuning.evaluation import create_tune_and_evaluate_jobs
from .recognition.decoder_config import DecoderConfig


def sllm_ep(
        experiment_path: str = "experiments/librispeech/sllm/ls960/baselines",
        debug: bool = False):
    """
    Sisyphus entry point.

    Objective: prepare experiment execution:
    - Download and prepare datasets
    - Prepare model config and all needed for returnn
    - Indicate wanted outputs

    :param experiment_path: Used for alias creation
    :type debug: Used to set up config for debugging in one GPU
    """
    # INITIALIZE DATASET
    train_dataset_settings = DatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        train_partition_epoch=20,
        train_seq_ordering="laplace:.1000",
        train_additional_options={
            "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        },
    )
    sampling_alpha = 0.7  # TODO: move somewhere else?!
    vocab_size = 10_240 # 151936 # TODO: TD - this should not be hardcoded ??? which value goes here, sentence piece has a max of 56367
    training_datasets, dev_dataset_tuples, test_dataset_tuples = create_datasets_jobs(experiment_path,
                                                                                      train_dataset_settings,
                                                                                      vocab_size,
                                                                                      sampling_alpha)


    # GENERAL CONSTANTS
    train_epochs = 500 if not debug else 1 # TODO: extract to a config file?
    debug_returnn_param = True # TODO: Make it depend on big debug?

    # MODULES
    network_module = "networks.conformer_qwen_v1" # TODO:  move outside the method. Maybe in a constants class or config file...
    train_step_module = "training.train_step"
    recognition_package = "recognition"

    # NETWORK
    encoder_alias = "v1" # TODO: could be imported - extract as parameter of method
    decoder_alias = "Qwen2-0_5B"
    model_alias, network_args = get_network_args_and_alias(decoder_alias, encoder_alias)

    # MODEL TRAINING
    training_name = f"{experiment_path}/{network_module}/{model_alias}"
    train_job = create_training_job(training_name, training_datasets,
                                    network_module, network_args,
                                    train_step_module, train_epochs,
                                    debug, debug_returnn_param,
                                    returnn_root=RETURNN_ROOT)

    # MODEL EVALUATION/INFERENCE
    # Which evals to run
    if not debug:
        run_best_4 = run_best = run_test = True
        epochs_to_evaluate = [train_epochs]
    else:
        run_test = True
        run_best_4 = run_best = False
        epochs_to_evaluate = []

    # Tune-Eval
    results = create_tune_and_evaluate_jobs(
        training_name=training_name,
        train_job=train_job,

        network_module=network_module,
        net_args=network_args,
        debug=debug_returnn_param,

        train_data=training_datasets,
        decoder_config=DecoderConfig(),
        decoder_module=recognition_package,

        test_dataset_tuples=test_dataset_tuples,
        dev_dataset_tuples=dev_dataset_tuples,

        lm_scales=[0.0],
        prior_scales=[0.0],

        specific_epoch=epochs_to_evaluate,
        run_test=run_test,
        run_best=run_best,
        run_best_4=run_best_4,

        use_gpu=True,  # CPU is way too slow for AED decoding
    )

    # MODEL REPORTING
    create_report_job(results=results, exp_name=training_name)
    report = {training_name: results}
    del results
    tk.register_report(
        "reports/ls_baseline_report",
        partial(build_base_report, report),
        required=report,
        update_frequency=900
    )

    return report


def get_network_args_and_alias(decoder_alias: str, encoder_alias: str) -> tuple[
    str, dict[str, Any]]:
    # MODEL CONFIG
    # Encoder Config

    encoder_config = copy.deepcopy(training_configs[encoder_alias])
    # Decoder Config

    download_config_job = DownloadJob("https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/config.json",
                                      target_filename=f"config-{decoder_alias}.json")
    decoder_config = {"config_path": download_config_job.out_file}
    # Full Model
    model_alias = f"{encoder_alias}-{decoder_alias}"
    network_args = encoder_config | decoder_config
    return model_alias, network_args


def create_datasets_jobs(prefix_name: str, train_settings: DatasetSettings, vocab_size: int, sampling_alpha: float) -> \
        tuple[TrainingDatasets, dict[Any, Any], dict[Any, Any]]:
    """
    build the training datasets object containing train, cv, dev-train and the extern_data dict
    :param prefix_name:
    :param train_settings:
    :param vocab_size:
    :param sampling_alpha:
    :return:
    """
    train_data = build_spm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
        vocab_size=vocab_size,
        returnn_root=MINI_RETURNN_ROOT,  # to import ogg zip job from Nick
        alpha=sampling_alpha,
    )

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )
    return train_data, dev_dataset_tuples, test_dataset_tuples,
