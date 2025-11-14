import copy
from functools import partial
from typing import Any, Dict, Tuple

from sisyphus import tk

from returnn_common.datasets import Dataset
from .configurations.qwen2_decoder_config_job import Qwen2DecoderConfigJob
from .configurations.training_configs import llm_training_configs
from .constants import NETWORK_MODULE, TRAIN_STEP_MODULE, RECOGNITION_PACKAGE
from .default_tools import RETURNN_ROOT, MINI_RETURNN_ROOT
from .experiments_core.data.dataset_commons import build_lm_test_dataset
from .experiments_core.data.spm_utils import build_spm_lm_training_datasets
from .experiments_core.model_creation.training_job_builder import create_training_job
from .experiments_core.reporting.report import create_report_job, build_base_report
from .experiments_core.tuning.evaluation import create_tune_and_evaluate_jobs
from .recognition.decoder_config import DecoderConfig
from ...data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from ...data.training_datasets import TrainingDatasets
from ...utils.returnn.checkpoint_helper import default_returnn_keep_epochs


def llm_ep(
        experiment_path: str = "experiments/librispeech/llm/ls960/baselines",
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

    # GENERAL CONSTANTS

    # Returnn
    debug_returnn_param = True  # TODO: Make it depend on big debug?

    # Training
    epochs: int = 100
    partition_epoch_factor: int = 20
    num_gpus: int = 1 # Should be 1 for 48gb in i6 cluster
    partition_epochs: int = int(epochs * partition_epoch_factor / num_gpus)
    TRAINING_GPU_MEMORY = 48

    if debug:
        partition_epochs = 1
        num_gpus = 1
        TRAINING_GPU_MEMORY = 24



    # TODO: probably use also search_gpu_memory > 11



    # INITIALIZE DATASET
    train_dataset_settings = ReturnnDatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        train_partition_epoch=partition_epoch_factor,
        train_seq_ordering="laplace:.1000",
        #train_additional_options={
        #    "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        #},
    )
    sampling_alpha = 0.7
    vocab_size = 10_240  # TODO: TD - this should not be hardcoded
    # TODO: not using:
    # - preemphasis=None,
    # - peak_normalization=True,
    # - "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
    # - sampling_alpha = 0.7

    training_datasets, dev_dataset_tuples, test_dataset_tuples = create_llm_datasets_jobs(experiment_path,
                                                                                          train_dataset_settings,
                                                                                          vocab_size,
                                                                                          sampling_alpha)

    # NETWORK
    training_alias = "v1"  # TODO: could be imported - use enums perhaps
    decoder_alias = "Qwen2-0_5B" # TODO: could be imported - use enums perhaps
    model_alias, network_args = get_network_args_and_alias(training_alias, decoder_alias)

    # MODEL TRAINING
    training_name = f"{experiment_path}/{NETWORK_MODULE}/{model_alias}"
    train_job = create_training_job(training_name, training_datasets, num_gpus,
                                    NETWORK_MODULE, network_args,
                                    TRAIN_STEP_MODULE, partition_epochs,
                                    debug_returnn_param,
                                    returnn_root=RETURNN_ROOT)
    train_job.rqmt["gpu_mem"] = TRAINING_GPU_MEMORY

    # MODEL EVALUATION/INFERENCE
    # Which evals to run
    if debug:
        run_test = run_best_4 = run_best = False
        epochs_to_evaluate = [partition_epochs]
    else:
        run_best_4 = run_best = run_test = True
        epochs_to_evaluate = default_returnn_keep_epochs(partition_epochs)

    # Tune-Eval
    results = create_tune_and_evaluate_jobs(
        training_name=training_name,
        train_job=train_job,

        network_module=NETWORK_MODULE,
        net_args=network_args,
        debug=debug_returnn_param,

        train_data=training_datasets,
        decoder_config=DecoderConfig(),
        decoder_module=RECOGNITION_PACKAGE,

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


def get_network_args_and_alias(training_alias: str, decoder_alias: str) -> tuple[str, dict[str, Any]]:
    """
    Builds network arguments and alias for the model.

    :param training_alias:
    :param decoder_alias:
    :return:
    """
    # Encoder Config
    training_config = copy.deepcopy(llm_training_configs[training_alias])  # TODO: this should be perfected

    # Decoder Config
    qwen2_decoder_config_job = Qwen2DecoderConfigJob(decoder_alias, training_config["bos_idx"],
                                                     training_config["eos_idx"], training_config["vocab_size"],
                                                     target_filename=f"config-{decoder_alias}-for-i6-spm.json")
    decoder_config = {"config_path": qwen2_decoder_config_job.out_file}

    # Full Model
    model_alias = f"LLM-{training_alias}-{decoder_alias}"
    network_args = training_config | decoder_config  # TODO: improve, dict collisions might happen (for now only config_path)
    return model_alias, network_args


def create_llm_datasets_jobs(prefix_name: str, train_settings: ReturnnDatasetSettings, vocab_size: int, sampling_alpha: float) -> \
        tuple[TrainingDatasets, Dict[str, Tuple[Dataset, tk.Path]], Dict[str, Tuple[Dataset, tk.Path]]]:
    """
    build the training datasets object containing train, cv, dev-train and the extern_data dict
    :param prefix_name:
    :param train_settings:
    :param vocab_size:
    :return:
    """
    training_datasets: TrainingDatasets = build_spm_lm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        return_settings=train_settings,
        vocab_size=vocab_size,
        returnn_root=MINI_RETURNN_ROOT,
        alpha=sampling_alpha,
    )

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_lm_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_lm_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )
    return training_datasets, dev_dataset_tuples, test_dataset_tuples,
