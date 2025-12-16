from dataclasses import asdict
from functools import partial
from typing import Any, Dict, Tuple

from sisyphus import tk

from returnn_common.datasets import Dataset
from .constants import NETWORK_MODULE, TRAIN_STEP_MODULE, RECOGNITION_PACKAGE
from .default_tools import RETURNN_ROOT, MINI_RETURNN_ROOT
from .experiments_core.data.dataset_commons import ReturnnDatasetSettings, build_test_dataset
from .experiments_core.data.spm_utils import build_spm_training_datasets
from .experiments_core.model_creation.training_job_builder import create_training_job
from .experiments_core.reporting.report import create_report_job, build_base_report
from .experiments_core.tuning.evaluation import create_tune_and_evaluate_jobs
from .configurations.data.dataset_config import DatasetConfig
from .configurations.experiment_config import ExperimentConfig
from .configurations.experiment_version import get_experiment_config
from ...data.training_datasets import TrainingDatasets
from ...sisyphus_jobs.configs.qwen2_decoder_config_job_v2 import Qwen2DecoderConfigJobV2
from ...utils.returnn.checkpoint_helper import default_returnn_keep_epochs


def sllm_ep(
        experiment_versions=None,
        experiment_path: str = "experiments/librispeech/sllm/ls960/baselines",
        debug: bool = False,
        itc_training: bool = False) -> Dict[str, Any]:
    """
    Sisyphus entry point.

    Objective: prepare experiment execution:
    - Download and prepare datasets
    - Prepare model config and all needed for returnn
    - Indicate wanted outputs

    :param experiment_versions: list of experiment versions to run. Default baseline
    :param experiment_path: Used for alias creation
    :type debug: Used to set up config for debugging in one GPU
    :param itc_training: Makes return training jobs run on ITC
    """
    assert experiment_versions is not None, "at least one of experiment_versions is required"
    assert len(experiment_versions) > 0, "experiment_versions cannot be empty"

    reports = {}
    for exp_name, exp_config in [(v.value, get_experiment_config(v)) for v in experiment_versions]:
        # TODO: extract inside

        # GENERAL CONSTANTS

        # Returnn
        debug_returnn_param = True  # TODO: Make it depend on big debug?

        # Training
        epochs: int = exp_config.training.epochs
        partition_epoch_factor: int = exp_config.training.partition_epoch_factor
        partition_epochs: int = int(
            epochs * partition_epoch_factor / exp_config.training.num_gpus)  # 2000 (1GPU) | 500 (4GPU)
        TRAINING_GPU_MEMORY = exp_config.training.gpu_memory
        TRAINING_BATCH_SIZE = exp_config.training.batch_size

        # Search
        SEARCH_GPU_MEMORY = exp_config.search.gpu_memory
        RECOGNITION_BATCH_SIZE = exp_config.search.batch_size
        PRIOR_BATCH_SIZE = exp_config.prior.batch_size


        # DEBUGGING CHANGES
        if debug: # TODO: this should modify the experiment object!
            TRAINING_BATCH_SIZE = 21_000
            partition_epochs = 1
            # TODO: move this to config? NUM_GPUS = 1
            TRAINING_GPU_MEMORY = 48


        # INITIALIZE DATASET
        training_datasets, dev_dataset_tuples, test_dataset_tuples = create_datasets_jobs(experiment_path,
                                                                                          exp_config.dataset,
                                                                                          partition_epoch_factor,
                                                                                          exp_config.labels.vocab_size)

        # NETWORK
        model_alias = exp_config.network.name
        network_args = get_network_args_and_alias(exp_config)

        # MODEL TRAINING
        training_name = f"{experiment_path}/{NETWORK_MODULE}/{model_alias}/{exp_name}"
        train_job = create_training_job(training_name, training_datasets, TRAINING_BATCH_SIZE,
                                        NETWORK_MODULE, network_args,
                                        TRAIN_STEP_MODULE, partition_epochs,
                                        debug_returnn_param,
                                        exp_config.training,
                                        returnn_root=RETURNN_ROOT)
        train_job.rqmt["gpu_mem"] = TRAINING_GPU_MEMORY

        # ITC Training
        if itc_training:
            train_job.hold()
            train_job.move_to_hpc = True
            train_job.rqmt["time_rqmt"] = 36  # ??

        # MODEL EVALUATION/INFERENCE
        # Which evals to run
        if debug:
            run_test = run_best_4 = run_best = False
            epochs_to_evaluate = [partition_epochs]
        else:
            run_best_4 = run_best = run_test = True
            specific_epochs = set({})
            epochs_to_evaluate = default_returnn_keep_epochs(partition_epochs) | specific_epochs

        # Tune-Eval
        results: Dict[Any, Any] = create_tune_and_evaluate_jobs(
            training_name=training_name,
            train_job=train_job,

            network_module=NETWORK_MODULE,
            net_args=network_args,
            debug=debug_returnn_param,

            train_data=training_datasets,
            decoder_config=exp_config.search.beam_search,
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
            search_gpu_memory=SEARCH_GPU_MEMORY,  # breaks for bigger searches

            recognition_batch_size=RECOGNITION_BATCH_SIZE,
            prior_batch_size=PRIOR_BATCH_SIZE,
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

        reports[training_name] = report  # TODO:REFACTOR change with config name

    return reports


def get_network_args_and_alias(config: ExperimentConfig) -> dict[str, Any]:
    """
    Builds network arguments and alias for the model.

    :param config:
    :return:
    """
    label_config = asdict(config.labels)
    fe_config = asdict(config.network.feature_extraction)
    encoder_config = asdict(config.network.encoder)
    adapter_config = asdict(config.network.adapter)
    qwen2_decoder_config_job = Qwen2DecoderConfigJobV2(config.network.decoder, config.labels, target_filename=f"config-{config.network.decoder.name}-for-i6-spm.json")
    decoder_config = {"config_path": qwen2_decoder_config_job.out_file}

    network_args = label_config | fe_config | encoder_config | adapter_config | decoder_config
    return network_args


def create_datasets_jobs(prefix_name: str, dataset_config: DatasetConfig, partition_epoch_factor: int, vocab_size: int,) -> \
        tuple[TrainingDatasets, Dict[str, Tuple[Dataset, tk.Path]], Dict[str, Tuple[Dataset, tk.Path]]]:
    """
    build the training datasets object containing train, cv, dev-train and the extern_data dict
    :param prefix_name:
    :param train_settings:
    :param vocab_size:
    :param sampling_alpha:
    :return:
    """
    train_dataset_settings = ReturnnDatasetSettings(
        preemphasis=dataset_config.preemphasis,
        peak_normalization=dataset_config.peak_normalization,
        train_partition_epoch=partition_epoch_factor,
        train_seq_ordering=dataset_config.train_seq_ordering,
        train_additional_options=dataset_config.train_additional_options,
    )

    training_datasets: TrainingDatasets = build_spm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        return_settings=train_dataset_settings,
        vocab_size=vocab_size,
        returnn_root=MINI_RETURNN_ROOT,  # to import ogg zip job from Nick
        alpha=dataset_config.sampling_alpha,
    )

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_dataset_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_dataset_settings,
        )
    return training_datasets, dev_dataset_tuples, test_dataset_tuples,
