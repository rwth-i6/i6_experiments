from dataclasses import asdict
from typing import Any, Dict, Tuple

from sisyphus import tk

from returnn_common.datasets import Dataset
from .configurations.data.dataset_config import DatasetConfig
from .configurations.data.label_config import LabelConfig
from .configurations.experiment_config import ExperimentConfig
from .configurations.experiment_version import get_experiment_config
from .constants import SIS_BASE_REPORT_EXTENSION, SIS_OUTPUTS_REPORTS, NETWORK_PACKAGE, \
    TRAIN_STEP_PACKAGE
from .default_tools import RETURNN_ROOT, MINI_RETURNN_ROOT
from .experiments_core.data.dataset_commons import ReturnnDatasetSettings, build_lm_test_dataset
from .experiments_core.data.spm_utils import build_spm_lm_training_datasets
from .experiments_core.model_creation.training_job_builder import create_training_job
from .experiments_core.reporting.report_helper import generate_experiment_results_report
from .experiments_core.tuning.evaluation import create_tune_and_evaluate_jobs
from ...data.training_datasets import TrainingDatasets
from ...sisyphus_jobs.configs.qwen2_decoder_config_job_v2 import Qwen2DecoderConfigJobV2
from ...utils.returnn.checkpoint_helper import default_returnn_keep_epochs


def llm_ep(
        experiment_versions=None,
        experiment_path: str = "experiments/librispeech/llm/ls960/baselines",
        debug: bool = False,
        itc_training: bool = False,
        specific_recognition_epochs: set[int] = set({}),
        only_specific_epochs: bool = False,
        test_forward_output_path: bool = False,
) -> Dict[str, Any]:
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
    :param specific_recognition_epochs:
    """
    assert experiment_versions is not None, "at least one of experiment_versions is required"
    assert len(experiment_versions) > 0, "experiment_versions cannot be empty"

    base_exps_name = "-".join([v.value for v in experiment_versions])

    results_per_experiment = {}
    for exp_name, exp_config in [(v.value, get_experiment_config(v)) for v in experiment_versions]:
        # TODO: extract inside

        # Training
        epochs: int = exp_config.training.epochs
        partition_epoch_factor: int = exp_config.training.partition_epoch_factor
        partition_epochs: int = int(
            epochs * partition_epoch_factor / exp_config.training.num_gpus
        )  # 2000 (1GPU) | 500 (4GPU)
        TRAINING_GPU_MEMORY = exp_config.training.gpu_memory
        TRAINING_BATCH_SIZE = exp_config.training.batch_size

        # DEBUGGING CHANGES
        if debug:  # TODO: this should modify the experiment object!
            # TRAINING_BATCH_SIZE = 6_000
            # partition_epochs = 1
            pass

        # INITIALIZE DATASET
        training_datasets, dev_dataset_tuples, test_dataset_tuples = create_llm_datasets_jobs(
            experiment_path, exp_config.dataset, exp_config.labels, partition_epoch_factor
        )

        # NETWORK
        model_alias = exp_config.network.name
        network_args = get_network_args(exp_config)

        network_module = f"{NETWORK_PACKAGE}.{exp_config.network.network_file_name}"
        network_import_path = f"{network_module}.{exp_config.network.network_class_name}"
        train_step_module = f"{TRAIN_STEP_PACKAGE}.{exp_config.network.training_step_file_name}"

        # MODEL TRAINING
        training_name = f"{experiment_path}/{network_module}/{model_alias}/{exp_name}"
        train_job = create_training_job(
            training_name,
            training_datasets,
            TRAINING_BATCH_SIZE,
            network_import_path,
            network_args,
            train_step_module,
            partition_epochs,
            exp_config.training,
            returnn_root=RETURNN_ROOT,
        )
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
            specific_epochs = specific_recognition_epochs | set(
                {}
            )  # Specify here default epochs to check in multiple exps
            epochs_to_evaluate = default_returnn_keep_epochs(partition_epochs, keep_last_epoch=True) | specific_epochs

        if only_specific_epochs:
            run_test = run_best_4 = run_best = False
            epochs_to_evaluate = specific_recognition_epochs

        forward_training_name = training_name if not test_forward_output_path else f"tests/{training_name}"

        # Tune-Eval
        results: Dict[str, Any] = create_tune_and_evaluate_jobs(
            training_name=forward_training_name,
            train_job=train_job,
            network_import_path=network_import_path,
            net_args=network_args,
            search_config=exp_config.search,
            train_data=training_datasets,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            specific_epochs=epochs_to_evaluate,
            run_test=run_test,
            run_best=run_best,
            run_best_4=run_best_4,
        )
        results_per_experiment[exp_name] = results

        # REPORTING
        # Experiment Report
        generate_experiment_results_report(exp_results=results, exp_name=training_name)

        # Update Base Report (for all experiment results)
        tk.register_report(
            f"{SIS_OUTPUTS_REPORTS}/base_report-{base_exps_name}.{SIS_BASE_REPORT_EXTENSION}",
            results_per_experiment,
            # partial(base_report_template_v0, results_per_experiment), # TODO: check the template
            required=results_per_experiment,
            update_frequency=900,
        )

    return results_per_experiment


def get_network_args(config: ExperimentConfig) -> dict[str, Any]:
    """
    Builds network arguments and alias for the model.

    :param config:
    :return:
    """
    label_config = asdict(config.labels)
    fe_config = asdict(config.network.feature_extraction)
    unused_but_not_optional_encoder_config = { # TODO: better fix in model, considering cases without encoder...
        "encoder_dim" : 512,
        "num_heads" : 8,
        "num_enc_layers" : 12,
        "aux_loss_layers" : (4, 8),
    }
    qwen2_decoder_config_job = Qwen2DecoderConfigJobV2(
        config.network.decoder, config.labels, target_filename=f"config-{config.network.decoder.name}-for-i6-spm.json"
    )
    decoder_config = {"config_path": qwen2_decoder_config_job.out_file}

    network_args = label_config | fe_config | decoder_config | unused_but_not_optional_encoder_config
    return network_args


def create_llm_datasets_jobs(prefix_name: str, dataset_config: DatasetConfig, label_config: LabelConfig,
                             partition_epoch_factor: int) -> \
        tuple[TrainingDatasets, Dict[str, Tuple[Dataset, tk.Path]], Dict[str, Tuple[Dataset, tk.Path]]]:
    """
    build the training datasets object containing train, cv, dev-train and the extern_data dict
    :param partition_epoch_factor:
    :param label_config:
    :param dataset_config:
    :param prefix_name:
    :return:
    """
    train_dataset_settings = ReturnnDatasetSettings(
        preemphasis=dataset_config.preemphasis,
        peak_normalization=dataset_config.peak_normalization,
        train_partition_epoch=partition_epoch_factor,
        train_seq_ordering=dataset_config.train_seq_ordering,
    )

    training_datasets: TrainingDatasets = build_spm_lm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        return_settings=train_dataset_settings,
        vocab_size=label_config.vocab_size,
        dataset_config=dataset_config,
        returnn_root=MINI_RETURNN_ROOT,
    )

    dev_dataset_tuples = {}
    for dev_set in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[dev_set] = build_lm_test_dataset(
            dataset_key=dev_set,
            settings=train_dataset_settings,
        )

    test_dataset_tuples = {}
    for test_set in ["test-clean", "test-other"]:
        test_dataset_tuples[test_set] = build_lm_test_dataset(
            dataset_key=test_set,
            settings=train_dataset_settings,
        )
    return training_datasets, dev_dataset_tuples, test_dataset_tuples,
