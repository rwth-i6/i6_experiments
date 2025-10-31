import copy
from functools import partial
from typing import Any

from sisyphus import tk

from i6_core.tools.download import DownloadJob
from .configurations import optimizer_configs, learning_rate_configs
from .configurations.training_configs import training_configs
from .default_tools import RETURNN_EXE, RETURNN_ROOT, MINI_RETURNN_ROOT
from .experiments_core.data.dataset_commons import DatasetSettings, build_test_dataset, TrainingDatasets
from .experiments_core.data.spm_utils import build_spm_training_datasets
from .experiments_core.model_creation.training_job_builder import create_training_job
from .experiments_core.reporting.report import create_report_job, build_base_report
from .experiments_core.tuning.evaluation import create_evaluation_jobs
from .recognition.beam_search import DecoderConfig

ROOT_RETURNN_ROOT = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def sllm_ep(
        prefix_name: str = "experiments/librispeech/sllm/ls960/baselines",
        debug: bool = False):
    """
    Sisyphus entry point.

    Objective: prepare experiment execution:
    - Download and prepare datasets
    - Prepare model config and all needed for returnn
    - Indicate wanted outputs

    :param prefix_name: Used for alias creation
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
    train_data, dev_dataset_tuples, test_dataset_tuples = create_datasets_jobs(prefix_name,
                                                                               train_dataset_settings,
                                                                               vocab_size,
                                                                               sampling_alpha)

    # GENERAL TRAINING CONSTANTS
    epochs = 500 if not debug else 1 # TODO: extract to a config file?
    batch_size_factor = 160
    batch_size = 15_000
    num_gpus = 4 if not debug else 1 # TODO: important! link with gpu mem, i think they should be hand in hand
    default_decoder_config = DecoderConfig()

    network_module = "networks.conformer_qwen_v1"  # important! # TODO:  move outside the method. Maybe in a constants class or config file...
    train_step_module = "training.train_step"  # important!
    recognition_module = "recognition"  # important!

    # MODEL CONFIG
    # Encoder Config
    encoder_alias = "v1"
    encoder_config = copy.deepcopy(training_configs[encoder_alias])  # TODO: extract as parameter of method

    # Decoder Config
    decoder_alias = "Qwen2-0_5B"
    download_config_job = DownloadJob("https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/config.json",
                                      target_filename=f"config-{decoder_alias}.json")
    decoder_config = {"config_path": download_config_job.out_file}

    # Full Model
    model_alias = f"{encoder_alias}-{decoder_alias}"
    network_args = encoder_config | decoder_config

    # MODEL TRAINING # TODO: move this inside create_training_job
    train_config = {
        **optimizer_configs.v1,
        **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(
            n_ep=epochs,
        ),
        "batch_size": batch_size * batch_size_factor,
        "max_seq_length": {"raw_audio": 19.5 * network_args["sampling_rate"]},  # 19.5 seconds
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,
        "__num_gpus": num_gpus,
        "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    }

    train_args = {
        "config": train_config,

        "network_module": network_module,
        "net_args": network_args,

        "train_step_module": train_step_module,
        "train_args": {  # TODO: could also be extracted in a file
            "aed_loss_scale": 1.0,
            "aux_loss_scales": (1.0, 1.0),
            "label_smoothing": 0.1,
            "label_smoothing_start_epoch": 0,
        },

        "debug": True,
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module + f"/{model_alias}"

    train_job = create_training_job(training_name, train_data, train_args, epochs, **ROOT_RETURNN_ROOT)

    # MODEL EVALUATION/INFERENCE
    if not debug:
        run_best_4 = run_best = run_test = True
        epochs_to_evaluate = [epochs]
    else:
        run_test = True
        run_best_4 = run_best = False
        epochs_to_evaluate = []

    results = create_evaluation_jobs(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data,
        decoder_config=default_decoder_config,
        decoder_module=recognition_module,

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
