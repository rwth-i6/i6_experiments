import copy
from functools import partial

from i6_experiments.users.juanola.data.dataset_settings.dataset_settings import ReturnnDatasetSettings
from i6_experiments.users.juanola.experiments.e25_09_08_demo1_sllm.aed import model_configs, optimizer_configs, \
    learning_rate_configs
from sisyphus import tk

from .tune_eval import build_base_report, eval_model
from ..data.common import build_test_dataset
from ..data.spm import build_spm_training_datasets
from ..default_tools import RETURNN_EXE, RETURNN_ROOT, MINI_RETURNN_ROOT
from ..pipeline import create_training_job
from ..recognition.aed.beam_search import DecoderConfig
from ..report import generate_report

ROOT_RETURNN_ROOT = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def aed_baseline():
    prefix_name = "experiments/librispeech/aed/ls960/baselines"

    # INITIALIZE DATASET
    train_settings = ReturnnDatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        train_partition_epoch=20,
        train_seq_ordering="laplace:.1000",
        train_additional_options={
            "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        },
    )

    # TRAINING PARTITIONS
    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_spm_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
        vocab_size=10_240,
        returnn_root=MINI_RETURNN_ROOT,  # to import ogg zip job from Nick
        alpha=0.7,  # alpha for sampling
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

    # GENERAL TRAINING CONSTANTS
    epochs = 500
    batch_size_factor = 160
    batch_size = 15_000
    default_decoder_config = DecoderConfig()

    # Build network, train, eval, report for each configuration in model_configs
    for model_config, model_alias in [
        # (copy.deepcopy(model_configs.v1), "v1"),  # did not converge
        (copy.deepcopy(model_configs.v2), "v2"),
        (copy.deepcopy(model_configs.v3), "v3"),  # For testing, worst than v2
    ]:
        # MODEL TRAINING
        sampling_rate = model_config["sampling_rate"]
        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(
                n_ep=epochs,
            ),
            "batch_size": batch_size * batch_size_factor,
            "max_seq_length": {"raw_audio": 19.5 * sampling_rate},  # 19.5 seconds
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 4,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        }
        network_module = "pytorch_networks.conformer_aed_v1"  # TODO: important!
        train_step_module = "training.aed_ctc_train_step"  # TODO: important!
        train_args = {
            "config": train_config,
            "network_module": network_module,
            "train_step_module": train_step_module,
            "net_args": model_config,
            "train_args": {
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

        # MODEL EVALUATION
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=default_decoder_config,
            decoder_module="recognition.aed",
            dev_dataset_tuples=dev_dataset_tuples,
            specific_epoch=[500],  # epochs,
            lm_scales=[0.0],
            prior_scales=[0.0],
            run_test=True,
            test_dataset_tuples=test_dataset_tuples,
            run_best=True,
            run_best_4=True,
            use_gpu=True,  # CPU is way too slow for AED decoding
        )

        # MODEL REPORT
        generate_report(results=results, exp_name=training_name)
        report = {training_name: results}
        del results  # TODO: MJ: understand better, i guess delete in memory results
        tk.register_report(
            "reports/ls_baseline_report", partial(build_base_report, report), required=report, update_frequency=900
        )

    return report
