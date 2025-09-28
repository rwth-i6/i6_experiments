import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from sisyphus import tk
from functools import partial

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.loquacious.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ..data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
# from ..data.spm import build_spm_training_datasets
from ..data.bpe import build_bpe_training_datasets
from ..pipeline import training
from .tune_eval import build_base_report, eval_model
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, MINI_RETURNN_ROOT
from ...report import generate_report
from ...recognition.aed.beam_search import DecoderConfig


def aed_small_baseline():
    prefix_name = "experiments/loquacious/aed/small/baselines"

    train_settings = DatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
        train_additional_options={
            "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        }
    )

    # # build the training datasets object containing train, cv, dev-train and the extern_data dict
    # train_data = build_bpe_training_datasets(
    #     prefix=prefix_name,
    #     loquacious_key="train.small",
    #     bpe_size=10_000,
    #     settings=train_settings,
    #     use_postfix=False,
    # )

    short_dev_dataset_tuples = {
        "dev.short": build_short_dev_dataset(train_settings)
    }

    dev_dataset_tuples = {}
    for testset in ["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test.commonvoice", "test.librispeech", "test.voxpopuli", "test.yodas"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": RETURNN_ROOT,
    }

    report = {}
    batch_size_factor = 160
    batch_size = 15_000
    default_decoder_config = DecoderConfig()

    for model_config, model_alias, epochs, bpe_size, out_dim in [
        (copy.deepcopy(model_configs.v2), "v2", 125, 10_000, 10_240),
        (copy.deepcopy(model_configs.v2), "v2", 125, 10_000, None),
        (copy.deepcopy(model_configs.v2), "v2", 125, 2_000, None),
        (copy.deepcopy(model_configs.v2), "v2", 125, 1_000, None),
        (copy.deepcopy(model_configs.v3), "v3", 125, 1_000, None),
        (copy.deepcopy(model_configs.v4), "v4", 125, 1_000, None),
    ]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data = build_bpe_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.small",
            bpe_size=bpe_size,
            settings=train_settings,
            use_postfix=False,
        )

        if out_dim is None:
            model_config["out_dim"] = train_data.datastreams["labels"].vocab_size

        sampling_rate = model_config["sampling_rate"]

        network_module = (
            "pytorch_networks.conformer_aed_v1"
        )
        train_config = {
            **optimizer_configs.v1,
            **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs,),
            #############
            "batch_size": batch_size * batch_size_factor,
            "max_seq_length": {"raw_audio": 19.5 * sampling_rate},  # 19.5 seconds
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 5.0,
            "__num_gpus": 4,  # 4, TODO: only for debugging: set to 1 GPU
            "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        }
        # batch size, adamw, speed pert, gradient clip,
        train_args = {
            "config": train_config,
            "post_config": {
                "torch_log_memory_usage": True,
            },
            "network_module": network_module,
            "train_step_module": "training.aed_ctc_train_step",
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
        results = {}
        training_name = (
            prefix_name
            + "/"
            + network_module
            + f"/{model_alias}_bpe-{bpe_size}_{epochs}-ep"
            + (f"_fix-out-dim" if out_dim is None else f"_wrong-out-dim")
        )
        train_job = training(
            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
        )

        eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data,
            decoder_config=default_decoder_config,
            decoder_module="recognition.aed",
            dev_dataset_tuples=short_dev_dataset_tuples,
            result_dict=results,
            specific_epoch=[epochs],
            lm_scales=[0.0],
            prior_scales=[0.0],
            run_test=True,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            run_best=True,
            run_best_4=True,
            use_gpu=True,  # CPU is way too slow for AED decoding
        )
        # generate_report(results=results, exp_name=training_name)
        # report[training_name] = results
        # del results
        # tk.register_report(
        #     "reports/ls_baseline_report", partial(build_base_report, report), required=report, update_frequency=900
        # )
    return report
