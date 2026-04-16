import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict, Sequence, Optional, Union, Tuple, List

from sisyphus import tk
from functools import partial

from i6_core.tools.download import DownloadJob
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from . import model_configs
from ..trafo_lm.model_configs import get_trafo_lm_config_v1
from ..data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
# from ..data.spm import build_spm_training_datasets
from ..data.bpe import build_bpe_training_datasets
from ..data.audio_only import build_audio_only_training_dataset
from ..pipeline import training
from .tune_eval import build_base_report, eval_model
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, MINI_RETURNN_ROOT
from ..report import generate_report
from ...recognition.aed.beam_search import DecoderConfig, DecoderWithLmConfig
# from ..codebook_ppl import calculate_wav2vec_codebook_ppl


default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


def _get_default_decoder_config() -> DecoderConfig:
    return DecoderConfig(
        beam_size=12,
        max_tokens_per_sec=20,
        sample_rate=16_000,
    )


def aux_ctc_recog(train_job: ReturnnTrainingJob):
    prefix_name = "experiments/loquacious/ctc/small/aux_ctc"

    batch_size_factor = 160  # 16kHz
    batch_size = 15_000
    default_decoder_config = _get_default_decoder_config()

    train_settings = DatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
        train_additional_options={
            "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        },
        num_workers=4,
    )

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

    add_test_sets = {}
    for train_set in ["train.small", "train.medium"]:
        add_test_sets[train_set] = build_test_dataset(
            dataset_key=train_set,
            settings=train_settings,
        )

    train_configs = [
        ({
            "model_config": copy.deepcopy(model_configs.v2),
            "model_alias": "v2",
            "bpe_size": 1_000,
            "epochs": 125,
           }, (
               f"/model-v2_bpe-1k_125-ep"
        ))
    ]
    for train_config, train_alias in train_configs:
        # train hyperparameters
        bpe_size = train_config["bpe_size"]
        epochs = train_config["epochs"]

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data = build_bpe_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.small",
            bpe_size=bpe_size,
            settings=train_settings,
            use_postfix=False,
        )

        # model hyperparameters
        model_config = train_config["model_config"]
        model_config["out_dim_wo_blank"] = train_data.datastreams["labels"].vocab_size

        extra_forward_config = {
            "preload_from_files": {
                "aed_w_aux_ctc": {
                    "filename": train_job.out_checkpoints[125],
                    # load the second aux loss layer of the aed model as the out logits layer of the ctc model
                    # note, the aed model needs to have this aux loss layer
                    # TODO: make this configurable
                    "var_name_mapping": {
                        "out_logits.weight": "out_aux_logits.2.weight"
                    }
                }
            },
            # "batch_size": 80_000 * batch_size_factor,
        }

        run_recog(
            prefix_name=prefix_name,
            train_data=train_data,
            short_dev_dataset_tuples=short_dev_dataset_tuples,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            add_test_sets=add_test_sets,
            batch_size=batch_size * batch_size_factor,
            decoder_config=train_config.get("decoder_config", default_decoder_config),
            model_config=model_config,
            train_alias=train_alias,
            epochs=epochs,
            checkpoints=[125,],
            extra_forward_config=extra_forward_config,
            train_job=train_job,
        )


def run_recog(
        prefix_name: str,
        train_data,
        train_job,
        short_dev_dataset_tuples: Dict,
        dev_dataset_tuples: Dict,
        test_dataset_tuples: Dict,
        add_test_sets: Dict,
        batch_size: int,
        decoder_config: DecoderConfig,
        model_config: Dict,
        train_alias: str,
        epochs: int,
        checkpoints: Optional[Sequence[Union[str, int]]] = None,
        extra_forward_config: Optional[Dict] = None,
):
    if checkpoints is None:
        checkpoints = [epochs, "best", "best4"]

    network_module = model_config.pop("network_module")

    recog_config = {
        #############
        "batch_size": batch_size,
        "torch_dataloader_opts": {"num_workers": 1},  # for multi proc dataset
    }
    # batch size, adamw, speed pert, gradient clip,
    recog_args = {
        "config": recog_config,
        "post_config": {
            "torch_log_memory_usage": True,
        },
        "network_module": network_module,
        "net_args": model_config,
        "train_args": {
            "aed_loss_scale": 1.0,
            "aux_loss_scales": tuple([1.0] * len(model_config["aux_loss_layers"])),
            "label_smoothing": 0.1,
            "label_smoothing_start_epoch": 0,
        },
        "debug": True,
    }
    results = {}
    training_name = (
            prefix_name
            + "/"
            + network_module
            + train_alias
            # + f"/{model_alias}_bpe-{bpe_size}_{epochs}-ep"
            # + (f"_fix-out-dim" if out_dim is None else f"_wrong-out-dim")
    )

    exp_name = f"{training_name}/recog-{str(decoder_config)}"
    results = eval_model(
        training_name=exp_name,
        train_job=None,
        train_args=recog_args,
        train_data=train_data,
        decoder_config=decoder_config,
        decoder_module="recognition.ctc",
        dev_dataset_tuples=short_dev_dataset_tuples,
        result_dict=results,
        specific_epoch=[epochs],
        lm_scales=[0.0],
        prior_scales=[0.0],
        checkpoints=checkpoints,
        run_test=True,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples, **add_test_sets},
        # run_best=False,  # True,
        # run_best_4=False,  # ./sisTrue,
        use_gpu=True,  # CPU is way too slow for AED decoding
        extra_forward_config=extra_forward_config,
    )
    generate_report(results=results, exp_name=exp_name)
