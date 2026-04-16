import copy
from dataclasses import asdict
import numpy as np
from typing import cast, Dict, Sequence, Optional, Union, Tuple, List

from sisyphus import tk
from functools import partial

from i6_core.tools.download import DownloadJob
from i6_core.returnn.training import ReturnnTrainingJob

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.loquacious.aed import model_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import learning_rate_configs
from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms import optimizer_configs
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.schmitt.util.dict_update import dict_update_deep

from ..trafo_lm.model_configs import get_trafo_lm_config_v1
from ..data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
# from ..data.spm import build_spm_training_datasets
from ..data.bpe import build_bpe_training_datasets
from ..data.audio_only import build_audio_only_training_dataset
from ..pipeline import training
from .tune_eval import build_base_report, eval_model
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, MINI_RETURNN_ROOT
from ..report import generate_report
from ...recognition.aed.beam_search import DecoderConfig, DecoderWithLmAndIlmConfig
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


def _get_default_lm_decoder_config() -> DecoderConfig:
    return DecoderWithLmAndIlmConfig(
        beam_size=12,
        max_tokens_per_sec=20,
        sample_rate=16_000,
        lm_scale=0.0,  # this is set during evaluation
        ilm_scale=0.0,  # this is set during evaluation
    )


def _get_lm_opts(bpe_size_alias: str, bpe_size: int):
    lm_checkpoint = tk.Path(
        f"/u/rossenbach/experiments/tts_decoder_asr/alias/experiments/loquacious/standalone_2025/lm_bpe/"
        f"train_small_bpe_{bpe_size_alias}_trafo/"
        f"lm.trafo.kazuki_trafo_zijian_variant_v2.24x768_2x3k_RAdam_3e-4_5ep_reduce_gcn2.0/"
        f"training/output/models/epoch.500.pt"
    )
    lm_module, lm_network_args = get_trafo_lm_config_v1(vocab_size=bpe_size)

    return {"checkpoint": lm_checkpoint, "lm_module": lm_module, "lm_network_args": asdict(lm_network_args)}


def aed_small_baseline() -> Dict[str, ReturnnTrainingJob]:
    prefix_name = "experiments/loquacious/aed/small/baselines"

    batch_size_factor = 160  # 16kHz
    batch_size = 15_000
    default_decoder_config = _get_default_decoder_config()

    train_jobs = {}

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

    train_configs = [
        # vary bpe size
        *[({
            "model_config": copy.deepcopy(model_configs.v2),
            "model_alias": "v2",
            "bpe_size": bpe_size,
            "epochs": 125,
           }, (
               f"/model-v2_bpe-{bpe_size // 1_000}k_125-ep"
           )) for bpe_size in [
            1_000,
            2_000,
            10_000
        ]],
        # vary dropout
        *[({
               "model_config": dict_update_deep(
                   copy.deepcopy(model_configs.v2),
                   {
                       "dropout": dropout,
                   }
               ),
               "model_alias": "v2",
               "bpe_size": 1_000,
               "epochs": 125,
           }, (
               f"/model-v2_drop-{dropout}_bpe-1k_125-ep"
           )) for dropout in [
            0.2
        ]],
        # vary model dim
        *[({
               "model_config": dict_update_deep(
                   copy.deepcopy(model_configs.v2),
                   {
                       "model_dim": model_dim
                   }
               ),
               "model_alias": "v2",
               "bpe_size": 1_000,
               "epochs": 125,
           }, (
               f"/model-v2_model-dim-{model_dim}_bpe-1k_125-ep"
           )) for model_dim in [
            256
        ]],
        # vary aux loss layers
        *[({
               "model_config": dict_update_deep(
                   copy.deepcopy(model_configs.v2),
                   {
                       "aux_loss_layers": aux_loss_layers
                   }
               ),
               "model_alias": "v2",
               "bpe_size": 1_000,
               "epochs": 125,
           }, (
               f"/model-v2_aux-loss-{aux_loss_layers}_bpe-1k_125-ep"
           )) for aux_loss_layers in [
            (4, 8, 12),
            (12,),
            (8, 12)
        ]],
        # vary regularization
        *[({
            "model_config": dict_update_deep(
               copy.deepcopy(model_configs.v2),
               {
                   "specaug_start": specaug_start
               }
            ),
            "model_alias": "v2",
            "bpe_size": 1_000,
            "epochs": 125,
            "use_speed_perturbation": use_speed_perturbation
           }, (
               f"/model-v2_speedpert-{use_speed_perturbation}_specaug-{specaug_start}_bpe-1k_125-ep"
           )) for use_speed_perturbation, specaug_start in [
            (False, None),
            (True, None),
            (False, (5_000, 15_000, 25_000))
        ]],
        # vary beam sizes
        *[({
                "model_config": copy.deepcopy(model_configs.v2),
                "model_alias": "v2",
                "bpe_size": 1_000,
                "epochs": 125,
                "decoder_config": DecoderConfig(
                    beam_size=beam_size,
                    max_tokens_per_sec=default_decoder_config.max_tokens_per_sec,
                    sample_rate=default_decoder_config.sample_rate,
                )

           }, (
               f"/model-v2_bpe-1k_125-ep"
           )) for beam_size in [
            1, 4, 8, 16, 32
        ]],
        # LM decoding
        ({
            "model_config": copy.deepcopy(model_configs.v2),
            "model_alias": "v2",
            "bpe_size": 1_000,
            "epochs": 125,
            "decoder_config": _get_default_lm_decoder_config(),
            "lm_scales": [0.2, 0.3, 0.4],
            "ilm_scales": [0.1, 0.2, 0.3],
            "recog_batch_size": 10_000,
            "checkpoints": [125]
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
        model_config["out_dim"] = train_data.datastreams["labels"].vocab_size

        recog_batch_size = None
        if "recog_batch_size" in train_config:
            recog_batch_size = train_config["recog_batch_size"] * batch_size_factor

        train_job = run_experiment(
            prefix_name=prefix_name,
            train_data=train_data,
            short_dev_dataset_tuples=short_dev_dataset_tuples,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            batch_size=batch_size * batch_size_factor,
            decoder_config=train_config.get("decoder_config", default_decoder_config),
            lm_scales=train_config.get("lm_scales"),
            ilm_scales=train_config.get("ilm_scales"),
            model_config=model_config,
            train_alias=train_alias,
            epochs=epochs,
            use_speed_perturbation=train_config.get("use_speed_perturbation", True),
            bpe_size=bpe_size,
            recog_batch_size=recog_batch_size,
            checkpoints=train_config.get("checkpoints"),
        )
        train_jobs.update(train_job)

    return train_jobs


def aed_medium_baseline():
    prefix_name = "experiments/loquacious/aed/medium/baselines"

    batch_size_factor = 160  # 16kHz
    batch_size = 15_000
    default_decoder_config = _get_default_decoder_config()

    train_settings = DatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        # training
        train_partition_epoch=20,
        train_seq_ordering="laplace:.1000",
        train_additional_options={
            "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        },
        num_workers=2
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


    train_configs = [
        *[({
               "model_config": copy.deepcopy(model_configs.v2),
               "model_alias": "v2",
               "bpe_size": bpe_size,
               "epochs": epochs,
           }, (
               f"/model-v2_bpe-{bpe_size // 1000}k_{epochs}-ep"
           )) for bpe_size, epochs in [
            (1000, 200),
            (2000, 200),
            # (10_000, 200),  # done below
            (1000, 100),
            (2000, 100),
            # (10_000, 100),  # done below
        ]
        ],
        *[({
               "model_config": dict_update_deep(
                   copy.deepcopy(model_configs.v2),
                   {
                       "model_dim": model_dim
                   }
               ),
               "model_alias": "v2",
               "bpe_size": 10_000,
               "epochs": 200,
           }, (
               f"/model-v2_model-dim{model_dim}_bpe-10k_200-ep"
           )) for model_dim in [
            1024
        ]
        ],
        # vary beam sizes; 100 epochs
        *[({
            "model_config": copy.deepcopy(model_configs.v2),
            "model_alias": "v2",
            "bpe_size": 10_000,
            "epochs": 100,
            "decoder_config": DecoderConfig(
                beam_size=0,  # dummy value, will be overwritten below
                max_tokens_per_sec=default_decoder_config.max_tokens_per_sec,
                sample_rate=default_decoder_config.sample_rate,
            ),
            "beam_sizes": (1, 4, 8, 12, 16, 32
                           # 4, 8, 12, 16, 32
                           ),

        }, (
            f"/model-v2_bpe-10k_100-ep"
        ))
        ],
        # vary beam sizes; 200 epochs
        *[({
                "model_config": copy.deepcopy(model_configs.v2),
                "model_alias": "v2",
                "bpe_size": 10_000,
                "epochs": 200,
                "decoder_config": DecoderConfig(
                   beam_size=0,  # dummy value, will be overwritten below
                   max_tokens_per_sec=default_decoder_config.max_tokens_per_sec,
                   sample_rate=default_decoder_config.sample_rate,
                ),
                "beam_sizes": (1, 4, 8, 12, 16, 32),

           }, (
               f"/model-v2_bpe-10k_200-ep"
           ))
        ],
        # greedy; 200 epochs
        *[({
               "model_config": copy.deepcopy(model_configs.v2),
               "model_alias": "v2",
               "bpe_size": 10_000,
               "epochs": 200,
               "decoder_config": DecoderConfig(
                   beam_size=0,  # dummy value, will be overwritten below
                   max_tokens_per_sec=default_decoder_config.max_tokens_per_sec,
                   sample_rate=default_decoder_config.sample_rate,
               ),
               "beam_sizes": (1,),

           }, (
               f"/model-v2_bpe-10k_200-ep"
           ))
        ],
    ]
    for train_config, train_alias in train_configs:
        # train hyperparameters
        bpe_size = train_config["bpe_size"]
        epochs = train_config["epochs"]

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data = build_bpe_training_datasets(
            prefix=prefix_name,
            loquacious_key="train.medium",
            bpe_size=bpe_size,
            settings=train_settings,
            use_postfix=False,
        )

        # model hyperparameters
        model_config = train_config["model_config"]
        model_config["out_dim"] = train_data.datastreams["labels"].vocab_size

        run_experiment(
            prefix_name=prefix_name,
            train_data=train_data,
            short_dev_dataset_tuples=short_dev_dataset_tuples,
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            batch_size=batch_size * batch_size_factor,
            decoder_config=train_config.get("decoder_config", default_decoder_config),
            model_config=model_config,
            train_alias=train_alias,
            epochs=epochs,
            gpu_mem=11 if model_config["model_dim"] <= 512 else 24,
            bpe_size=bpe_size,
            beam_sizes=train_config.get("beam_sizes"),
        )


def run_experiment(
        prefix_name: str,
        train_data,
        short_dev_dataset_tuples: Dict,
        dev_dataset_tuples: Dict,
        test_dataset_tuples: Dict,
        batch_size: int,
        decoder_config: DecoderConfig,
        model_config: Dict,
        train_alias: str,
        epochs: int,
        bpe_size: int,
        gpu_mem: int = 11,
        use_speed_perturbation: bool = True,
        lm_scales: Optional[List[float]] = None,
        ilm_scales: Optional[List[float]] = None,
        checkpoints: Optional[Sequence[Union[str, int]]] = None,
        recog_batch_size: Optional[int] = None,
        beam_sizes: Optional[Sequence[int]] = None,
):
    extra_forward_config = {}

    if checkpoints is None:
        checkpoints = [epochs, "best", "best4"]
    if lm_scales is None:
        lm_scales = [0.0]
    if ilm_scales is None:
        ilm_scales = [0.0]
    if recog_batch_size is not None:
        extra_forward_config["batch_size"] = recog_batch_size
    if beam_sizes is None:
        beam_sizes = [12]

    sampling_rate = model_config["sampling_rate"]

    network_module = model_config.pop("network_module")

    model_config_eval = copy.deepcopy(model_config)
    if lm_scales != [0.0]:
        lm_opts = _get_lm_opts(bpe_size_alias=str(bpe_size), bpe_size=model_config["out_dim"])
        lm_checkpoint = lm_opts.pop("checkpoint")
        extra_forward_config.update({
            "preload_from_files": {
                "external_lm": {
                    "filename": lm_checkpoint,
                    "prefix": "external_lm.",

                }
            }
        })
        model_config_eval.update({"lm_opts": lm_opts})

    train_config = {
        **optimizer_configs.v1,
        **learning_rate_configs.get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep=epochs, ),
        #############
        "batch_size": batch_size,
        "max_seq_length": {"raw_audio": 19.5 * sampling_rate},  # 19.5 seconds
        "accum_grad_multiple_step": 1,
        "gradient_clip_global_norm": 5.0,
        "__num_gpus": 4,
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
            "aux_loss_scales": tuple([1.0] * len(model_config["aux_loss_layers"])),
            "label_smoothing": 0.1,
            "label_smoothing_start_epoch": 0,
        },
        "debug": True,
        "use_speed_perturbation": use_speed_perturbation,
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
    train_job = training(
        training_name, train_data, train_args, num_epochs=epochs, gpu_mem=gpu_mem, **default_returnn
    )

    eval_args = copy.deepcopy(train_args)
    eval_args["net_args"] = model_config_eval

    exp_name = f"{training_name}/recog_beam-sizes-{beam_sizes}lms-{lm_scales}_ilms-{ilm_scales}"
    results = eval_model(
        training_name=exp_name,
        train_job=train_job,
        train_args=eval_args,
        train_data=train_data,
        decoder_config=decoder_config,
        decoder_module="recognition.aed",
        dev_dataset_tuples=short_dev_dataset_tuples,
        result_dict=results,
        specific_epoch=[epochs],
        lm_scales=lm_scales,
        ilm_scales=ilm_scales,
        checkpoints=checkpoints,
        run_test=True,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        # run_best=False,  # True,
        # run_best_4=False,  # ./sisTrue,
        use_gpu=True,  # CPU is way too slow for AED decoding
        extra_forward_config=extra_forward_config,
        beam_sizes=beam_sizes,
    )
    generate_report(results=results, exp_name=exp_name)

    print(f"Running {training_name}")
    return {training_name: train_job}


def calculate_codebook_ppl_small():
    prefix_name = "experiments/loquacious/codebook_ppl/small"

    wav2vec2_chkpt = DownloadJob(
        "https://huggingface.co/facebook/wav2vec2-large-lv60/resolve/main/pytorch_model.bin?download=true",
        target_filename="wav2vec2_large_60kh_no_finetune.bin",
    ).out_file

    wav2vec_fine_tune_config = DownloadJob(
        "https://huggingface.co/facebook/wav2vec2-large-lv60/resolve/main/config.json?download=true",
        target_filename="wav2vec2_large_60kh_no_finetune_config.json",
    ).out_file

    train_settings = DatasetSettings(
        preemphasis=None,
        peak_normalization=True,
        # training
        train_partition_epoch=1,
        train_seq_ordering="sorted",
        num_workers=4,
    )

    train_data = build_audio_only_training_dataset(
        prefix=prefix_name,
        loquacious_key="train.small",
        settings=train_settings,
    )

    # calculate_wav2vec_codebook_ppl(
    #     train_data.datastreams["features"],
    #     train_data.datastreams["labels"],
    #     prefix=f"{MINI_RETURNN_ROOT}/egs/loquacious/wav2vec2/exp/aed_small_baseline/bpe-1k",
    #     returnn_exe=RETURNN_EXE,
    #     returnn_root=RETURNN_ROOT,
    # )
