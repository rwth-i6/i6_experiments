import copy
import math
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel


def eow_phon_ls100_ctc_base(model_conf_w2v: Optional[dict] = None, train_conf_w2v: Optional[dict] = None):
    prefix_name = "example_setups/ctc_eow_phon"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets( 
        prefix=prefix_name,
        librispeech_key="train-clean-100", # TODO: Change to "train-clean-100" for the final setup
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

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

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.decoder.flashlight_ctc_v1 import DecoderConfig

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        lm_scales: List[float],
        prior_scales: List[float],
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass
        :param lm_scales: lm scales for tuning
        :param prior_scales: prior scales for tuning, same length as lm scales
        """
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module="decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn,
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module="decoder.flashlight_ctc_v1", # TODO: maybe need to change file
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn,
            )

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.wav2vec.w2v_ctc_wrapper_cfg import ModelConfig

    # num_epochs = n_updates / (corpus_size / batch_size) = 80k / (100h / 1920s) ~= 427
    num_epochs = 427
    init_lr_scale = 0.01
    final_lr_scale = 0.05
    lr = 3e-5
    if train_conf_w2v is None:
        # default train config
        train_conf_w2v = {
            "optimizer": {"class": "adam", "betas": [0.9, 0.98], "eps": 1e-8, "weight_decay": 0.0, },
            "learning_rates": list(np.linspace(lr * init_lr_scale, lr, int(math.ceil(num_epochs * 0.1))))
            + list(np.linspace(lr, lr, int(math.ceil(num_epochs * 0.4))))
            + list(np.geomspace(lr, final_lr_scale * lr, int(math.ceil(num_epochs * 0.5)))), # tri-stage lr schedule, see:
            # https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/lr_scheduler/tri_stage_lr_scheduler.py
            "batch_size": 1920 * 16000 / 8, # batch size: 1920s, 16000 samples per second, accum_grad 8
            "accum_grad_multiple_step": 8,
            "gradient_clip": 1
        }

    # see: https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/config/finetuning/base_100h.yaml
    if model_conf_w2v is None:
        # default model config
        model_conf_w2v = {
            "_name": "wav2vec_ctc",
            "w2v_path": "/u/andreas.pletschko/fairseq/models/wav2vec_small.pt",
            "apply_mask": True,
            "mask_prob": 0.65,
            "mask_channel_prob": 0.5,
            "mask_channel_length": 64,
            "layerdrop": 0.1,
            "activation_dropout": 0.1,
            "feature_grad_mult": 0.0,
            "freeze_finetune_updates": 10000 # was 0 in fairseq config 
        }
    task_conf_w2v = {
        "_name": "audio_finetuning",
        "normalize": False,
        "data": "-", # can be ignored
    }

    net_args_w2v = ModelConfig(
        model_config_updates = model_conf_w2v,
        task_config_updates = task_conf_w2v,
        label_target_size = vocab_size_without_blank,
    )

    network_module_w2v = "wav2vec.w2v_ctc_wrapper"
    train_args_w2v = {
        # default train config
        "config": train_conf_w2v,
        "network_module": network_module_w2v,
        "net_args": {"w2v_config_updates": asdict(net_args_w2v)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module_w2v + ".ls100_24gbgpu"
    train_job = training(training_name, train_data, train_args_w2v, num_epochs=num_epochs, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_w2v, with_prior=True, datasets=train_data, get_specific_checkpoint=num_epochs
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
    )

    # No improvement, just as example
    # asr_model_best4 = prepare_asr_model(
    #     training_name+ "/best4", train_job, train_args, with_prior=True, datasets=train_data, get_best_averaged_checkpoint=(4, "dev_loss_ctc")
    # )
    # tune_and_evaluate_helper(training_name + "/best4", asr_model_best4, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4])

