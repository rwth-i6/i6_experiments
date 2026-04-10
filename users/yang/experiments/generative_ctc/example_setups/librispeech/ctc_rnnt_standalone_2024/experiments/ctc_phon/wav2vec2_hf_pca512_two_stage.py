import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...config import get_pca_fit_config
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import compute_pca_state, training, prepare_asr_model, search


def eow_phon_ls960_wav2vec2_hf_ctc_pca512_two_stage():
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_wav2vec2_hf_pca512_two_stage"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
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

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.wav2vec2_hf_ctc_v2_cfg import ModelConfig

    base_model_config = ModelConfig(
        label_target_size=vocab_size_without_blank,
        hf_model_name="facebook/wav2vec2-base",
        pretrained=True,
        freeze_feature_encoder=True,
        freeze_encoder=True,
        apply_spec_augment=True,
        final_dropout=0.1,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        activation_dropout=0.1,
        layerdrop=0.05,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        ctc_loss_reduction="sum",
        gradient_checkpointing=False,
        aux_ctc_loss_layers=[3, 6, 9, -1],
        aux_ctc_loss_scales=[0.3, 0.3, 0.3, 1.0],
        pca_dim=512,
    )

    stage1_model_config = copy.deepcopy(base_model_config)
    stage1_model_config.update_pca_during_training = True
    stage1_model_config.freeze_output_layers = True

    stage1_forward_config = {
        "batch_size": 1000 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "torch_amp_options": {"dtype": "bfloat16"},
        "num_workers_per_gpu": 2,
    }

    network_module = "ctc.wav2vec2_hf_ctc_v2"
    stage1_net_args = {"model_config_dict": asdict(stage1_model_config)}
    stage1_returnn_config = get_pca_fit_config(
        forward_dataset=train_data.prior,
        network_module=network_module,
        config=stage1_forward_config,
        net_args=stage1_net_args,
        debug=False,
    )

    stage1_name = ".wav2vec2_base_pca512_stage1_fitpca_fullpass"
    stage1_training_name = prefix_name + "/" + network_module + stage1_name

    stage1_outputs, stage1_job = compute_pca_state(
        stage1_training_name,
        stage1_returnn_config,
        **default_returnn,
    )
    stage1_job.rqmt["gpu_mem"] = 24

    stage2_model_config = copy.deepcopy(base_model_config)
    stage2_model_config.pca_state_path = stage1_outputs["pca_state.pt"]
    stage2_model_config.update_pca_during_training = False
    stage2_model_config.freeze_output_layers = False

    stage2_train_config = {
        "optimizer": {"class": "adamw", "epsilon": 1e-08, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(1e-05, 3e-05, 40))
        + list(np.linspace(3e-05, 5e-05, 40))
        + list(np.linspace(5e-05, 1e-06, 120)),
        "batch_size": 200 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 2,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
        "num_workers_per_gpu": 2,
    }

    stage2_train_args = {
        "config": stage2_train_config,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(stage2_model_config)},
        "use_speed_perturbation": True,
        "debug": False,
    }

    stage2_name = ".wav2vec2_base_pca512_stage2_fixedenc_fixedpca_aux3369final_lr5e-5_bs120x2_amp"
    stage2_training_name = prefix_name + "/" + network_module + stage2_name

    stage2_job = training(stage2_training_name, train_data, stage2_train_args, num_epochs=200, **default_returnn)
    stage2_job.rqmt["gpu_mem"] = 24

    asr_model = prepare_asr_model(
        stage2_training_name,
        stage2_job,
        stage2_train_args,
        with_prior=True,
        datasets=train_data,
        get_specific_checkpoint=200,
    )

    for decode_layer_index in [6]:
        decoder_config = DecoderConfig(
            lexicon=get_text_lexicon(),
            returnn_vocab=label_datastream.vocab,
            beam_size=1024,
            beam_size_token=12,
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
            lm_weight=1.8,
            prior_scale=0.3,
            decode_layer_index=decode_layer_index,
        )

        search(
            stage2_training_name + f"/decode_dev_layer_{decode_layer_index}",
            forward_config={"batch_size": 50 * 16000},
            asr_model=copy.deepcopy(asr_model),
            decoder_module="ctc.decoder.flashlight_ctc_v1",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            use_gpu=True,
            **default_returnn,
        )

        search(
            stage2_training_name + f"/decode_test_layer_{decode_layer_index}",
            forward_config={"batch_size": 50 * 16000},
            asr_model=copy.deepcopy(asr_model),
            decoder_module="ctc.decoder.flashlight_ctc_v1",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=test_dataset_tuples,
            use_gpu=True,
            **default_returnn,
        )


py = eow_phon_ls960_wav2vec2_hf_ctc_pca512_two_stage
