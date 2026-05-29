"""
Posterior HMM baseline with BPE-128
"""
import copy
from dataclasses import asdict
from sisyphus import tk
import numpy as np
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

from ...data.bpe import build_bpe_training_datasets, get_bpe_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, LIBRASR_WHEEL
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...rasr import CreateLibrasrVenvJob, build_fsa_exporter_config


def bpe128_ls960_0924_base():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_bpe_128"

    BPE_SIZE = 128

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data_bpe = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=BPE_SIZE,
        settings=train_settings,
        use_postfix=False,
        use_raw_text_labels=True,
    )

    label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
    vocab_size_without_blank = 182

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

    phmm_returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        extra_pip_packages=["ninja"],
        python_wrapper_name="python_with_path",
    ).out_python_bin

    default_returnn = {
        "returnn_exe": phmm_returnn_exe,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    bpe_bliss_lexicon = get_bpe_lexicon(librispeech_key="train-other-960", bpe_size=BPE_SIZE)
    tk.register_output(prefix_name + f"/{BPE_SIZE}/bpe_lexicon_phmm.xml.gz", bpe_bliss_lexicon)
    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=bpe_bliss_lexicon,
        corpus_path=librispeech_corpus,
    )

    from ...pytorch_networks.phmm.decoder.greedy_bpe_phmm_v1 import DecoderConfig as GreedyDecoderConfig

    def greedy_search_helper(training_name: str, asr_model: ASRModel, decoder_config: GreedyDecoderConfig):
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="phmm.decoder.greedy_bpe_phmm_v1",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            **default_returnn,
        )

    default_greedy_config = GreedyDecoderConfig(
        lexicon=bpe_bliss_lexicon,
    )

    from ...pytorch_networks.phmm.phmm_zhou_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        ConformerPosEmbConfig,
    )

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # classic style
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        pos_emb_config=posemb_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        mhsa_with_bias=True,
        conv_kernel_size=31,
        final_dropout=0.05,
        specauc_start_epoch=11,
        dropout_broadcast_axes=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=[5, 11],
        aux_ctc_loss_scales=[0.2, 0.8],
    )

    ckpt_list = [50, 100, 200, 300, 400, 450, 500]

    for peak_lr, init_lr in [(3e-4, 1e-5)]:
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(init_lr, peak_lr, 240))
            + list(np.linspace(peak_lr, init_lr, 240))
            + list(np.linspace(init_lr, 1e-7, 20)),
            #############
            "batch_size": 400 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 10.0,
            "torch_amp_options": {"dtype": "bfloat16"},
            "num_workers_per_gpu": 2,
            "log_grad_norm": True,
            "cleanup_old_models": {"keep": ckpt_list},
        }

        network_module = "phmm.phmm_zhou"
        train_args_radam = {
            "config": train_config_24gbgpu_amp_radam,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "train_step_args": {
                "fsa_exporter_config_path": fsa_exporter_config,
                "label_smoothing_scale": 0.1,
            },
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = (
            prefix_name
            + "/"
            + str(BPE_SIZE)
            + "/"
            + network_module
            + f".512dim_sub4_50eps_sp_lp_fullspec_gradnorm_radam_lr{peak_lr:.0e}"
        )
        train_job = training(training_name, train_data_bpe, train_args_radam, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48

        for epoch in ckpt_list:
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args_radam,
                with_prior=False,
                datasets=train_data_bpe,
                get_specific_checkpoint=epoch,
            )
            greedy_search_helper(training_name + f"/greedy_ep{epoch}", asr_model, default_greedy_config)
