"""
Posterior HMM baseline with NON-EOW (plain monophone) phonemes.

Exact copy of :func:`experiments.phmm_phon.baseline.eow_phon_phmm_ls960_base`, with the only
difference being the phoneme inventory: plain monophones instead of the EOW-augmented set. That
means the lexicon is built without ``AddEowPhonemesToLexiconJob`` (``get_phmm_phon_lexicon`) and
the training data uses the matching non-EOW vocab (``build_phon_phmm_training_datasets``). Network,
optimizer, schedule, SpecAugment, aux losses, multi-GPU setup, FSA topology and recognition are
all unchanged.
"""
import copy
from dataclasses import asdict
from sisyphus import tk
import numpy as np
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict

from ...data.phon import build_phon_phmm_training_datasets, get_phmm_phon_lexicon
from ...default_tools import RETURNN_EXE, RETURNN_ROOT, LIBRASR_WHEEL
from ...lm import get_4gram_lm_rasr_config
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...rasr import (
    AddSentenceBoundaryLemmataToPhmmLexiconJob,
    CreateLibrasrVenvJob,
    build_fsa_exporter_config,
    build_librasr_phmm_recognition_config,
)


def phon_phmm_ls960_base():
    prefix_name = "example_setups/librispeech/phmm_standalone_2024/ls960_phmm_phon"

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    train_data = build_phon_phmm_training_datasets(
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

    phmm_returnn_exe = CreateLibrasrVenvJob(
        python_exe=RETURNN_EXE,
        librasr_wheel=LIBRASR_WHEEL,
        # "ninja" for i6_native_ops JIT; "dm-tree"/"h5py" are real-RETURNN deps that MiniReturnn
        # did not require (dm-tree provides `tree`, imported by returnn.frontend at startup).
        extra_pip_packages=["ninja", "dm-tree", "h5py"],
        python_wrapper_name="python_with_path",
        track_wheel_contents=True,
    ).out_python_bin

    default_returnn = {
        "returnn_exe": phmm_returnn_exe,
        "returnn_root": RETURNN_ROOT,
    }

    phmm_lexicon = get_phmm_phon_lexicon(g2p_librispeech_key="train-other-960")
    tk.register_output(prefix_name + "/phmm_phon_lexicon.xml.gz", phmm_lexicon)
    phmm_recog_lexicon = AddSentenceBoundaryLemmataToPhmmLexiconJob(phmm_lexicon).out_lexicon
    tk.register_output(prefix_name + "/phmm_phon_recog_lexicon.xml.gz", phmm_recog_lexicon)
    librispeech_corpus = get_bliss_corpus_dict(audio_format="ogg")["train-other-960"]
    fsa_exporter_config = build_fsa_exporter_config(
        lexicon_path=phmm_lexicon,
        corpus_path=librispeech_corpus,
    )
    recog_rasr_config = build_librasr_phmm_recognition_config(
        lexicon_path=phmm_recog_lexicon,
        lm_config=get_4gram_lm_rasr_config(lexicon_file=phmm_recog_lexicon, scale=1.0),
        logfile_suffix="phmm_phon_recog",
    )

    from ...pytorch_networks.phmm.decoder.rasr_phmm_v1 import DecoderConfig as RasrDecoderConfig

    def librasr_search_helper(training_name: str, asr_model: ASRModel, decoder_config: RasrDecoderConfig):
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_librasr"
        search_jobs, wers = search(
            search_name,
            forward_config={"num_workers_per_gpu": 0},
            asr_model=asr_model,
            decoder_module="phmm.decoder.rasr_phmm_v1",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            **default_returnn,
        )

    default_rasr_decoder_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        lexicon=phmm_recog_lexicon,
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
        specauc_start_epoch=3,  # subepoch units: 11/4 (rounded) to track the /4 schedule below
        dropout_broadcast_axes=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=[5, 11],
        aux_ctc_loss_scales=[0.2, 0.8],
    )

    # --- Multi-GPU data-parallel training (new default) ---------------------------------------
    # Single-node, `num_gpus` processes, RETURNN `torch_distributed` parameter-averaging (no DDP
    # wrapping -> the custom RASR-FSA train step runs unchanged). With the default OggZip pipeline
    # RETURNN uses the `random_seed_offset` data distribution (sharding needs DistributeFilesDataset),
    # so each worker sees a different draw of the full data. The LR schedule / num_epochs / ckpt_list
    # below are the single-GPU values divided by `num_gpus`, keeping total data/compute ~constant while
    # cutting wall-clock ~num_gpus x.
    #
    # gpu_mem is a Sisyphus job *requirement* (NOT part of the hash). batch_size is kept at the small
    # 24GB-tuned value and is hashed -> the GPU memory tier can be switched freely (just change gpu_mem)
    # without rehashing/restarting the training.
    num_gpus = 4
    gpu_mem = 24

    ckpt_list = [10, 20, 40, 60, 80, 100, 110, 125]  # checkpoints kept + evaluated (all of them)

    for peak_lr, init_lr in [(3e-4, 1e-5)]:
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            # schedule = single-GPU 240/240/20 (=500) divided by num_gpus=4 -> 60/60/5 (=125 subepochs).
            # peak_lr is NOT scaled: torch_distributed param-averaging is not a true large-batch step.
            "learning_rates": list(np.linspace(init_lr, peak_lr, 60))
            + list(np.linspace(peak_lr, init_lr, 60))
            + list(np.linspace(init_lr, 1e-7, 5)),
            # Single-node multi-GPU via parameter averaging every 100 steps (i6-recommended mode).
            "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},
            #############
            # extern_data is required by real RETURNN. "labels" maps to the OggZip "orth" stream,
            # i.e. the raw UTF-8 bytes of the orthography (sparse dim 256, uint8); the train step
            # decodes these bytes back to text for the RASR FSA builder.
            "behavior_version": 21,
            "extern_data": {
                "raw_audio": {"dim": 1},
                "labels": {"dim": 256, "sparse": True, "dtype": "uint8"},
            },
            "batch_size": 400 * 16000,
            "max_seq_length": {"raw_audio": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_global_norm": 10.0,
            "torch_amp": {"dtype": "bfloat16"},
            "torch_dataloader_opts": {"num_workers": 2},
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
            + "/phon/"
            + network_module
            + f".512dim_sub4_50eps_sp_lp_fullspec_gradnorm_radam_lr{peak_lr:.0e}"
        )
        train_job = training(
            training_name,
            train_data,
            train_args_radam,
            num_epochs=125,  # single-GPU 500 / num_gpus=4
            num_processes=num_gpus,
            distributed_launch_cmd="torchrun",
            **default_returnn,
        )
        # gpu_mem is a (non-hashed) requirement -> switch GPU tier here without rehashing.
        # i6_core already scales cpu (6->24) / gpu (1->4) / mem (24->96 GB) by num_gpus.
        train_job.rqmt["gpu_mem"] = gpu_mem

        asr_models_by_epoch = {}
        for epoch in ckpt_list:
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args_radam,
                with_prior=False,
                datasets=train_data,
                get_specific_checkpoint=epoch,
            )
            asr_models_by_epoch[epoch] = asr_model
            librasr_search_helper(training_name + f"/recog_ep{epoch}", asr_model, default_rasr_decoder_config)

    return {
        "prefix_name": prefix_name,
        "training_name": training_name,
        "train_job": train_job,
        "train_args": train_args_radam,
        "train_data": train_data,
        "phmm_lexicon": phmm_lexicon,
        "dev_dataset_tuples": dev_dataset_tuples,
        "test_dataset_tuples": test_dataset_tuples,
        "asr_models_by_epoch": asr_models_by_epoch,
        "default_returnn": default_returnn,
        "vocab_size_without_blank": vocab_size_without_blank,
    }
