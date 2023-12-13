from sisyphus import tk
import numpy as np

import copy
from dataclasses import asdict
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search, get_best_checkpoint

from .config import get_training_config, get_search_config


def conformer_v2_bpe():
    BPE_SIZE = 300
    prefix_name = "experiments/librispeech/librispeech_100_bpe_ctc/standalone_pt_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings,
        use_v2_subnmt=True,
    )

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None,
            use_v2_subnmt=True,
        )

        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
    from typing import cast

    label_datastream = cast(LabelDatastream, train_data.datastreams["bpe_labels"])
    vocab_size_without_blank = label_datastream.vocab_size
    
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    from i6_experiments.users.rossenbach.lexicon.bpe_lexicon import CreateBPELexiconJob
    from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
    ls_lexicon = get_bliss_lexicon(use_stress_marker=False, add_unknown_phoneme_and_mapping=True)
    from i6_experiments.common.helpers.text_labels.subword_nmt_bpe import get_returnn_subword_nmt

    bpe_lexicon = CreateBPELexiconJob(
        ls_lexicon,
        bpe_codes=train_data.datastreams["bpe_labels"].codes,
        bpe_vocab=train_data.datastreams["bpe_labels"].vocab,
        subword_nmt_repo=get_returnn_subword_nmt(),
    ).out_lexicon
    bpe_lexicon = BlissLexiconToWordLexicon(bpe_lexicon).out_lexicon

    config = {

    }

    def run_exp(ft_name, datasets, train_args, search_args=None, best=False, last=True, speed_perturbation=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, speed_perturbation=speed_perturbation,
                                             **train_args)
        if speed_perturbation:
            from i6_core.returnn.config import CodeWrapper
            returnn_config.config["train"]["datasets"]["zip_dataset"]["audio"]["pre_process"] = CodeWrapper(
                "speed_perturbation")
        returnn_search_config = get_search_config(**train_args, search_args=search_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=300)

        # averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        best_checkpoint = get_best_checkpoint(train_job)

        if last:
            search(ft_name + "/default_300", returnn_search_config, train_job.out_checkpoints[300], test_dataset_tuples,
                   RETURNN_EXE, MINI_RETURNN_ROOT)
        if best:
            search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE,
                   MINI_RETURNN_ROOT)
        # search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

    from .pytorch_networks.ctc_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
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
        out_features=384,
        activation=None,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=9,
        final_dropout=0.2,
    )

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_02 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
    }

    # # did not converge with epsilon 1e-8
    # train_args = {
    #     **train_args_adamw_02,
    #     "network_module": "ctc_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_transparent",
    #     "debug": True,
    #     "net_args": {
    #         "model_config_dict": asdict(model_config),
    #     },
    #     "with_devtrain": True,
    # }

    # for lm_weight in [1.0, 1.2, 1.4, 1.6, 1.8]:
    #     search_args = {
    #         "lexicon": bpe_lexicon,
    #         "beam_size": 256,
    #         "beam_threshold": 14,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": lm_weight,
    #         "returnn_vocab": label_datastream.vocab,
    #     }
    #     run_exp(
    #         prefix_name + "/bpe_conf_v2/i6modelsV1_VGG4LayerActFrontendV1_transparent/sub4_base_lm%.1f" % lm_weight,
    #         datasets=train_data,
    #         train_args=train_args, search_args=search_args)
        
    # did not converge with epsilon 1e-8
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_transparent_posenc",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
        "with_devtrain": True,
    }

    for lm_weight in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        search_args = {
            "lexicon": bpe_lexicon,
            "beam_size": 256,
            "beam_threshold": 14,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
            "returnn_vocab": label_datastream.vocab,
        }
        run_exp(
            prefix_name + "/bpe_conf_v2/i6modelsV1_VGG4LayerActFrontendV1_transparent_posenc/sub4_base_lm%.1f" % lm_weight,
            datasets=train_data,
            train_args=train_args, search_args=search_args)