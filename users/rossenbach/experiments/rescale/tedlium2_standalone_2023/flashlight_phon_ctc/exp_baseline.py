from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from .data import build_phon_training_datasets, TrainingDatasetSettings, get_eow_text_lexicon
from ..data import build_test_dataset
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config


def conformer_baseline():
    prefix_name = "experiments/rescale/tedliumv2/flashlight_phon_ctc/"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=5,
        epoch_wise_filters=[],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_phon_training_datasets(
        settings=train_settings
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev"]:
            test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
        )

    from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm
    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    arpa_ted_lm = lm.ngram_lm

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(ft_name, datasets, train_args, search_args=None, with_prior=False, num_epochs=250, decoder="ctc.decoder.flashlight_phoneme_ctc"):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(training_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file

        returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)

        _, _, search_jobs = search(ft_name + "/default_%i" % num_epochs, returnn_search_config, train_job.out_checkpoints[num_epochs], test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)

        return train_job, search_jobs

    from ..pytorch_networks.ctc.conformer_0923.transparent_i6modelsV1_2x1D_frontend_xavierinit_cfg import \
        SpecaugConfig, TwoLayer1DFrontendConfig, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=256,
        conv2_channels=384,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=2,
        conv2_stride=2,
        dropout=0.1,
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
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    
    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw03_accum2 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 125)) + list(np.linspace(1e-3, 1e-6, 125)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
    }

    default_search_args = {
        "lexicon": get_eow_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 64,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 50,
    }

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }

    for lm_weight in [1.5, 2.0, 2.5]:
        for prior_scale in [0.3, 0.5, 0.75, 1.0]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(prefix_name + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.2f" % (lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    for pruning in [10, 20, 30, 40, 50]:
        search_args = {
            **default_search_args,
            "lm_weight": 2.0,
            "prior_scale": 0.5,
        }
        search_args["beam_size"] = 256
        search_args["beam_threshold"] = pruning
        run_exp(prefix_name + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm2.0_prior0.5_bs256_prune%i" % pruning,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
        
    for pruning in [10, 12, 14, 16, 18, 20]:
        # 10 = 10.0
        # 12 = 9.9
        # 14 = 9.9
        # 16 = 9.8
        search_args = {
            **default_search_args,
            "lm_weight": 2.0,
            "prior_scale": 0.5,
        }
        search_args["beam_size"] = 1024
        search_args["beam_threshold"] = pruning
        run_exp(prefix_name + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm2.0_prior0.5_bs1024_prune%i" % pruning,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    # re-tune prior and lm-weight using beampruning 16
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.0, 0.3, 0.4, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(
                prefix_name + "conformer_0923/transparent_i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.1f_bs1024_prune16" % (lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    
    # Ted-Lium can be larger
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=512,
        conv2_channels=512,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=2,
        conv2_stride=2,
        dropout=0.1,
    )
    model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw03_accum2 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 125)) + list(np.linspace(1e-3, 1e-6, 125)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
    }

    default_search_args = {
        "lexicon": get_eow_text_lexicon(),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 64,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 50,
    }

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }

    for lm_weight in [1.5, 2.0, 2.5]:
        for prior_scale in [0.3, 0.5, 0.75, 1.0]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(prefix_name + "conformer_0923/transparent_12x512_i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.2f" % (
            lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            

    # same with AMP
    train_args_amp = copy.deepcopy(train_args)
    train_args_amp["config"]["torch_amp_options"] = {"dtype": "float16"}  # Pascal / 1080 GPUs can only do float16
    for lm_weight in [1.5, 2.0, 2.5]:
        for prior_scale in [0.3, 0.5, 0.75, 1.0]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(prefix_name + "conformer_0923/transparent_12x512_i6modelsV1_2x1D_frontend_xavierinit_amp/lm%.1f_prior%.2f" % (
            lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args_amp, search_args=search_args, with_prior=True)


    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_2x1D_frontend_xavierinit_cfg import \
        SpecaugConfig, TwoLayer1DFrontendConfig, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = TwoLayer1DFrontendConfig(
        in_features=80,
        conv1_channels=256,
        conv2_channels=384,
        conv1_kernel_size=5,
        conv2_kernel_size=5,
        conv1_stride=2,
        conv2_stride=2,
        dropout=0.1,
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
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    
    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_2x1D_frontend_xavierinit/lm%.1f_prior%.2f" % (
            lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["optimizer"] = {"class": "adam", "epsilon": 1e-16}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_2x1D_frontend_xavierinit_adam/lm%.1f_prior%.2f" % (
            lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
    

    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_cfg import \
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
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1/lm%.1f_prior%.2f" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
    
    
    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_posenc",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc/lm%.1f_prior%.2f" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_convfirst",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_convfirst/lm%.1f_prior%.2f" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_posenc_convfirst",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc_convfirst/lm%.1f_prior%.2f" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
    
    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_xavierinit/lm%.1f_prior%.2f" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_xavierinit/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 256
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_posenc_xavierinit/lm%.1f_prior%.2f" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
    
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
    ######################################################

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["optimizer"] = {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_decay-2/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["optimizer"] = {"class": "adamw", "epsilon": 1e-16, "weight_decay": 5e-3}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_decay5-3/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    #############################################


    # Train long basic
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(1e-5, 1e-3, 250)) + list(np.linspace(1e-3, 1e-6, 250))
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_ep500/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True, num_epochs=500)


    # Train long skewed
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(1e-5, 1e-3, 200)) + list(np.linspace(1e-3, 1e-7, 300))
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(
                prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_ep500skewed/lm%.1f_prior%.2f_bs1024" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True,
                num_epochs=500)

    bene_model_config = ModelConfig(
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=6,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=9,
        final_dropout=0.2,
    )
    
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(bene_model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_bene_param/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
            
    # No Subsampling
    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config_nosub = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(1, 1),
        pool1_stride=(1, 1),
        pool1_padding=None,
        pool2_kernel_size=(1, 1),
        pool2_stride=(1, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config_nosub = ModelConfig(
        frontend_config=frontend_config_nosub,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_nosub),
        },
    }
    train_args["config"]["batch_size"] = 150 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 4
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 16
            train_job, _ = run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_JJLR_nosub/lm%.1f_prior%.2f_bs1024" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            train_job.rqmt["gpu_mem"] = 24
            
            
            
    #### New experiments with corrected FF-Dim
    
    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_cfg import \
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
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v2",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 180 * 16000
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v2_JJLR/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 180 * 16000
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

            # beam search token
            if lm_weight == 2.0 and prior_scale == 0.5:
                for bst in [10, 20, 30, 40, 50]:
                    search_args = copy.deepcopy(search_args)
                    search_args["beam_size_token"] = bst
                    run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/lm%.1f_prior%.2f_bs1024_th14_bst_%i" % (
                        lm_weight, prior_scale, bst),
                            datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
                    if bst == 20:
                        run_exp(
                            prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/lm%.1f_prior%.2f_bs1024_th14_bst_%i_exp1" % (
                                lm_weight, prior_scale, bst),
                            datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True,
                            decoder="ctc.decoder.flashlight_experimental_phoneme_ctc"
                        )

    # Search GRID
    for lm_weight in [1.6, 1.8, 2.0, 2.2, 2.4]:  # 5
        for prior_scale in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:  # 5
            for beam_threshold in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:  # 12
                # for beam_size in [256, 1024, 4096, 8192]:  # 4
                for beam_size in [256, 1024]:  # 4
                    search_args = {
                        **copy.deepcopy(default_search_args),
                        "lm_weight": lm_weight,
                        "prior_scale": prior_scale,
                    }
                    search_args["beam_size"] = beam_size
                    search_args["beam_threshold"] = beam_threshold
                    search_args["node"] = "intel"
                    _, search_jobs = run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR/search_grid_intel_full/lm%.1f_prior%.2f_bs%i_th%i" % (
                        lm_weight, prior_scale, beam_size, beam_threshold),
                            datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
                    for search_job in search_jobs:
                        search_job.rqmt["sbatch_args"] = "-p rescale_intel -A rescale_speed"
                        if beam_size > 1024:
                            search_job.rqmt["mem"] = 12
                        elif beam_size > 4096:
                            search_job.rqmt["mem"] = 16

    # with speed perturbation
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
        "use_speed_perturbation": True
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 180 * 16000
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_speed/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
    
    
    
    from ..pytorch_networks.ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v4_cfg import \
        ModelConfig as ModelConfigV4

    model_config_v4 = ModelConfigV4(
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
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=1,
    )

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v5",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_v4),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 180 * 16000
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.5, 0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v5_JJLR/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)


            
    frontend_config_large = VGG4LayerActFrontendV1Config_mod(
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
    model_config_large = ModelConfig(
        frontend_config=frontend_config_large,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=4,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_large),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 100 * 16000
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum2/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                    datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_large),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 110)) + list(
        np.linspace(7e-4, 7e-5, 110)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 100 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 3
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(
                prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum3/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "ctc.conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_large),
        },
    }
    train_args["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 135)) + list(
        np.linspace(7e-4, 7e-5, 135)) + list(np.linspace(7e-5, 1e-8, 30))
    train_args["config"]["batch_size"] = 100 * 16000
    train_args["config"]["accum_grad_multiple_step"] = 4
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.5, 0.7]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            search_args["beam_size"] = 1024
            search_args["beam_threshold"] = 14
            run_exp(
                prefix_name + "conformer_0923/i6modelsV1_VGG4LayerActFrontendV1_v3_JJLR_large_accum4_300ep/lm%.1f_prior%.2f_bs1024_th14" % (
                    lm_weight, prior_scale),
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True, num_epochs=300)