from sisyphus import tk

from dataclasses import asdict

import copy
import numpy as np
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config

def conformer_baseline():
    prefix_name = "experiments/librispeech/librispeech_100_phon_ctc/standalone_normal_2023/"

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
        settings=train_settings
    )

    # build testing datasets
    test_dataset_tuples = {}
    #for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
        )


        # ---------------------------------------------------------------------------------------------------------------- #


    from .data import get_lexicon
    lexicon = get_lexicon(with_g2p=False)
    from i6_experiments.users.rossenbach.lexicon.conversion import BlissLexiconToWordLexicon
    word_lexicon = BlissLexiconToWordLexicon(lexicon).out_lexicon

    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

    def run_exp(ft_name, datasets, train_args, search_args=None, with_prior=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=300)

        if with_prior:
            returnn_config = get_prior_config(training_datasets=datasets, **train_args)
            prior_file = compute_prior(
                ft_name,
                returnn_config,
                checkpoint=train_job.out_checkpoints[300],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(ft_name + "/prior.txt", prior_file)
            search_args["prior_file"] = prior_file
        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        #best_checkpoint = get_best_checkpoint(train_job)

        returnn_search_config = get_search_config(**train_args, search_args=search_args)

        search(ft_name + "/default_300", returnn_search_config, train_job.out_checkpoints[300], test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)
        #search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    from .pytorch_networks.ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit_cfg import \
        SpecaugConfig, TwoLayer1DFrontendConfig, ModelConfig

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=50,
        max_dim_time=20,
        max_dim_feat=8,
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
        conformer_size=512,
        num_layers=8,
        num_heads=4,
        ff_dim=512,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
    )

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_02 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 200 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }

    # normal conv with 31 but with transparent attention
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [2, 3, 5]:
        search_args = {
            "lexicon": word_lexicon,
            "beam_size": 64,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
            "beam_threshold": 50,
        }
        run_exp(
            prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/baseline_bs64_lm%.1f" % lm_weight,
            datasets=train_data,
            train_args=train_args, search_args=search_args)
        
    for lm_weight in [2.0, 2.5, 3.0, 3.5, 4.0]:
        search_args = {
            "lexicon": word_lexicon,
            "beam_size": 128,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
            "beam_threshold": 50,
        }
        run_exp(
            prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/baseline_bs128_lm%.1f" % lm_weight,
            datasets=train_data,
            train_args=train_args, search_args=search_args)

    # PRIOR test
    for lm_weight in [1.5, 2.0, 2.5, 3.0]:
    # for lm_weight in [3.0]:
        for prior_scale in [0.1, 0.2, 0.3, 0.4, 0.5]:
        # for prior_scale in [0.3]:
            search_args = {
                "lexicon": word_lexicon,
                "beam_size": 128,
                "arpa_lm": get_arpa_lm_dict()["4gram"],
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_threshold": 50,
            }
            run_exp(
                prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/priortest/baseline_bs128_lm%.1f_prior%.1f" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)
        
    # for word_score in [0.1, 0.2, 0.3, 0.5, 1.0]:
    #     search_args = {
    #         "lexicon": word_lexicon,
    #         "beam_size": 128,
    #         "arpa_lm": get_arpa_lm_dict()["4gram"],
    #         "lm_weight": 3.0,
    #         "word_score": -word_score
    #     }
    #     run_exp(
    #         prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit.py/baseline_bs128_lm3.0_word_score%.1f" % word_score,
    #         datasets=train_data,
    #         train_args=train_args, search_args=search_args)
        
        
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=50,
        max_dim_time=20,
        max_dim_feat=8,
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

    # normal conv with 31 but with transparent attention
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        search_args = {
            "lexicon": word_lexicon,
            "beam_size": 128,
            "arpa_lm": get_arpa_lm_dict()["4gram"],
            "lm_weight": lm_weight,
            "beam_threshold": 50,
        }
        run_exp(
            prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_12x384_xavierinit.py/baseline_bs128_lm%.1f" % lm_weight,
            datasets=train_data,
            train_args=train_args, search_args=search_args)
        
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                "lexicon": word_lexicon,
                "beam_size": 128,
                "arpa_lm": get_arpa_lm_dict()["4gram"],
                "lm_weight": lm_weight,
                "beam_threshold": 50,
                "prior_scale": prior_scale,
                "sil_score": -99999999,
            }
            run_exp(
                prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_12x384_xavierinit.py/baseline_bs128_lm%.1f_prior0.3_silinf" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)


    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,
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

    # specaugment fix
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                "lexicon": word_lexicon,
                "beam_size": 128,
                "arpa_lm": get_arpa_lm_dict()["4gram"],
                "lm_weight": lm_weight,
                "beam_threshold": 50,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_12x384_xavierinit_specfix/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)

    ######################

    model_config_spec = copy.deepcopy(model_config)
    model_config_spec.specaug_config.max_dim_feat = 16

    default_search_args = {
        "lexicon": word_lexicon,
        "beam_size": 128,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "beam_threshold": 50,
    }

    # stronger specaug
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transparent_i6modelsV1_2x1D_frontend_xavierinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                "lexicon": word_lexicon,
                "beam_size": 128,
                "arpa_lm": get_arpa_lm_dict()["4gram"],
                "lm_weight": lm_weight,
                "beam_threshold": 50,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_12x384_xavierinit_specfixv2/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
    train_args_adameps = copy.deepcopy(train_args)
    train_args_adameps["config"]["optimizer"]["epsilon"] = 1e-16
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transparent_i6modelsV1_2x1D_frontend_12x384_xavierinit_specfixv2_eps16/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args_adameps, search_args=search_args, with_prior=True)

    # Removing transparent attention
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_normal_i6modelsV1_2x1D_frontend_xavierinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_normal_i6modelsV1_2x1D_frontend_12x384_xavierinit_specfixv2/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    # same with grad accum
    train_args = {
        **copy.deepcopy(train_args_adamw_02),
        "network_module": "ctc_conformer_0923.conformer_normal_i6modelsV1_2x1D_frontend_xavierinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    train_args["config"]["accum_grad_multiple_step"] = 2
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_normal_i6modelsV1_2x1D_frontend_12x384_xavierinit_specfixv2_accum2/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)

    # New specaugment and jingjing relmhsa
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transp_JJrelv1_2x1D_frontend_defaultinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transp_JJrelv1_2x1D_frontend_defaultinit_specfix/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)
    
    # New specaugment and ESPNet relmhsa
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)
            
    # New specaugment and ESPNet relmhsa
    # with strong grad accum
    train_args = {
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    train_args["config"]["accum_grad_multiple_step"] = 4
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix_accum4/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)
            
    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_03 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(1e-5, 1e-3, 150)) + list(np.linspace(1e-3, 1e-6, 150)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    # New specaugment and ESPNet relmhsa
    # with strong grad accum
    train_args = {
        **train_args_adamw_03,
        "network_module": "ctc_conformer_0923.conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    train_args["config"]["accum_grad_multiple_step"] = 3
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix_bs300s_accum3_adam03/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)
            
    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw_03 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(5e-6, 5e-4, 150)) + list(np.linspace(5e-4, 5e-7, 150)),
            #############
            "batch_size": 300 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
        },
    }
    # New specaugment and ESPNet relmhsa
    # with strong grad accum
    train_args = {
        **train_args_adamw_03,
        "network_module": "ctc_conformer_0923.conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_spec),
        },
    }
    train_args["config"]["accum_grad_multiple_step"] = 3
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "conformer_transp_espnetmhsa_2x1D_frontend_defaultinit_specfix_bs300s_accum3_adam03_lowlr/baseline_bs128_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data,
                train_args=train_args, search_args=search_args, with_prior=True)
            
            
            
    # new setup based on ted-lium experience

    prune_search_args = {
        "lexicon": word_lexicon,
        "beam_size": 1024,
        "arpa_lm": get_arpa_lm_dict()["4gram"],
        "beam_threshold": 16,
    }

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
        **train_args_adamw_02,
        "network_module": "ctc_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **prune_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "i6modelsV1_VGG4LayerActFrontendV1/baseline_bs1024_prune16_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)
            
            
    train_args = {
        **copy.deepcopy(train_args_adamw_02),
        "network_module": "ctc_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }
    train_args["config"]["accum_grad_multiple_step"] = 2
    for lm_weight in [2.5, 3.0, 3.5]:
        for prior_scale in [0.3]:
            search_args = {
                **prune_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
            }
            run_exp(
                prefix_name + "i6modelsV1_VGG4LayerActFrontendV1_accum2/baseline_bs1024_prune16_lm%.1f_prior0.3" % lm_weight,
                datasets=train_data, train_args=train_args, search_args=search_args, with_prior=True)