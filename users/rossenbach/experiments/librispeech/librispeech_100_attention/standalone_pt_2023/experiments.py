from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from .data import build_training_datasets, TrainingDatasetSettings, build_test_dataset
from .default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from .pipeline import training, search

from .config import get_training_config, get_search_config

def conformer_baseline():
    BPE_SIZE = 2000
    prefix_name = "experiments/librispeech/librispeech_100_attention/standalone_pt_2023"

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None,
        partition_epoch=3,
        epoch_wise_filters=[(1, 5, 1000)],
        seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        "train-clean-100",
        bpe_size=BPE_SIZE,
        preemphasis=None,
        settings=train_settings
    )

    # build testing datasets
    test_dataset_tuples = {}
    #for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
    for testset in ["dev-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            librispeech_key="train-clean-100",
            dataset_key=testset,
            bpe_size=BPE_SIZE,
            preemphasis=None
        )


        # ---------------------------------------------------------------------------------------------------------------- #
    # local experiment function
    
    # local experiment function
    from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream
    from typing import cast

    label_datastream = cast(LabelDatastream, train_data.datastreams["bpe_labels"])
    vocab_size_without_blank = label_datastream.vocab_size


    config = {

    }

    def run_exp(ft_name, datasets, train_args, search_args=None, do_search=False):
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, **train_args)
        train_job = training(ft_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=250)

        #averaged_checkpoint = get_average_checkpoint(train_job, num_average=4)
        #best_checkpoint = get_best_checkpoint(train_job)

        if do_search:
            returnn_search_config = get_search_config(**train_args, search_args=search_args)
            search(ft_name + "/default_250", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples, RETURNN_EXE, MINI_RETURNN_ROOT)
        #search(ft_name + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        #search(ft_name + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_convfirst",
    #     "debug": True,
    #     "config": {},
    # }
    # run_exp(prefix_name + "/test", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_convfirst",
    #     "debug": True,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
    #         "batch_size": 30000 * 160,
    #     },

    # }
    # run_exp(prefix_name + "/test2", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_convfirst_specaug_lesszone",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
    #         "batch_size": 30000 * 160,
    #     },

    # }
    # run_exp(prefix_name + "/test3", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_convfirst_specaug",
    #     "debug": True,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
    #         "batch_size": 30000 * 160,
    #     },

    # }
    # run_exp(prefix_name + "/test4", datasets=train_data, train_args=train_args, do_search=True)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_fairseqinit_correctconvfirst_specaug_lesszone",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
    #         "batch_size": 30000 * 160,
    #     },

    # }
    # run_exp(prefix_name + "/test5", datasets=train_data, train_args=train_args)

    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
    #         "batch_size": 30000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/test6", datasets=train_data, train_args=train_args)
    #
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone_speedtest",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 30)) + list(np.linspace(8e-4, 1e-6, 220)),
    #         "batch_size": 15000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/speed_test", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone_speedtest",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 1e-4, 30)) + list(np.linspace(1e-4, 1e-6, 220)),
    #         "batch_size": 30000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/speed_test_v2", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_basic_newinit_correctconvfirst_specaug_lesszone_speedtest",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 8e-4, 50)) + list(np.linspace(8e-4, 1e-6, 200)),
    #         "batch_size": 30000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/speed_test_v3", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_somewhat_tflike_pretraintest",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 8e-4, 50)) + list(np.linspace(8e-4, 1e-6, 200)),
    #         "batch_size": 30000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/pretrain_test", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_somewhat_tflike_pretraintest",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-4, 8e-4, 50)) + list(np.linspace(8e-4, 1e-6, 200)),
    #         "batch_size": 30000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/pretrain_test_v2", datasets=train_data, train_args=train_args)
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_somewhat_tflike_pretraintest",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": ([1e-4] * 15) + list(np.linspace(1e-4, 8e-4, 35)) + list(np.linspace(8e-4, 1e-6, 200)),
    #         "batch_size": 30000 * 160,
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/pretrain_test_v3", datasets=train_data, train_args=train_args)


    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": ([1e-4] * 50) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
    #         "batch_size": 30000 * 160,
    #         "max_seq_length": {"audio_features": 3000 * 160},
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1", datasets=train_data, train_args=train_args)

    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
    #         "batch_size": 25000 * 160,
    #         "max_seq_length": {"audio_features": 3000 * 160},
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_morewarmup", datasets=train_data, train_args=train_args)

    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_pretrain",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": ([1e-4] * 50) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
    #         "batch_size": 30000 * 160,
    #         "max_seq_length": {"audio_features": 3000 * 160},
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_pretrain", datasets=train_data, train_args=train_args)
    #
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
    #         "batch_size": 25000 * 160,
    #         "max_seq_length": {"audio_features": 3000 * 160},
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout_morewarmup", datasets=train_data, train_args=train_args)


    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_noattdrop",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
    #         "batch_size": 25000 * 160,
    #         "max_seq_length": {"audio_features": 3000 * 160},
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_noattdrop_morewarmup", datasets=train_data, train_args=train_args)
    #
    #
    # train_args = {
    #     "net_args":{},
    #     "network_module": "attention_simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout",
    #     "debug": False,
    #     "config": {
    #         "learning_rates": list(np.linspace(1e-5, 1e-4, 50)) + list(np.linspace(1e-4, 8e-4, 100)) + list(np.linspace(8e-4, 1e-6, 100)),
    #         "batch_size": 25000 * 160,
    #         "max_seq_length": {"audio_features": 3000 * 160},
    #         "optimizer": {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-6},
    #     },
    #     "use_v3": True,
    # }
    # run_exp(prefix_name + "/simpleconv3x2_enc12x512_dec1024_simplespecv1_nozoneout_morewarmup_adamw", datasets=train_data, train_args=train_args)
    
    
    
    from .pytorch_networks.att_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_cfg import ModelConfig, VGG4LayerActFrontendV1Config_mod, SpecaugConfig
    from .pytorch_networks.att_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v2_cfg import ModelConfig as ModelConfigV2
    from .pytorch_networks.att_conformer_0923.i6modelsV1_VGG4LayerActFrontendV1_v3_cfg import ModelConfig as ModelConfigV3

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
        target_embed_dim=128,
        target_embed_dropout=0.1,
        lstm_hidden_size=512,
        zone_dropout_c=0.05,
        zone_dropout_h=0.15,
        attention_dim=256,
        additive_att_weights_dropout=0.1,
        out_proj_dim=512,
        output_dropout=0.2,
    )

    # from here on onwards, use default AdamW with same OCLR
    train_args_adamw03_accum2 = {
        "config": {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
            "learning_rates": list(np.linspace(8e-6, 8e-4, 125)) + list(np.linspace(8e-4, 8e-7, 125)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
        },
    }
    
    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier", datasets=train_data, train_args=train_args)

    frontend_config_sub6 = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config_sub6 = copy.deepcopy(model_config)
    model_config_sub6.frontend_config = frontend_config_sub6


    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6", datasets=train_data,
            train_args=train_args)
    
    model_config_sub6_larger = ModelConfig(
        frontend_config=frontend_config_sub6,
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
        target_embed_dim=256,
        target_embed_dropout=0.1,
        lstm_hidden_size=768,
        zone_dropout_c=0.05,
        zone_dropout_h=0.15,
        attention_dim=768,
        additive_att_weights_dropout=0.1,
        out_proj_dim=768,
        output_dropout=0.2,
    )


    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger", datasets=train_data,
            train_args=train_args)
    
    model_config_sub6_larger_v2 = ModelConfig(
        frontend_config=frontend_config_sub6,
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
        target_embed_dim=256,
        target_embed_dropout=0.1,
        lstm_hidden_size=768,
        zone_dropout_c=0.15,
        zone_dropout_h=0.05,
        attention_dim=768,
        additive_att_weights_dropout=0.0,
        out_proj_dim=768,
        output_dropout=0.2,
    )


    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger_v2),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger_v2", datasets=train_data,
            train_args=train_args)

    model_config_sub6_larger_ctc03 = ModelConfigV2(
        frontend_config=frontend_config_sub6,
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
        target_embed_dim=256,
        target_embed_dropout=0.1,
        lstm_hidden_size=768,
        zone_dropout_c=0.15,
        zone_dropout_h=0.05,
        attention_dim=768,
        additive_att_weights_dropout=0.0,
        out_proj_dim=768,
        output_dropout=0.2,
        ctc_loss_scale=0.3,
    )


    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_v2_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger_ctc03),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger_ctc03", datasets=train_data,
            train_args=train_args)
    
    train_args_lowlr = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_v2_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger_ctc03),
        },
    }
    train_args_lowlr["config"]["learning_rates"] = list(np.linspace(8e-6, 4e-4, 125)) + list(np.linspace(4e-4, 8e-7, 125))

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger_ctc03_lowlr", datasets=train_data,
            train_args=train_args_lowlr)
    
    
    model_config_sub6_larger_ctc03_nozone = copy.deepcopy(model_config_sub6_larger_ctc03)
    model_config_sub6_larger_ctc03_nozone.zone_dropout_h = 0.0
    model_config_sub6_larger_ctc03_nozone.zone_dropout_c = 0.0

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_v2_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger_ctc03_nozone),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger_ctc03_nozone", datasets=train_data,
            train_args=train_args)

    

    model_config_sub6_larger_ctc02 = copy.deepcopy(model_config_sub6_larger_ctc03)
    model_config_sub6_larger_ctc02.ctc_loss_scale = 0.2

    train_args = {
        **train_args_adamw03_accum2,
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_v2_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger_ctc02),
        },
    }

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger_ctc02", datasets=train_data,
            train_args=train_args)
    
    
    model_config_sub6_larger_ctc03_ctcdrop_lowzone = ModelConfigV3(
        frontend_config=frontend_config_sub6,
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
        target_embed_dim=256,
        target_embed_dropout=0.1,
        lstm_hidden_size=768,
        zone_dropout_c=0.05,
        zone_dropout_h=0.05,
        attention_dim=768,
        additive_att_weights_dropout=0.0,
        out_proj_dim=768,
        output_dropout=0.2,
        ctc_loss_scale=0.3,
        ctc_dropout=0.2,
    )

    train_args_lowlr = {
        **copy.deepcopy(train_args_adamw03_accum2),
        "network_module": "att_conformer_0923.i6modelsV1_VGG4LayerActV1_v3_transparent_posenc_xavierinit",
        "debug": True,
        "net_args": {
            "model_config_dict": asdict(model_config_sub6_larger_ctc03_ctcdrop_lowzone),
        },
    }
    train_args_lowlr["config"]["learning_rates"] = list(np.linspace(8e-6, 4e-4, 125)) + list(np.linspace(4e-4, 8e-7, 125))

    run_exp(prefix_name + "/conf_0923_i6modelsv1_vggV1_transparent_posenc_xavier_sub6_larger_ctc03_ctcdrop_lowzone_lowlr", datasets=train_data,
            train_args=train_args_lowlr)