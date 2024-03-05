import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.users.rossenbach.experiments.jaist_project.data.aligner import build_training_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_prior_config, get_forward_config
from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, extract_durations
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_tts_log_mel_datastream

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import add_duration



def get_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    samples_per_frame = int(16000*0.0125)
    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "gradient_clip": 1.0,
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_ctc",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        "batch_size": 56000*samples_per_frame,
        "max_seq_length": {"audio_features": 1600*samples_per_frame},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/jaist_project/standalone_2024/nar_tts/ctc_aligner/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-100", partition_epoch=1)

    def run_exp(name, params, net_module, config, use_custom_engine=False, debug=False, v2=False, num_epochs=100, with_prior=0.0):
        aligner_config = get_training_config(
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
        )  # implicit reconstruction loss
        train_job = training(
            returnn_config=aligner_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix_name=prefix + name,
            num_epochs=num_epochs,
        )

        if with_prior > 0:
            prior_config = get_prior_config(
                training_dataset=training_datasets,
                network_module=net_module,
                net_args=params,
                debug=debug,
            )
            prior = compute_prior(
                prefix_name=prefix + name,
                returnn_config=prior_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            forward_config = get_pt_raw_forward_config_v2(
                forward_dataset=training_datasets.joint,
                network_module=net_module,
                net_args=params,
                init_args={"prior_file": prior, "prior_scale": with_prior},
                debug=debug,
            )
            duration_hdf = ctc_forward(
                checkpoint=train_job.out_checkpoints[num_epochs],
                config=forward_config,
                returnn_exe=RETURNN_PYTORCH_EXE,
                returnn_root=MINI_RETURNN_ROOT,
                prefix=prefix + name
            )
        else:

            forward_config = get_forward_config(
                network_module=net_module,
                net_args=params,
                decoder=net_module,
                decoder_args={},
                config={
                    "forward": training_datasets.joint.as_returnn_opts()
                },
                debug=debug,
            )
            duration_hdf =extract_durations(
                prefix_name=prefix + name,
                returnn_config=forward_config,
                checkpoint=train_job.out_checkpoints[num_epochs],
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            tk.register_output(prefix + name + "/duration.hdf", duration_hdf)
        return train_job, duration_hdf

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ..pytorch_networks.ctc.tts_aligner_1223.ctc_aligner_tts_fe_v7 import DbMelFeatureExtractionConfig, TTSPredictorConfig, Config
    assert isinstance(log_mel_datastream.options.feature_options, DBMelFilterbankOptions)
    fe_config = DbMelFeatureExtractionConfig(
        sample_rate=log_mel_datastream.options.sample_rate,
        win_size=log_mel_datastream.options.window_len,
        hop_size=log_mel_datastream.options.step_len,
        f_min=log_mel_datastream.options.feature_options.fmin,
        f_max=log_mel_datastream.options.feature_options.fmax,
        min_amp=log_mel_datastream.options.feature_options.min_amp,
        num_filters=log_mel_datastream.options.num_feature_filters,
        center=log_mel_datastream.options.feature_options.center,
        norm=norm
    )

    net_module = "ctc.tts_aligner_1223.ctc_aligner_tts_fe_v8"

    model_config = Config(
        conv_hidden_size=256,
        lstm_size=512,
        speaker_embedding_size=256,
        dropout=0.35,
        final_dropout=0.35,
        tts_loss_from_epoch=3,
        target_size=training_datasets.datastreams["phonemes"].vocab_size,
        feature_extraction_config=fe_config,
        tts_predictor_config=TTSPredictorConfig(
            hidden_dim=512,
            speaker_embedding_size=256,
            dropout=0.0,
        )
    )

    params = {
        "config": asdict(model_config)
    }
    local_config = copy.deepcopy(config)
    local_config["optimizer"] = {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-4}
    local_config["accum_grad_multiple_step"] = 2
    local_config["batch_size"] = 28000 * samples_per_frame
    train, duration_hdf = run_exp(net_module + "_tfstyle_v2", params, net_module, local_config, debug=True,
                           v2=True)
    # train.hold()

    add_duration(net_module + "_tfstyle_v2", duration_hdf)
    
    
    local_config = copy.deepcopy(config)
    local_config["optimizer"] = {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-4}
    local_config["accum_grad_multiple_step"] = 2
    local_config["max_seq_length"] = {"audio_features": 3000 * samples_per_frame}
    # local_config["torch_amp_options"] = {"dtype": "bfloat16"}
    local_config["batch_size"] = 28000 * samples_per_frame
    train, duration_hdf = run_exp(net_module + "_tfstyle_v2_fulllength", params, net_module, local_config, debug=True,
                           v2=True)
    # train.hold()

    add_duration(net_module + "_tfstyle_v2_fullength", duration_hdf)
    
    #duration_hdf = run_exp(net_module + "_tfstyle_v2_prior0.3", params, net_module, local_config, debug=True,
    #                       v2=True, with_prior=0.3)
    #add_duration(net_module + "_tfstyle_v2_prior0.3", duration_hdf)


     # local_config = copy.deepcopy(config)
     # local_config["optimizer"] = {"class": "adamw", "epsilon": 1e-8, "weight_decay": 1e-4}
     # local_config["accum_grad_multiple_step"] = 2
     # local_config["max_seq_length"] = {"audio_features": 3000 * samples_per_frame}
     # # local_config["torch_amp_options"] = {"dtype": "bfloat16"}
     # local_config["batch_size"] = 28000 * samples_per_frame
     # train, duration_hdf = run_exp(net_module + "_tfstyle_v2_fulllength", params, net_module, local_config, debug=True,
     #                        v2=True)
     # # train.hold()

     # add_duration(net_module + "_tfstyle_v2_fullength", duration_hdf)



    ##################################