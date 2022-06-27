import copy

from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob

from .pipeline import build_training_dataset, tts_training
from .config import get_training_config


baseline_network_options = {

    # audio options related to the network
    "frame_reduction_factor": 3,
    "feature_size": 80,

    # encoder options
    "embedding_dim": 256,
    "encoder_conv_dims": [256, 256, 256],
    "encoder_conv_filter_sizes": [5, 5, 5],
    "encoder_lstm_type": "zoneoutlstm",
    "encoder_lstm_dim": 256,
    "encoder_dropout": 0.5,
    "encoder_regularization": "batch_norm_new",

    "encoder_position_dim": 64,
    "encoder_lstm_dropout": 0.1,
    "encoder_lstm_dropout_broadcasting": True,

    # attention options
    "attention_dim": 128,
    "num_location_filters": 32,
    "location_filter_size": 31,
    "attention_in_dropout": 0.5,
    "location_feedback_dropout": 0.1,

    #decoder options
    "decoder_dim": 768,
    "decoder_type": "zoneout",
    "zoneout": 0.1,
    "decoder_dropout": 0.2,

    #pre-net options
    "prenet_layer1_dim": 128,
    "prenet_layer2_dim": 64,
    "prenet_layer1_dropout": 0.5,
    "prenet_layer2_dropout": 0.5,


    #post-net options
    "post_conv_dims": [256, 256, 256, 256, 256],
    "post_conv_filter_sizes": [5, 5, 5, 5, 5],
    "post_conv_dropout": 0.5,

    # decoding options
    "decoding_stop_threshold": 0.4,
    "decoding_additional_steps": 5,

    "stop_token_ramp_length": 5,
    "max_decoder_seq_len": 1000,

    # loss options
    "target_loss": "mean_l1",
    "l2_norm": 1e-7,
}


def run_tacotron2_aligner_training():

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_data = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="37ba06ab2697e7af4de96037565fdf4f78acdb80").out_repository
    returnn_root_training = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="fbeca88980962ad47e2622a3be517c2a43b2b530").out_repository

    prefix_path = "experiments/alignment_analysis_tts/tacotron2_aligner/baseline/"

    train_dataset, cv_dataset, extern_data = build_training_dataset(returnn_root=returnn_root_data, returnn_cpu_exe=returnn_exe, alias_path=prefix_path)

    network_options = copy.deepcopy(baseline_network_options)

    config = get_training_config(
        network_options=network_options,
        train_dataset=train_dataset,
        cv_dataset=cv_dataset,
        extern_data=extern_data
    )
    tts_training(returnn_config=config, num_epochs=200, returnn_gpu_exe=returnn_exe, returnn_root=returnn_root_training, output_path=prefix_path)



