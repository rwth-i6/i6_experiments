import copy
from functools import lru_cache
import numpy

from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.returnn.config import ReturnnConfig

from i6_experiments.common.datasets.librispeech import get_ogg_zip_dict

from i6_experiments.users.rossenbach.datasets.librispeech import get_librispeech_bpe
from i6_experiments.users.rossenbach.setups import returnn_standalone
from i6_experiments.users.rossenbach.returnn.config import ExtendedReturnnConfig

from .specaugment_clean import SpecAugmentSettings



@lru_cache()
def get_audio_datastream(statistics_ogg_zip, returnn_python_exe, returnn_root, output_path):
    # default: mfcc-40-dim
    extract_audio_opts = returnn_standalone.data.audio.AudioFeatureDatastream(
        available_for_inference=True,
        window_len=0.025,
        step_len=0.010,
        num_feature_filters=40,
        features="mfcc")

    audio_datastream = returnn_standalone.data.audio.add_global_statistics_to_audio_features(
        extract_audio_opts, statistics_ogg_zip,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=output_path,
    )
    return audio_datastream

@lru_cache()
def get_bpe_datastream(bpe_size, is_recog):
    """

    :param bpe_size:
    :param is_recog:
    :return:
    """
    # build dataset
    bpe_settings = get_librispeech_bpe(corpus_key="train-clean-100", bpe_size=bpe_size, unk_label='<unk>')
    bpe_targets = returnn_standalone.data.vocabulary.BpeDatastream(
        available_for_inference=False,
        bpe_settings=bpe_settings,
        use_unk_label=is_recog
    )
    return bpe_targets


@lru_cache()
def build_training_datasets(returnn_python_exe, returnn_root, output_path):
    bpe_size=2000

    ogg_zip_dict = get_ogg_zip_dict("corpora")
    train_clean_100_ogg = ogg_zip_dict['train-clean-100']
    dev_clean_ogg = ogg_zip_dict['dev-clean']
    dev_other_ogg = ogg_zip_dict['dev-other']

    train_bpe_datastream = get_bpe_datastream(bpe_size=bpe_size, is_recog=False)

    audio_datastream = get_audio_datastream(
        statistics_ogg_zip=train_clean_100_ogg,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_path=output_path,
    )

    extern_data = {
        'audio_features': audio_datastream.as_returnn_data_opts(),
        'bpe_labels': train_bpe_datastream.as_returnn_data_opts()
    }

    data_map = {"audio_features": ("zip_dataset", "data"),
                "bpe_labels": ("zip_dataset", "classes")}

    train_zip_dataset = returnn_standalone.data.datasets.OggZipDataset(
        path=train_clean_100_ogg,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        partition_epoch=3,
        seq_ordering="laplace:.1000",
        other_opts={"epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}}  # still hardcoded, future work
    )
    train_dataset = returnn_standalone.data.datasets.MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": train_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    from i6_experiments.users.rossenbach.datasets.librispeech import get_mixed_cv_segments
    cv_zip_dataset = returnn_standalone.data.datasets.OggZipDataset(
        path=[dev_clean_ogg, dev_other_ogg],
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse"
    )
    cv_dataset = returnn_standalone.data.datasets.MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": cv_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    devtrain_zip_dataset = returnn_standalone.data.datasets.OggZipDataset(
        path=train_clean_100_ogg,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=train_bpe_datastream.as_returnn_targets_opts(),
        segment_file=get_mixed_cv_segments(),
        seq_ordering="sorted_reverse",
        subset=3000,
    )
    devtrain_dataset = returnn_standalone.data.datasets.MetaDataset(
        data_map=data_map,
        datasets={"zip_dataset": devtrain_zip_dataset},
        seq_order_control_dataset="zip_dataset"
    )

    return train_dataset, cv_dataset, devtrain_dataset, extern_data



def trainig_network_mohammad(specaug_settings=None, stronger_pt_reg=False):
    """
    Network derived from Mohammads Librispeech-100h system with new encoder pre-training

    :return:
    """
    from .prototype_network import EncoderWrapper, static_decoder

    specaug_settings_full = SpecAugmentSettings()


    specaug_settings_light = SpecAugmentSettings(
        min_frame_masks=0,
        mask_each_n_frames=100,
        max_frames_per_mask=10,
        min_feature_masks=0,
        max_feature_masks=2,
        max_features_per_mask=4,
    )

    stage_nets = []
    for i in range(6):
        # pooling pretraing

        network_stage = 0 if i == 0 else i - 1
        if network_stage == 0:
            lstm_pool_sizes = [6]
        else:
            lstm_pool_sizes = [3, 2]
        # dropout from 0 to 0.3
        if stronger_pt_reg:
            lstm_dropout = 0.15 + (i/5.0 * 0.15)
            l2 = 0.0005 + (i/5.0 * 0.0005)
        else:
            lstm_dropout = (i/5.0 * 0.3)
            l2 = (i/5.0 * 0.001)
        # grow lstm dim from 512 to 1024
        lstm_dim = int(512 + (network_stage/4.0 * 512))
        #enable specaugment after epoch 10

        encoder = EncoderWrapper(audio_feature_key="audio_features", target_label_key="bpe_labels",
                                 num_lstm_layers=2 + network_stage, lstm_pool_sizes=lstm_pool_sizes, lstm_dropout=lstm_dropout,
                                 lstm_single_dim=lstm_dim, l2=l2,
                                 specaugment_settings=specaug_settings_light if i < (1 if stronger_pt_reg else 3) else specaug_settings_full)
        encoder_dict = encoder.make_root_net_dict()
        # Eval layer hack for specaugment
        encoder_dict['encoder']['subnetwork']['specaug_block']['subnetwork']['eval_layer'].pop("kind")

        local_static_decoder = copy.deepcopy(static_decoder)
        if i < 5:
            local_static_decoder["output"]["unit"]["output_prob"]["loss_opts"]["label_smoothing"] = 0

        stage_net = {**encoder_dict, **local_static_decoder, "#copy_param_mode": "subset"}
        if i < 5:
            stage_net["#config"] = {}
            stage_net["#config"]["batch_size"] = 15000
        stage_nets.append(stage_net)

    network_dict = {
        1: stage_nets[0], # network 0
        11: stage_nets[1], # still network 0, only specaug
        26: stage_nets[2], # network 1
        31: stage_nets[3], # network 2
        36: stage_nets[4], # network 3
        41: stage_nets[5], # network 4
    }
    return network_dict


def get_config(with_pretraining=True, **kwargs):

    # changing these does not change the hash
    post_config = {
        'use_tensorflow': True,
        'tf_log_memory_usage': True,
        'cleanup_old_models': True,
        'log_batch_size': True,
        'debug_print_layer_output_template': True,
    }

    wup_start_lr = 0.0003
    initial_lr = 0.0008

    learning_rates = [wup_start_lr] * 10 + list(numpy.linspace(wup_start_lr, initial_lr, num=10))

    config = {
        'gradient_clip': 0,
        'optimizer': {'class': 'Adam', 'epsilon': 1e-8},
        'accum_grad_multiple_step': 2,
        'gradient_noise': 0.0,
        'learning_rates': learning_rates,
        'min_learning_rate': 0.00001,
        'learning_rate_control': "newbob_multi_epoch",
        'learning_rate_control_relative_error_relative_lr': True,
        'learning_rate_control_min_num_epochs_per_new_lr': 3,
        'use_learning_rate_control_always': True,
        'newbob_multi_num_epochs': 3,
        'newbob_multi_update_interval': 1,
        'newbob_learning_rate_decay': 0.9,
        'batch_size': 10000,
        'max_seqs': 200,
        # 'truncation': -1
    }


    network_dict = trainig_network_mohammad(specaug_settings=kwargs.get("specaug_settings", None),
                                            stronger_pt_reg=kwargs.get("stronger_pt_reg", False))

    from .specaugment_clean import get_funcs

    if with_pretraining:
        returnn_config = ReturnnConfig(
            config=config,
            post_config=post_config,
            staged_network_dict=network_dict,
            python_prolog=get_funcs(),
            hash_full_python_code=True,
        )
    else:
        returnn_config = ExtendedReturnnConfig()
    return returnn_config


def training(prefix_name, returnn_exe, returnn_root, **kwargs):
    returnn_config = get_config(**kwargs)
    train_dataset, cv_dataset, devtrain_dataset, extern_data = build_training_datasets(
        returnn_python_exe=returnn_exe, returnn_root=returnn_root, output_path=prefix_name)

    returnn_config.config["extern_data"] = extern_data
    returnn_config.config["train"] = train_dataset.as_returnn_opts()
    returnn_config.config["dev"] = cv_dataset.as_returnn_opts()

    from i6_core.returnn.training import ReturnnTrainingJob

    default_rqmt = {
        'mem_rqmt': 15,
        'time_rqmt': 168,
        'log_verbosity': 5,
        'returnn_python_exe': returnn_exe,
        'returnn_root': returnn_root,
    }

    train_job = ReturnnTrainingJob(
        returnn_config=returnn_config,
        num_epochs=250,
        **default_rqmt
    )
    train_job.add_alias(prefix_name + "/training")
    tk.register_output(prefix_name + "/learning_rates", train_job.out_learning_rates)



def test():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn_tf2.3_launcher_custom.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="e799984d109029c42816e6d5e0b5361b8bd7f05d").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/lstm_encdec_2021/prototype_pipelines"

    specaug_settings = SpecAugmentSettings()
    training(prefix_name + "/test_enc_only_fixed_specaug", returnn_exe, returnn_root, specaug_settings=specaug_settings)

    training(prefix_name + "/test_enc_only_fixed_specaug_stronger_pt_reg", returnn_exe, returnn_root, specaug_settings=specaug_settings, stronger_pt_reg=True)




