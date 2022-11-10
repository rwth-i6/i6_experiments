import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, search_single, get_average_checkpoint, get_average_checkpoint_v2

from .attention_asr_config import create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs
from .base_config import get_lm_opts, apply_fairseq_init_to_conformer_encoder
from .feature_extraction_net import log10_net, log10_net_10ms


def conformer_tts_tf_features():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    prefix_name = "experiments/librispeech/librispeech_100_attention/conformer_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True)
    training_datasets_speedperturbed = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True)
    # retrain dataset has curriculum options disabled
    training_datasets_speedperturbed_retrain = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000, use_raw_features=True, link_speed_perturbation=True, use_curicculum=False)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name, use_raw_features=True)

    # ------------------------------------------------------------------------------------------------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()
    training_args = {}

    # LR scheduling
    training_args['const_lr'] = [42, 100]  # use const LR during pretraining
    training_args['wup_start_lr'] = 0.0002
    training_args['wup'] = 20

    training_args['speed_pert'] = True

    # overwrite BN params
    conformer_enc_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': False,
        'delay_sample_update': False
    }

    # pretraining
    training_args['pretrain_opts'] = {'variant': 4, "initial_batch_size": 20000*200}
    training_args['pretrain_reps'] = 6
    training_args['with_staged_network'] = True

    transf_lm_opts = get_lm_opts()

    # conformer round 2
    name = 'tf_feature_conformer_12l_lstm_1l_short_v2'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_training_args = copy.deepcopy(training_args)

    # pretraining
    local_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 20000*200}
    local_training_args['pretrain_reps'] = 6
    local_training_args['batch_size'] = 15000*200  # frames * samples per frame

    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy({**local_training_args, "encoder_args": local_conformer_enc_args, "decoder_args": rnn_dec_args})
    args['name'] = name
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="3f62155a08722310f51276792819b3c7c64ad356").out_repository

    def run_exp_v2(ft_name, feature_extraction_net, datasets, train_epochs=250, **args):
        returnn_config = create_config(training_datasets=datasets, **args, feature_extraction_net=feature_extraction_net)
        returnn_search_config = create_config(training_datasets=datasets, **args, feature_extraction_net=feature_extraction_net, is_recog=True)
        train_job = training(ft_name, returnn_config, returnn_exe, returnn_root, num_epochs=train_epochs)
        average = get_average_checkpoint_v2(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
        from i6_core.returnn.training import GetBestTFCheckpointJob
        best_checkpoint_job = GetBestTFCheckpointJob(
            train_job.out_model_dir,
            train_job.out_learning_rates,
            key="dev_score_output/output_prob",
            index=0)

        search(ft_name + "/default_best", returnn_search_config, best_checkpoint_job.out_checkpoint, test_dataset_tuples, returnn_exe, returnn_root)
        search(ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[train_epochs], test_dataset_tuples, returnn_exe, returnn_root)
        search(ft_name + "/average_4", returnn_search_config, average, test_dataset_tuples, returnn_exe, returnn_root)

        ext_lm_search_args = copy.deepcopy(args)
        ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

        for lm_scale in [0.36, 0.38, 0.4, 0.42, 0.44]:
            search_args = copy.deepcopy(ext_lm_search_args)
            search_args['ext_lm_opts']['lm_scale'] = lm_scale
            returnn_search_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=feature_extraction_net, is_recog=True)
            returnn_search_config.config["batch_size"] = 10000*200  # smaller size for recognition
            search_single(ft_name + "/default_last_ext_lm_%.2f" % lm_scale,
                          returnn_search_config,
                          train_job.out_checkpoints[train_epochs],
                          test_dataset_tuples["dev-other"][0],
                          test_dataset_tuples["dev-other"][1],
                          returnn_exe,
                          returnn_root)
        return train_job

    run_exp_v2(exp_prefix + "/" + "raw_log10", log10_net, datasets=training_datasets_speedperturbed, train_epochs=300, **args)