import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    training, search, get_best_checkpoint, search_single, get_average_checkpoint_v2
from i6_experiments.users.rossenbach.experiments.librispeech.librispeech_960_attention.conformer_2022.data import \
    build_training_datasets, build_test_dataset

from .attention_asr_config import create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs
from .zeineldeen_helpers.models.lm.transformer_lm import TransformerLM

from .feature_extraction_net import log10_net_10ms, log10_net_10ms_long_bn

def conformer_tf_features():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                  commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    prefix_name = "experiments/librispeech/librispeech_960_attention/conformer_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets_speedperturbed = build_training_datasets(
        returnn_exe,
        returnn_root_datasets,
        prefix_name,
        bpe_size=10000,
        use_raw_features=True,
        link_speed_perturbation=True
    )
    training_datasets_speedperturbed_retrain = build_training_datasets(
        returnn_exe,
        returnn_root_datasets,
        prefix_name, bpe_size=10000,
        use_raw_features=True,
        link_speed_perturbation=True,
        use_curicculum=False
    )

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name, use_raw_features=True)

    # ------------------------------------------------------------------------------------------------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    # fairseq init
    fairseq_ff_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"  # limit = sqrt(6 / (fan_in + fan_out))
    fairseq_mhsa_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)"  # limit = sqrt(6 * 0.5 / (fan_in + fan_out)) = sqrt(3 / (fan_in + fan_out))
    conformer_enc_args.ff_init = fairseq_ff_init
    conformer_enc_args.mhsa_init = fairseq_mhsa_init
    conformer_enc_args.mhsa_out_init = fairseq_ff_init
    conformer_enc_args.conv_module_init = fairseq_ff_init
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6, embed_dropout=0.1, label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc='rel',
        ff_init=fairseq_ff_init,
        mhsa_init=fairseq_mhsa_init,
        mhsa_out_init=fairseq_ff_init
    )

    training_args = {}

    # LR scheduling
    training_args['const_lr'] = [42, 100]  # use const LR during pretraining
    training_args['wup_start_lr'] = 0.0002
    training_args['wup'] = 20
    training_args['with_staged_network'] = True
    training_args['speed_pert'] = True

    # overwrite BN params
    conformer_enc_args.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': False,
        'delay_sample_update': False
    }

    # ---------------------------------------------------------
    # LM Settings
    # transf_lm_net = TransformerLM(
    #     source='prev:output', num_layers=24, vocab_size=2051, use_as_ext_lm=True, prefix_name='lm_')
    # transf_lm_net.create_network()
    # transf_lm_opts = {
    #     'lm_subnet': transf_lm_net.network.get_net(),
    #     'lm_output_prob_name': 'lm_output',
    #     'is_recurrent': True,
    #     'preload_from_files': {
    #         'lm_model': {
    #             'filename': '/work/asr4/zeineldeen/setups-data/librispeech/2021-02-21--lm-bpe/dependencies/lm_models/transf/epoch.016',
    #             'prefix': 'lm_'
    #         }
    #     },
    #     'name': 'trafo',
    # }
    # ---------------------------------------------------------

    # conformer round 2
    name = 'tf_feature_conformer_12l_lstm_1l_normal_v2'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_conformer_enc_args.ctc_loss_scale = 1.0
    local_training_args = copy.deepcopy(training_args)

    # pretraining
    local_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 22500*160}
    local_training_args['pretrain_reps'] = 5
    local_training_args['batch_size'] = 15000*160  # frames * samples per frame

    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy({**local_training_args, "encoder_args": local_conformer_enc_args, "decoder_args": rnn_dec_args})
    args['name'] = name
    args['with_staged_network'] = True
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="3f62155a08722310f51276792819b3c7c64ad356").out_repository

    def run_exp_v2(ft_name, feature_extraction_net, datasets, train_args, search_args=None):
        search_args = search_args if search_args is not None else train_args
        returnn_config = create_config(training_datasets=datasets, **train_args, feature_extraction_net=feature_extraction_net)
        returnn_search_config = create_config(training_datasets=datasets, **search_args, feature_extraction_net=feature_extraction_net, is_recog=True)
        train_job = training(ft_name, returnn_config, returnn_exe, returnn_root, num_epochs=250)
        # average = get_average_checkpoint(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
        average = get_average_checkpoint_v2(train_job, returnn_exe=returnn_exe, returnn_root=returnn_root, num_average=4)
        from i6_core.returnn.training import GetBestTFCheckpointJob
        best_checkpoint_job = GetBestTFCheckpointJob(
            train_job.out_model_dir,
            train_job.out_learning_rates,
            key="dev_score_output/output_prob",
            index=0)

        search(ft_name + "/default_best", returnn_search_config, best_checkpoint_job.out_checkpoint, test_dataset_tuples, returnn_exe, returnn_root)
        search(ft_name + "/default_last", returnn_search_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
        search(ft_name + "/average_4", returnn_search_config, average, test_dataset_tuples, returnn_exe, returnn_root)
        # ext_lm_search_args = copy.deepcopy(args)
        # ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

        # for lm_scale in [0.36, 0.38, 0.4, 0.42, 0.44]:
        #     search_args = copy.deepcopy(ext_lm_search_args)
        #     search_args['ext_lm_opts']['lm_scale'] = lm_scale
        #     returnn_config = create_config(training_datasets=training_datasets, **search_args, feature_extraction_net=feature_extraction_net)
        #     returnn_config.config["batch_size"] = 10000*200  # smaller size for recognition
        #     search_single(ft_name + "/default_last_ext_lm_%.2f" % lm_scale,
        #                   returnn_config,
        #                   train_job.out_checkpoints[250],
        #                   test_dataset_tuples["dev-other"][0],
        #                   test_dataset_tuples["dev-other"][1],
        #                   returnn_exe,
        #                   returnn_root)
        return train_job

    train_job_base = run_exp_v2(exp_prefix + "/" + "raw_log10", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=args)

    local_args = copy.deepcopy(args)
    local_args["config_override"] = {"newbob_error_threshold": 0.0}
    run_exp_v2(exp_prefix + "/" + "raw_log10_newbob_threshold_0", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=local_args)
 
    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain, train_args=args_retrain)


    # conformer round 2
    name = 'tf_feature_conformer_12l_trafo_6l_normal_v2'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_conformer_enc_args.ctc_loss_scale = 1.0
    local_training_args = copy.deepcopy(training_args)
    local_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 20000*160}
    local_training_args['pretrain_reps'] = 5
    local_training_args['batch_size'] = 12000*160  # frames * samples per frame
    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy({**local_training_args, "encoder_args": local_conformer_enc_args, "decoder_args": trafo_dec_args})
    args['name'] = name

    train_job_base = run_exp_v2(exp_prefix + "/" + "raw_log10", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=args)
    train_job_base_lbn = run_exp_v2(exp_prefix + "/" + "raw_log10_bnfeat", log10_net_10ms_long_bn, datasets=training_datasets_speedperturbed, train_args=args)

    args_retrain = copy.deepcopy(args)
    args_retrain["retrain_checkpoint"] = train_job_base.out_checkpoints[250]
    train_job = run_exp_v2(exp_prefix + "/" + "raw_log10_retrain", log10_net_10ms, datasets=training_datasets_speedperturbed_retrain, train_args=args_retrain)


    local_conformer_enc_args_fix_bn = copy.deepcopy(local_conformer_enc_args)
    local_conformer_enc_args_fix_bn.batch_norm_opts = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': True,
        'delay_sample_update': True,
    }
    args = copy.deepcopy({**local_training_args, "encoder_args": local_conformer_enc_args_fix_bn, "decoder_args": trafo_dec_args})
    args['name'] = name

    train_job_base = run_exp_v2(exp_prefix + "/" + "raw_log10_fix_bn", log10_net_10ms, datasets=training_datasets_speedperturbed, train_args=args)

