import copy, os

from i6_experiments.users.zeineldeen.experiments.librispeech_960.conformer_att_2022.attention_asr_config import \
    create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs
from i6_experiments.users.zeineldeen.experiments.librispeech_960.conformer_att_2022.additional_config import \
    apply_fairseq_init_to_conformer_encoder, apply_fairseq_init_to_transformer_decoder, reset_params_init
from i6_experiments.users.zeineldeen.experiments.librispeech_960.conformer_att_2022.data import \
    build_training_datasets, build_test_dataset
from i6_experiments.users.zeineldeen.experiments.librispeech_960.conformer_att_2022.default_tools import \
    RETURNN_EXE, RETURNN_ROOT
from i6_experiments.users.zeineldeen.experiments.librispeech_960.conformer_att_2022.feature_extraction_net import \
    log10_net_10ms, log10_net_10ms_long_bn
from i6_experiments.users.zeineldeen.experiments.librispeech_960.conformer_att_2022.pipeline import \
    training, search, get_average_checkpoint, get_best_checkpoint


train_jobs_map = {}  # dict[str, ReturnnTrainJob]


def conformer_baseline():

    abs_name = os.path.abspath(__file__)
    #prefix_name = abs_name[abs_name.find('/experiments') + len('/experiments') + 1:][:-len('.py')]
    prefix_name = os.path.basename(abs_name)[:-len('.py')]

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, use_raw_features=True)

    def run_exp(exp_name, train_args, feature_extraction_net=log10_net_10ms, num_epochs=300, search_args=None, **kwargs):
        exp_prefix = os.path.join(prefix_name, exp_name)

        train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get('epoch_wise_filter', [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get('speed_pert', True),
            seq_ordering=kwargs.get('seq_ordering', 'laplace:.1000'),
        )

        returnn_config = create_config(
            training_datasets=train_data, **train_args, feature_extraction_net=feature_extraction_net)

        train_job = training(
            exp_prefix, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=num_epochs)

        train_jobs_map[exp_name] = train_job

        search_args = search_args if search_args is not None else train_args

        returnn_search_config = create_config(
            training_datasets=train_data, **search_args, feature_extraction_net=feature_extraction_net, is_recog=True)

        averaged_checkpoint = get_average_checkpoint(
            train_job, returnn_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT, num_average=4)
        best_checkpoint = get_best_checkpoint(train_job)

        search(exp_prefix + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs], test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        search(exp_prefix + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        search(exp_prefix + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

    # --------------------------- General Settings --------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)
    apply_fairseq_init_to_conformer_encoder(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6, embed_dropout=0.1, label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc='rel',
    )
    apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

    training_args = dict()

    # LR scheduling
    training_args['const_lr'] = [42, 100]  # use const LR during pretraining
    training_args['wup_start_lr'] = 0.0002
    training_args['wup'] = 20
    training_args['with_staged_network'] = True
    training_args['speed_pert'] = True

    trafo_training_args = copy.deepcopy(training_args)
    trafo_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 20000 * 160}
    trafo_training_args['pretrain_reps'] = 5
    trafo_training_args['batch_size'] = 12000 * 160  # frames * samples per frame

    trafo_dec_exp_args = copy.deepcopy({**trafo_training_args, "encoder_args": conformer_enc_args, "decoder_args": trafo_dec_args})

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 22500 * 160}
    lstm_training_args['pretrain_reps'] = 5
    lstm_training_args['batch_size'] = 15000 * 160  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy({**lstm_training_args, "encoder_args": conformer_enc_args, "decoder_args": rnn_dec_args})

    # --------------------------- Experiments --------------------------- #

    run_exp(exp_name='base_conf_12l_trafo_6l', train_args=trafo_dec_exp_args)
    run_exp(exp_name='base_conf_12l_lstm_1l', train_args=lstm_dec_exp_args)

    for bn_mom in [0.01, 0.001]:
        feature_net = copy.deepcopy(log10_net_10ms_long_bn)
        feature_net['log_mel_features']['momentum'] = bn_mom
        run_exp(
            exp_name=f'base_conf_12l_trafo_6l_featureBN{bn_mom}',
            feature_extraction_net=feature_net,
            train_args=trafo_dec_exp_args)

    # TODO: BN eps
    args = copy.deepcopy(trafo_dec_exp_args)
    args['batch_norm_opts'] = {'epsilon': 1e-5}
    run_exp(exp_name='base_conf_12l_trafo_6l_BNeps_1e-5', train_args=trafo_dec_exp_args)

    # TODO: conv for lstm decoder
    # OOM for batch size 15k
    for bs in [10, 11, 15]:
        args = copy.deepcopy(lstm_dec_exp_args)
        args['batch_size'] = bs * 1000 * 160
        args['encoder_args'].input_layer = 'conv-4'
        args['encoder_args'].input_layer_conv_act = 'relu'
        run_exp(exp_name=f'base_conf_12l_lstm_1l_conv4_relu_bs{bs}k', train_args=args)

    args = copy.deepcopy(lstm_dec_exp_args)
    args['decoder_args'].zoneout = False
    args['decoder_args'].dropout = 0.1
    run_exp(exp_name='base_conf_12l_lstm_1l_noZonout_drop1e-1', train_args=args)

    args = copy.deepcopy(lstm_dec_exp_args)
    args['decoder_args'].lstm_num_units = 1000
    args['decoder_args'].output_num_units = 1000
    run_exp(exp_name='base_conf_12l_lstm_1l_dim1k', train_args=args)

    args = copy.deepcopy(lstm_dec_exp_args)
    args['decoder_args'].lstm_num_units = args['encoder_args'].enc_key_dim
    args['decoder_args'].output_num_units = args['encoder_args'].enc_key_dim
    args['decoder_args'].embed_dim = args['encoder_args'].enc_key_dim
    run_exp(exp_name='base_conf_12l_lstm_1l_encKeyDim', train_args=args)

    # TODO: no epoch wise filter
    args = copy.deepcopy(trafo_dec_exp_args)
    run_exp('base_conf12l_trafo_noCurrLR', train_args=args, epoch_wise_filter=None)

    # TODO: seq ordering
    args = copy.deepcopy(trafo_dec_exp_args)
    run_exp('base_conf12l_trafo_seqOrdLaplace281', train_args=args, seq_ordering='laplace:281')

    # TODO: more curr learning
    args = copy.deepcopy(trafo_dec_exp_args)
    run_exp('base_conf12l_trafo_currLRv2', train_args=args,
            epoch_wise_filter=[(1, 5, 1000), (5, 10, 2000), (10, 20, 3000)])

    args = copy.deepcopy(trafo_dec_exp_args)
    args['pretrain_opts']['variant'] = 4
    args['pretrain_reps'] = 6
    args['gradient_clip'] = 20
    run_exp('base_conf12l_trafo_currLRv2_pre4_reps6_gradClip20', train_args=args,
            epoch_wise_filter=[(1, 5, 1000), (5, 10, 2000), (10, 20, 3000)])

    # TODO: conv front-end
    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].input_layer = 'conv-4'
    run_exp(exp_name='base_conf_12l_trafo_6l_conv4_relu', train_args=args)

    # TODO: OOV with 12k
    for bs in [10, 11, 12]:
        args = copy.deepcopy(trafo_dec_exp_args)
        args['batch_size'] = bs * 1000 * 160
        args['encoder_args'].input_layer = 'conv-4'
        run_exp(exp_name=f'base_conf_12l_trafo_6l_conv4_relu_bs{bs}k', train_args=args)

    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].input_layer = 'conv-6'
    run_exp(exp_name=f'base_conf_12l_trafo_6l_conv6_relu', train_args=args)

    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].input_layer = 'conv-4'
    args['encoder_args'].input_layer_conv_act = 'swish'
    run_exp(exp_name='base_conf_12l_trafo_6l_conv4_swish', train_args=args)

    # TODO: default init
    args = copy.deepcopy(trafo_dec_exp_args)
    reset_params_init(args['encoder_args'])
    reset_params_init(args['decoder_args'])
    run_exp('base_conf12l_trafo_defaultInit', train_args=args)

    # TODO: tune L2
    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].l2 = 1e-6
    args['decoder_args'].l2 = 1e-6
    run_exp('base_conf12l_trafo_6l_L2e-6', train_args=args)

    # TODO: wo apply embed weight
    args = copy.deepcopy(trafo_dec_exp_args)
    args['decoder_args'].apply_embed_weight = False
    run_exp('base_conf12l_trafo_6l_noEmbWeight', train_args=args)

    # TODO: LR scheduling
    args = copy.deepcopy(trafo_dec_exp_args)
    args['const_lr'] = [35, 20]
    run_exp('base_conf12l_trafo_6l_constLR_35-20', train_args=args)

    for bn_mom in [0.01, 0.001]:
        args = copy.deepcopy(trafo_dec_exp_args)
        args['const_lr'] = [35, 20]
        feature_net = copy.deepcopy(log10_net_10ms_long_bn)
        feature_net['log_mel_features']['momentum'] = bn_mom
        run_exp(f'base_conf12l_trafo_6l_constLR_35-20_featBN{bn_mom}', feature_extraction_net=feature_net, train_args=args)

    # TODO: use LN instead
    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].use_ln = True
    args['gradient_clip'] = 20
    run_exp('base_conf12l_trafo_6l_LN_gradClip20', train_args=args)

    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].use_sqrd_relu = True
    run_exp('base_conf12l_trafo_6l_sqrdReLU', train_args=args)

    # --------------------------- No Pretraining --------------------------- #

    # TODO: more curr learning
    # TODO: const lr in the beginning

    # TODO: OCLR
    for const_lr_ep in [10, 20, 30, 40]:
        args = copy.deepcopy(trafo_dec_exp_args)
        args['oclr_opts'] = {
            'peak_lr': 8e-4,
            'final_lr': 1e-6,
            'cycle_ep': int((180 + const_lr_ep) * 0.45),
            'total_ep': 400 + const_lr_ep,  # 20 epochs
            'n_step': 1700,
            'learning_rates': [8e-5] * const_lr_ep
        }
        args['with_pretrain'] = False
        run_exp(f'base_conf12l_trafo6l_OCLR_nopre_constLR{const_lr_ep}_Curr4', train_args=args, num_epochs=400 + const_lr_ep,
                epoch_wise_filter=[(1, 5, 1000), (5, 10, 2000), (10, 20, 3000)])

    # TODO: warmup LR
    # 20 full epochs = 400 subepochs
    # each subepoch is around 1700 steps
    # args = copy.deepcopy(trafo_dec_exp_args)
    # for peak_lr, wup_steps in [(8e-4, 6800), (8e-4, 8500), (8e-4, 10200)]:
    #     args['warmup_lr_opts'] = {'peak_lr': peak_lr, 'warmup_steps': wup_steps}
    #     args['with_pretrain'] = False
    #     run_exp(f'base_conf12l_trafo6l_warmupLR_{peak_lr}-{wup_steps}_nopre_Curr4', train_args=args, num_epochs=400,
    #             epoch_wise_filter=[(1, 5, 1000), (5, 10, 2000), (10, 20, 3000)])

