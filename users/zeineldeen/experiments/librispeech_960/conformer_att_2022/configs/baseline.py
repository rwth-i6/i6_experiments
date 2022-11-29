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

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_training_datasets(
        bpe_size=10000,
        use_raw_features=True,
        link_speed_perturbation=True
    )

    train_data_retrain = build_training_datasets(
        bpe_size=10000,
        use_raw_features=True,
        link_speed_perturbation=True,
        use_curicculum=False
    )

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, use_raw_features=True)


    def run_exp(exp_name, feature_extraction_net, datasets, train_args, search_args=None):
        exp_prefix = os.path.join(prefix_name, exp_name)

        returnn_config = create_config(
            training_datasets=datasets, **train_args, feature_extraction_net=feature_extraction_net)

        num_epochs = train_args.get('num_epochs', 300)
        train_job = training(
            exp_prefix, returnn_config, RETURNN_EXE, RETURNN_ROOT, num_epochs=num_epochs)

        train_jobs_map[exp_name] = train_job

        search_args = search_args if search_args is not None else train_args

        returnn_search_config = create_config(
            training_datasets=datasets, **search_args, feature_extraction_net=feature_extraction_net, is_recog=True)

        averaged_checkpoint = get_average_checkpoint(
            train_job, returnn_exe=RETURNN_EXE, returnn_root=RETURNN_ROOT, num_average=4)
        best_checkpoint = get_best_checkpoint(train_job)

        search(exp_prefix + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs], test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        search(exp_prefix + "/default_best", returnn_search_config, best_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)
        search(exp_prefix + "/average_4", returnn_search_config, averaged_checkpoint, test_dataset_tuples, RETURNN_EXE, RETURNN_ROOT)

        return train_job

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
    lstm_training_args['pretrain_opts'] = {'variant': 3, "initial_batch_size": 22500*160}
    lstm_training_args['pretrain_reps'] = 5
    lstm_training_args['batch_size'] = 15000*160  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy({**lstm_training_args, "encoder_args": conformer_enc_args, "decoder_args": rnn_dec_args})


    # --------------------- Experiments --------------------- #

    run_exp(exp_name='base_conf_12l_trafo_6l', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=trafo_dec_exp_args)
    run_exp(exp_name='base_conf_12l_lstm_1l', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=lstm_dec_exp_args)

    # TODO: default init
    args = copy.deepcopy(trafo_dec_exp_args)
    reset_params_init(args['encoder_args'])
    reset_params_init(args['decoder_args'])
    run_exp('base_conf12l_trafo_defaultInit', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)

    # TODO: pretrain variant 4
    for reps in [5, 6]:
        args = copy.deepcopy(trafo_dec_exp_args)
        args['pretrain_opts']['variant'] = 4
        args['pretrain_reps'] = reps
        name = f'base_conf12l_trafo_6l_pretrain4_reps{reps}'
        run_exp(exp_name=name, feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)

    # TODO: tune L2
    args = copy.deepcopy(trafo_dec_exp_args)
    args['encoder_args'].l2 = 1e-6
    args['decoder_args'].l2 = 1e-6
    run_exp('base_conf12l_trafo_6l_L2e-6', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)

    # TODO: wo apply embed weight
    args = copy.deepcopy(trafo_dec_exp_args)
    args['decoder_args'].apply_embed_weight = False
    run_exp('base_conf12l_trafo_6l_noEmbWeight', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)

    # TODO: LR scheduling
    args = copy.deepcopy(trafo_dec_exp_args)
    args['const_lr'] = 0
    run_exp('base_conf12l_trafo_6l_noConstLR', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)

    args = copy.deepcopy(trafo_dec_exp_args)
    args['const_lr'] = [35, 20]
    run_exp('base_conf12l_trafo_6l_constLR_35-20', feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)

    # TODO: No pretraining and long warmup
    for wup in [20, 30, 40]:
        for wup_start_lr in [2e-4, 2e-5, 1e-4]:
            args = copy.deepcopy(trafo_dec_exp_args)
            args['const_lr'] = 0
            args['with_pretrain'] = False
            args['wup_start_lr'] = wup_start_lr
            args['wup'] = wup
            run_exp(f'base_conf12l_trafo_6l_noPre_wup{wup}_startLR{wup_start_lr}',
                    feature_extraction_net=log10_net_10ms, datasets=train_data, train_args=args)
