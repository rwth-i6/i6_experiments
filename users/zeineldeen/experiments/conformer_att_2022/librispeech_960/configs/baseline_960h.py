import copy, os

import numpy

from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.attention_asr_config import \
    create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs, ConformerDecoderArgs
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.additional_config import \
    apply_fairseq_init_to_conformer, apply_fairseq_init_to_transformer_decoder, reset_params_init
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.data import \
    build_training_datasets, build_test_dataset
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.default_tools import \
    RETURNN_EXE, RETURNN_ROOT, RETURNN_CPU_EXE
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.feature_extraction_net import \
    log10_net_10ms, log10_net_10ms_long_bn
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.pipeline import \
    training, search, get_average_checkpoint, get_best_checkpoint, search_single
from i6_experiments.users.zeineldeen.models.lm import generic_lm
from i6_experiments.users.zeineldeen.models.lm.transformer_lm import TransformerLM
from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960 import ilm_helpers

train_jobs_map = {}  # dict[str, ReturnnTrainJob]
train_job_avg_ckpt = {}
train_job_best_epoch = {}

# --------------------------- LM --------------------------- #

lstm_lm_opts = {
    'lm_subnet': generic_lm.libri_lstm_bpe10k_net,
    'lm_model': generic_lm.libri_lstm_bpe10k_model,
    'name': 'lstm',
}

trafo_lm_net = TransformerLM(source='prev:output', num_layers=24, vocab_size=10025, use_as_ext_lm=True)
trafo_lm_net.create_network()
trafo_lm_opts = {
    'lm_subnet': trafo_lm_net.network.get_net(),
    'load_on_init_opts': {
        'filename': '/work/asr3/irie/experiments/lm/librispeech/2018-03-05--lmbpe-zeyer/data-train/transfo_24_d00.4096_1024.sgd.lr1.8_heads/bk-net-model/network.023',
        'params_prefix': '',
        'load_if_prefix': 'lm_output/',
    },
    'name': 'trafo'
}

# ----------------------------------------------------------- #

def conformer_baseline():

    abs_name = os.path.abspath(__file__)
    prefix_name = os.path.basename(abs_name)[: -len(".py")]

    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            testset, use_raw_features=True,
        )

    def run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs):
        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_config = create_config(
            training_datasets=train_data,
            **train_args,
            feature_extraction_net=feature_extraction_net,
            recog_epochs=recog_epochs,
        )
        train_job = training(exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs)
        return train_job

    def run_single_search(
            exp_name, train_data, search_args, checkpoint, feature_extraction_net, recog_dataset, recog_ref,
            mem_rqmt=8, time_rqmt=4, **kwargs):

        exp_prefix = os.path.join(prefix_name, exp_name)
        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )
        search_single(
            exp_prefix,
            returnn_search_config,
            checkpoint,
            recognition_dataset=recog_dataset,
            recognition_reference=recog_ref,
            returnn_exe=RETURNN_CPU_EXE,
            returnn_root=RETURNN_ROOT,
            mem_rqmt=mem_rqmt,
            time_rqmt=time_rqmt,
        )

    def run_lm_fusion(
            lm_type, exp_name, epoch, test_set_names, lm_scales, train_job, train_data, feature_net, args, beam_size=12,
            prior_scales=None, prior_type=None, mini_lstm_ckpt=None, length_norm=True, prior_type_name=None,
            coverage_scale=None, coverage_threshold=None, **kwargs,
    ):
        assert lm_type in ['lstm', 'trafo'], 'lm type should be lstm or trafo'

        if isinstance(lm_scales, float):
            lm_scales = [lm_scales]
        if prior_scales and isinstance(prior_scales, float):
            prior_scales = [prior_scales]
        if isinstance(test_set_names, str):
            test_set_names = [test_set_names]
        assert isinstance(test_set_names, list)

        if epoch == 'avg':
            search_checkpoint = train_job_avg_ckpt[exp_name]
        elif epoch == 'best':
            search_checkpoint = train_job_best_epoch[exp_name]
        else:
            assert isinstance(epoch, int), 'epoch must be either a defined integer or a string in {avg, best}.'
            search_checkpoint = train_job.out_checkpoints[epoch]

        ext_lm_opts = lstm_lm_opts if lm_type == 'lstm' else trafo_lm_opts

        time_rqmt = 1.0

        search_args = copy.deepcopy(args)

        if lm_type == 'lstm':
            if beam_size > 128:
                search_args['batch_size'] = 4000 * 160

        if lm_type == 'trafo':
            search_args['batch_size'] = 4000 * 160 if beam_size <= 32 else 2000 * 160
            time_rqmt = 1

        search_args['beam_size'] = beam_size
        if kwargs.get('batch_size', None):
            search_args['batch_size'] = kwargs['batch_size']

        if not length_norm:
            search_args['decoder_args'].length_normalization = False

        if 'decoder_args' in kwargs:
            for k, v in kwargs['decoder_args'].items():
                setattr(search_args['decoder_args'], k, v)

        scales = [(e,) for e in lm_scales]

        for test_set in test_set_names:

            if prior_scales:
                import itertools
                scales = itertools.product(lm_scales, prior_scales)

            for scale in scales:
                lm_scale = scale[0]
                prior_scale = scale[1] if len(scale) == 2 else None
                if prior_scale and prior_scale > lm_scale:
                    continue

                # External LM opts
                ext_lm_opts['lm_scale'] = lm_scale
                search_args['ext_lm_opts'] = ext_lm_opts

                # ILM opts
                if prior_scale:
                    ilm_opts = {
                        'scale': prior_scale,
                        'type': prior_type,
                        'ctx_dim': search_args['encoder_args'].enc_key_dim,  # this is needed for mini-lstm
                    }
                    # this is needed for mini-self-att
                    if hasattr(search_args['decoder_args'], 'num_layers'):
                        ilm_opts['num_dec_layers'] = search_args['decoder_args'].num_layers
                        search_args['decoder_args'].create_ilm_decoder = True
                        search_args['decoder_args'].ilm_type = prior_type

                    search_args['prior_lm_opts'] = ilm_opts
                    search_args['preload_from_files'] = {
                        'prior_lm': {
                            'filename': search_checkpoint,  # copy ASR decoder to be used as ILM decoder
                            'prefix': 'prior_'
                        }
                    }
                    if prior_type == 'mini_lstm':
                        assert mini_lstm_ckpt, 'Mini-LSTM checkpoint not set.'
                        search_args['preload_from_files'].update(
                            {
                                'mini_lstm': {
                                    'filename': mini_lstm_ckpt,
                                    'prefix': 'mini_'
                                }
                            })

                if prior_type_name is None:
                    prior_type_name = prior_type

                lm_desc = f'lm-scale-{lm_scale}'
                if prior_scale:
                    lm_desc += f'-prior-{prior_scale}-{prior_type_name}'
                lm_desc += f'-beam-{beam_size}'
                if length_norm is False:
                    lm_desc += '-woLenNorm'

                if coverage_scale and coverage_threshold:
                    assert isinstance(search_args['decoder_args'], RNNDecoderArgs)
                    search_args['decoder_args'].coverage_scale = coverage_scale
                    search_args['decoder_args'].coverage_threshold = coverage_threshold
                    lm_desc += f'_coverage-thre{coverage_threshold}-scale{coverage_scale}'

                name = f'{exp_name}/recog-{lm_type}-lm/ep-{epoch}/{lm_desc}/{test_set}'

                run_single_search(
                    exp_name=name,
                    train_data=train_data,
                    search_args=search_args,
                    checkpoint=search_checkpoint,
                    feature_extraction_net=feature_net,
                    recog_dataset=test_dataset_tuples[test_set][0],
                    recog_ref=test_dataset_tuples[test_set][1],
                    time_rqmt=kwargs.get('time_rqmt', time_rqmt),
                )


    def run_search(
        exp_name, train_args, train_data, train_job, feature_extraction_net, num_epochs, search_args, recog_epochs,
        **kwargs):
        
        exp_prefix = os.path.join(prefix_name, exp_name)

        search_args = search_args if search_args is not None else train_args

        returnn_search_config = create_config(
            training_datasets=train_data,
            **search_args,
            feature_extraction_net=feature_extraction_net,
            is_recog=True,
        )

        num_avg = kwargs.get('num_avg', 4)
        averaged_checkpoint = get_average_checkpoint(
            train_job, returnn_exe=RETURNN_CPU_EXE, returnn_root=RETURNN_ROOT, num_average=num_avg,
        )
        if num_avg == 4:  # TODO: just for now to not break hashes
            train_job_avg_ckpt[exp_name] = averaged_checkpoint

        best_checkpoint = get_best_checkpoint(train_job)
        train_job_best_epoch[exp_name] = best_checkpoint

        if recog_epochs is None:
            default_recog_epochs = [80 * i for i in range(1, int(num_epochs / 80) + 1)]
            if num_epochs % 80 != 0:
                default_recog_epochs += [num_epochs]
        else:
            default_recog_epochs = recog_epochs

        for ep in default_recog_epochs:
            search(
                exp_prefix + f"/recogs/ep-{ep}", returnn_search_config, train_job.out_checkpoints[ep],
                test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

        search(
            exp_prefix + "/default_last", returnn_search_config, train_job.out_checkpoints[num_epochs],
            test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

        search(
            exp_prefix + "/default_best", returnn_search_config, best_checkpoint,
            test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

        search(
            exp_prefix + f"/average_{num_avg}", returnn_search_config, averaged_checkpoint,
            test_dataset_tuples, RETURNN_CPU_EXE, RETURNN_ROOT)

    def run_exp(
        exp_name, train_args, feature_extraction_net=log10_net_10ms, num_epochs=300, search_args=None,
        recog_epochs=None, bpe_size=10000, **kwargs
    ):
        if train_args.get('retrain_checkpoint', None):
            assert kwargs.get('epoch_wise_filter', None) is None, 'epoch_wise_filter should be disabled for retraining.'
        train_data = build_training_datasets(
            bpe_size=bpe_size,
            use_raw_features=True,
            epoch_wise_filter=kwargs.get("epoch_wise_filter", [(1, 5, 1000)]),
            link_speed_perturbation=train_args.get("speed_pert", True),
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
        train_job = run_train(exp_name, train_args, train_data, feature_extraction_net, num_epochs, recog_epochs, **kwargs)
        train_jobs_map[exp_name] = train_job
          
        run_search(
            exp_name, train_args, train_data, train_job, feature_extraction_net, num_epochs, search_args, recog_epochs,
            **kwargs
        )
        return train_job, train_data

    def train_mini_lstm(
        exp_name, checkpoint, args, num_epochs=20, lr=8e-4, time_rqmt=4, l2=1e-4, name='mini_lstm',
        w_drop=False, **kwargs
    ):
        if not w_drop:
            params_freeze_str = ilm_helpers.get_mini_lstm_params_freeze_str()
        else:
            params_freeze_str = ilm_helpers.get_mini_lstm_params_freeze_str_w_drop()

        mini_lstm_args = copy.deepcopy(args)
        mini_lstm_args['batch_size'] = 20000 * 160
        mini_lstm_args['with_pretrain'] = False
        mini_lstm_args['lr'] = lr
        mini_lstm_args['allow_lr_scheduling'] = False
        mini_lstm_args['encoder_args'].with_ctc = False
        mini_lstm_args['keep_all_epochs'] = True  # keep everything
        mini_lstm_args['extra_str'] = params_freeze_str
        mini_lstm_args['preload_from_files'] = {
            'import': {
                'init_for_train': True,
                'ignore_missing': True,
                'filename': checkpoint,
            }
        }
        mini_lstm_args.update(kwargs)

        exp_prefix = os.path.join(prefix_name, exp_name, name)
        mini_lstm_train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,  # depends only on text
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )
        returnn_config = create_config(
            training_datasets=mini_lstm_train_data,
            **mini_lstm_args,
            feature_extraction_net=log10_net_10ms,
        )

        returnn_config.config['network']['output']['unit']['att_lstm'] = {
            "class": "rec", "unit": "nativelstm2", "from": 'prev:target_embed', "n_out": 50,
        }

        returnn_config.config['network']['output']['unit']['att'] = {
            "class": "linear", "from": "att_lstm", "activation": None,
            "n_out": mini_lstm_args['encoder_args'].enc_key_dim, "L2": l2,
        }

        train_job = training(
            exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs, time_rqmt=time_rqmt)
        return train_job

    def train_mini_self_att(
        exp_name, checkpoint, args, num_epochs=20, lr=8e-4, time_rqmt=4, name='mini_self_att', **kwargs):
        """
        Same idea as Mini-LSTM but use masked (mini-)self-attention models instead of cross attention.
        Note that each layer has its own (mini-)self-attention.

        In the case of transformer decoder, we want to replace cross-attention layers namely:
            transformer_decoder_{idx}_att_linear
        with masked self-attention models.
        """

        params_freeze_str = ilm_helpers.get_mini_self_att_params_freeze_str_w_drop(args['decoder_args'].num_layers)

        mini_self_att = copy.deepcopy(args)
        mini_self_att['batch_size'] = 20000 * 160  # TODO: does this fit now?
        mini_self_att['with_pretrain'] = False
        mini_self_att['lr'] = lr
        mini_self_att['allow_lr_scheduling'] = False
        mini_self_att['encoder_args'].with_ctc = False
        #mini_self_att['keep_all_epochs'] = True  # keep everything
        mini_self_att['extra_str'] = params_freeze_str
        mini_self_att['preload_from_files'] = {
            'import': {
                'init_for_train': True,
                'ignore_missing': True,
                'filename': checkpoint,
            }
        }
        if 'decoder_args' in kwargs:
            assert isinstance(kwargs['decoder_args'], dict)
            for k, v in kwargs['decoder_args'].items():
                setattr(mini_self_att['decoder_args'], k, v)
            kwargs.pop('decoder_args')
        mini_self_att.update(kwargs)

        exp_prefix = os.path.join(prefix_name, exp_name, name)
        mini_self_att_train_data = build_training_datasets(
            bpe_size=10000,
            use_raw_features=True,
            epoch_wise_filter=None,
            link_speed_perturbation=False,  # depends only on text
            seq_ordering=kwargs.get("seq_ordering", "laplace:.1000"),
        )

        # use masked self-att instead of cross-att with layer names having "ilm_" as prefix
        mini_self_att['decoder_args'].replace_cross_att_w_masked_self_att = True

        returnn_config = create_config(
            training_datasets=mini_self_att_train_data,
            **mini_self_att,
            feature_extraction_net=log10_net_10ms,
        )
        train_job = training(
            exp_prefix, returnn_config, RETURNN_CPU_EXE, RETURNN_ROOT, num_epochs=num_epochs, time_rqmt=time_rqmt)
        return train_job


    # --------------------------- General Settings --------------------------- #

    conformer_enc_args = ConformerEncoderArgs(
        num_blocks=12,
        input_layer="conv-6",
        att_num_heads=8,
        ff_dim=2048,
        enc_key_dim=512,
        conv_kernel_size=32,
        pos_enc="rel",
        dropout=0.1,
        att_dropout=0.1,
        l2=0.0001,
    )
    apply_fairseq_init_to_conformer(conformer_enc_args)
    conformer_enc_args.ctc_loss_scale = 1.0

    rnn_dec_args = RNNDecoderArgs()

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6,
        embed_dropout=0.1,
        label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc="rel",
    )
    apply_fairseq_init_to_transformer_decoder(trafo_dec_args)

    conformer_dec_args = ConformerDecoderArgs()
    apply_fairseq_init_to_conformer(conformer_dec_args)

    training_args = dict()

    # LR scheduling
    training_args["const_lr"] = [42, 100]  # use const LR during pretraining
    training_args["wup_start_lr"] = 0.0002
    training_args["wup"] = 20
    training_args["with_staged_network"] = True
    training_args["speed_pert"] = True

    trafo_training_args = copy.deepcopy(training_args)
    trafo_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 20000 * 160}
    trafo_training_args["pretrain_reps"] = 5
    trafo_training_args["batch_size"] = 12000 * 160  # frames * samples per frame

    trafo_dec_exp_args = copy.deepcopy(
        {
            **trafo_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": trafo_dec_args,
        }
    )

    conformer_dec_exp_args = copy.deepcopy(trafo_dec_exp_args)
    conformer_dec_exp_args['decoder_args'] = conformer_dec_args

    lstm_training_args = copy.deepcopy(training_args)
    lstm_training_args["pretrain_opts"] = {"variant": 3, "initial_batch_size": 22500 * 160}
    lstm_training_args["pretrain_reps"] = 5
    lstm_training_args["batch_size"] = 15000 * 160  # frames * samples per frame

    lstm_dec_exp_args = copy.deepcopy(
        {
            **lstm_training_args,
            "encoder_args": conformer_enc_args,
            "decoder_args": rnn_dec_args,
        }
    )

    # --------------------------- Experiments --------------------------- #


    # TODO: OCLR
    oclr_args = copy.deepcopy(trafo_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 8e-4,
        "final_lr": 1e-6,
        "cycle_ep": 195,
        "total_ep": 435,  # 20 epochs
        "n_step": 1700,
        "learning_rates": [8e-5] * 35  # TODO: this is not used so should be removed.
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    train_j, train_data = run_exp("base_conf_12l_trafo6l_OCLR_conv6_relu", train_args=oclr_args, num_epochs=435)

    args = copy.deepcopy(oclr_args)
    args['oclr_opts']['cycle_ep'] = 465
    args['oclr_opts']['total_ep'] = 1035
    args['encoder_args'].use_sqrd_relu = True
    train_j, train_data = run_exp(
        "base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035", train_args=args, num_epochs=1035)

    # # TODO: retrain
    # for lr in [8e-4, 5e-4]:
    #     for const in [0, 5]:
    #         retrain_args = copy.deepcopy(args)
    #         retrain_args['retrain_checkpoint'] = train_job_avg_ckpt['base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035']
    #         if const:
    #             retrain_args['learning_rates_list'] = [8e-4] * (const * 20)
    #         retrain_args['lr'] = lr
    #         retrain_args['lr_decay'] = 0.95
    #         name = f"base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035_retrain1_lr{lr}_const{const}_decay0.95"
    #         run_exp(name, train_args=retrain_args, num_epochs=400)
    #
    #         if const == 0:
    #             j = train_mini_self_att(
    #                 exp_name=name,
    #                 checkpoint=train_job_avg_ckpt[name],
    #                 args=args, num_epochs=40, lr=lr, name=f'mini_self_att_ep40'
    #             )
    #             for beam_size in [32, 40]:
    #                 for lm_type in ['lstm']:
    #                     if lm_type == 'trafo' and beam_size == 40:
    #                         run_lm_fusion(
    #                             lm_type=lm_type, exp_name=name, epoch='avg',
    #                             test_set_names=['dev-other'],
    #                             lm_scales=[0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24],
    #                             batch_size=4000 * 160 if lm_type == 'lstm' else 2000 * 160,
    #                             train_job=j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=beam_size,
    #                         )
    #
    #                     # lm - scale - 0.48 - prior - 0.3 - mini_lstm_ep40 - beam - 32 / dev - other / wer 4.55
    #                     run_lm_fusion(
    #                         lm_type=lm_type, exp_name=name, epoch='avg',
    #                         test_set_names=['dev-other'],
    #                         lm_scales=[0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54],
    #                         prior_scales=[0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44],
    #                         mini_lstm_ckpt=get_best_checkpoint(j, key='dev_score'), prior_type='mini_lstm',
    #                         prior_type_name='mini_lstm_ep40', batch_size=4000 * 160 if lm_type == 'lstm' else 2000 * 160,
    #                         train_job=j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=beam_size,
    #                     )
    #
    #
    # # TODO: SF
    # for lm_type in ['lstm']:
    #     run_lm_fusion(
    #         lm_type=lm_type, exp_name="base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035", epoch='avg',
    #         test_set_names=['dev-clean'],
    #         lm_scales=[0.14, 0.16, 0.18, 0.2, 0.22],
    #         train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=32,
    #     )
    #     for test, scale in [('test-clean', 0.26), ('test-other', 0.36)]:
    #         run_lm_fusion(
    #             lm_type=lm_type, exp_name="base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035", epoch='avg',
    #             test_set_names=[test],
    #             lm_scales=[scale],
    #             train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=32,
    #         )
    #
    # # TODO: tune LR?
    # for lr in [8e-4, 5e-4]:
    #     j = train_mini_self_att(
    #         exp_name='base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035',
    #         checkpoint=train_job_avg_ckpt['base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035'],
    #         args=args, num_epochs=40, lr=lr, name=f'mini_self_att_{lr}_40'
    #     )
    #     if lr == 8e-4:
    #         for lm_type in ['lstm', 'trafo']:
    #             run_lm_fusion(
    #                 lm_type=lm_type, exp_name="base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035", epoch='avg',
    #                 test_set_names=['dev-clean', 'dev-other'],
    #                 lm_scales=[0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.58],
    #                 prior_scales=[0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4],
    #                 mini_lstm_ckpt=get_best_checkpoint(j, key='dev_score'), prior_type='mini_lstm',
    #                 prior_type_name='mini_lstm_ep40', batch_size=4000 * 160 if lm_type == 'lstm' else 2000 * 160,
    #                 train_job=j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=32,
    #             )
    #
    # # TODO: train with warmup
    # for peak_lr in [8e-4, 5e-4]:
    #     j = train_mini_self_att(
    #         exp_name='base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035',
    #         checkpoint=train_job_avg_ckpt['base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035'],
    #         args=args, num_epochs=100, lr=peak_lr, name=f'mini_self_att_wup30_2e-4-{peak_lr}_ep100',
    #         learning_rates_list=list(numpy.linspace(2e-4, peak_lr, 20)), time_rqmt=8,
    #     )
    #     if peak_lr == 8e-4:
    #         run_lm_fusion(
    #             lm_type='lstm', exp_name="base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035", epoch='avg',
    #             test_set_names=['dev-other'],
    #             lm_scales=[0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.58],
    #             prior_scales=[0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4],
    #             mini_lstm_ckpt=get_best_checkpoint(j, key='dev_score'), prior_type='mini_lstm',
    #             prior_type_name='mini_lstm_wup30_8e-04_ep100', batch_size=4000 * 160,
    #             train_job=j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=32,
    #         )
    #
    # # TODO: train smaller self-att
    # for key_dim in [32, 64, 128, 256]:
    #     for att_heads in [2, 4, 8]:
    #         j = train_mini_self_att(
    #             exp_name='base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035',
    #             checkpoint=train_job_avg_ckpt['base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035'],
    #             args=args, num_epochs=80, name=f'mini_self_att_encK-{key_dim}-attH-{att_heads}',
    #             decoder_args={'ilm_args': {'enc_key_dim': key_dim, 'att_num_heads': att_heads}}
    #         )
    #         run_lm_fusion(
    #             lm_type='lstm', exp_name="base_conf_12l_trafo6l_conv6_sqrdReLU_OCLR_cyc465_1035", epoch='avg',
    #             test_set_names=['dev-other'],
    #             lm_scales=[0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.58],
    #             prior_scales=[0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4],
    #             mini_lstm_ckpt=get_best_checkpoint(j, key='dev_score'), prior_type='mini_lstm',
    #             prior_type_name=f'mini_self_att_encK-{key_dim}-attH-{att_heads}', batch_size=4000 * 160,
    #             train_job=j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=32,
    #             decoder_args={'ilm_args': {'enc_key_dim': key_dim, 'att_num_heads': att_heads}}
    #         )

    run_exp("base_conf_12l_trafo6l_conv6_OCLR_cyc195_435_CurrLRv3", train_args=oclr_args, num_epochs=435,
            epoch_wise_filter=[(1, 5, 500), (5, 10, 1000)])

    for model_dim in [512]:
        args = copy.deepcopy(trafo_dec_exp_args)
        args["oclr_opts"] = {
            "peak_lr": 8e-4,
            "final_lr": 1e-6,
            "cycle_ep": 195,
            "total_ep": 435,  # 20 epochs
            "n_step": 1700,
            "learning_rates": [8e-5] * 35
        }
        args['encoder_args'].input_layer = 'conv-6'
        args['encoder_args'].enc_key_dim = model_dim
        args['encoder_args'].ff_dim = model_dim * 4
        args['encoder_args'].att_num_heads = 16
        args["decoder_args"].att_num_heads = 16
        args["decoder_args"].ff_dim = model_dim * 4
        args['encoder_args'].use_sqrd_relu = True
        run_exp(
            f"base_conf_12l_trafo6l_OCLR_conv6_sqrdReLU_modelDim-{model_dim}_AttH-16",
            train_args=args, num_epochs=435
        )

    # --------------------------- LSTM Decoder --------------------------- #

    oclr_args = copy.deepcopy(lstm_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 8e-4,
        "final_lr": 1e-6,
        "cycle_ep": 195,
        "total_ep": 435,  # 20 epochs
        "n_step": 1350,
        "learning_rates": [8e-5] * 35
    }
    oclr_args["encoder_args"].input_layer = "conv-6"
    oclr_args['encoder_args'].use_sqrd_relu = True
    run_exp("base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU", train_args=oclr_args, num_epochs=435)

    #run_exp('base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_bpe1k', train_args=oclr_args, num_epochs=435, bpe_size=1000)
    #run_exp('base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_bpe5k', train_args=oclr_args, num_epochs=435, bpe_size=5000)

    # TODO: pretrain font-end?

    # TODO: longer training + more regularization
    for peak_lr in [9e-4]:
        for drop_val in [0.1, 0.2]:
            for num_epochs in [1035, 1635, 2035]:
                args = copy.deepcopy(oclr_args)
                cyc_ep = int(0.45 * num_epochs)
                args['oclr_opts']['peak_lr'] = peak_lr
                args['oclr_opts']['cycle_ep'] = cyc_ep
                args['oclr_opts']['total_ep'] = num_epochs
                args['decoder_args'].att_dropout = drop_val
                args['decoder_args'].dropout = drop_val
                train_j, train_data = run_exp(
                    f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc{cyc_ep}_ep{num_epochs}_peakLR{peak_lr}-drop{drop_val}-attDrop-{drop_val}",
                    train_args=args, num_epochs=num_epochs)

    # base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc465_ep1035_peakLR0.0009   2.4/6.0/2.6/6.0

    for num_epochs in [635, 1035]:
        for peak_lr in [8e-4, 9e-4, 1e-3]:
            if peak_lr == 8e-4 and num_epochs == 635:
                continue  # already trained
            args = copy.deepcopy(oclr_args)
            cyc_ep = int(0.45 * num_epochs)
            args['oclr_opts']['peak_lr'] = peak_lr
            args['oclr_opts']['cycle_ep'] = cyc_ep
            args['oclr_opts']['total_ep'] = num_epochs
            train_j, train_data = run_exp(
                f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc{cyc_ep}_ep{num_epochs}_peakLR{peak_lr}",
                train_args=args, num_epochs=num_epochs)

            if peak_lr == 9e-4 and num_epochs == 1035:
                name = f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc{cyc_ep}_ep{num_epochs}_peakLR{peak_lr}"
                mini_lstm_j = train_mini_lstm(
                    exp_name=name,
                    checkpoint=train_job_avg_ckpt[name],
                    args=args, num_epochs=80, w_drop=True
                )

                # Without LM:
                # base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc465_ep1035_peakLR0.0009   2.4/6.0/2.6/6.0

                for lm_type in ['lstm', 'trafo']:
                    for beam_size in [32, 40, 45, 50]:
                        run_lm_fusion(
                            lm_type=lm_type, exp_name=name, epoch='avg',
                            test_set_names=['dev-clean', 'dev-other'],
                            lm_scales=[0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52],
                            train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args,
                            beam_size=beam_size,
                        )
                        # with ILM
                        run_lm_fusion(
                            lm_type=lm_type, exp_name=name, epoch='avg',
                            test_set_names=['dev-clean'],
                            lm_scales=[0.38, 0.4, 0.42, 0.44, 0.46, 0.48],
                            prior_scales=[0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42],
                            prior_type='mini_lstm', mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key='dev_score'),
                            prior_type_name='mini_lstm_best',
                            train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args,
                            beam_size=beam_size, batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                        )
                    # if lm_type == 'trafo':
                    #     for coverage_scale in [0.2, 0.3, 0.4]:
                    #         for coverage_thre in [0.2]:
                    #             run_lm_fusion(
                    #                 lm_type=lm_type, exp_name=name, epoch='avg',
                    #                 test_set_names=['dev-other'],
                    #                 lm_scales=[0.5, 0.52, 0.54],
                    #                 prior_scales=[0.34, 0.36, 0.38],
                    #                 prior_type='mini_lstm', mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key='dev_score'),
                    #                 prior_type_name='mini_lstm_best',
                    #                 train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args,
                    #                 beam_size=45, batch_size=1000 * 160, time_rqmt=2,
                    #                 coverage_scale=coverage_scale, coverage_threshold=coverage_thre,
                    #             )

                # lstm dev-clean 0.3 beam 40 -> 2.09
                # trafo dev-clean 0.29 beam 45 -> 2.01

                # test
                #('trafo', 0.36, 'test-other'), ('trafo', 0.32, 'test-clean'),
                # ('lstm', 0.32, 'test-other'), ('lstm', 0.29, 'test-clean')
                for lm_type, lm_scale, prior_scale, beam_size in [
                    ('trafo', 0.52, 0.36, 45), ('lstm', 0.58, 0.4, 45)
                ]:
                    run_lm_fusion(
                        lm_type=lm_type, exp_name=name, epoch='avg',
                        test_set_names=['test-other'],
                        lm_scales=[lm_scale],
                        prior_scales=[prior_scale],
                        prior_type='mini_lstm', prior_type_name='mini_lstm_best',
                        mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key='dev_score'),
                        train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args,
                        beam_size=beam_size, time_rqmt=4,
                    )

                # TODO: retrain
                retrain_args = copy.deepcopy(args)
                retrain_args['retrain_checkpoint'] = train_job_avg_ckpt[name]
                retrain_args['lr_decay'] = 0.95

                # linear decay from 5e-4
                for lr in [9e-4, 5e-4]:
                    retrain_v1 = copy.deepcopy(retrain_args)
                    retrain_v1['learning_rates_list'] = list(numpy.linspace(lr, 1e-6, 600))
                    train_j, train_data = run_exp(exp_name=name + f'_retrain1_linDecay400_{lr}', train_args=retrain_v1, num_epochs=600)

                    if lr == 5e-4:
                        _name = name + f'_retrain1_linDecay400_{lr}'
                        # TODO: train less
                        # TODO: start with smaller LR
                        mini_lstm_j = train_mini_lstm(
                            exp_name=_name,
                            checkpoint=train_job_avg_ckpt[_name],
                            args=args, num_epochs=80, w_drop=True
                        )
                        for lm_type in ['lstm', 'trafo']:
                            for beam_size in [32, 40, 45, 50]:
                                run_lm_fusion(
                                    lm_type=lm_type, exp_name=_name, epoch='avg',
                                    test_set_names=['dev-clean', 'dev-other'],
                                    lm_scales=[0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48,
                                               0.5, 0.52],
                                    train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args,
                                    beam_size=beam_size,
                                )
                                # with ILM
                                run_lm_fusion(
                                    lm_type=lm_type, exp_name=_name, epoch='avg',
                                    test_set_names=['dev-other'],
                                    lm_scales=[0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6],
                                    prior_scales=[0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42],
                                    prior_type='mini_lstm',
                                    mini_lstm_ckpt=get_best_checkpoint(mini_lstm_j, key='dev_score'),
                                    prior_type_name='mini_lstm_best',
                                    train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args,
                                    beam_size=beam_size, batch_size=(1000 * 160) if beam_size > 40 else (2000 * 160),
                                )

                        # for lm_type, lm_scale, prior_scale, beam_size in [
                        #     ('trafo', 0.52, 0.36, 45), ('trafo', 0.5, 0.36, 50)
                        # ]:


                # TODO: retrain with OCLR


    # TODO: sampling preemphasis

    # TODO: speed perturbation v2

    # TODO: + volumn perturbation

    # ------------------------------------------------------- #

    args = copy.deepcopy(oclr_args)
    args['oclr_opts']['cycle_ep'] = 915
    args['oclr_opts']['total_ep'] = 2035
    run_exp("base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035", train_args=args, num_epochs=2035)

    for peak_lr in [9e-4, 1e-3]:
        args = copy.deepcopy(oclr_args)
        args['oclr_opts']['cycle_ep'] = 915
        args['oclr_opts']['total_ep'] = 2035
        args['oclr_opts']['peak_lr'] = peak_lr
        run_exp(f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc915_ep2035_peak{peak_lr}", train_args=args, num_epochs=2035)

    args = copy.deepcopy(oclr_args)
    args['oclr_opts']['cycle_ep'] = 285
    args['oclr_opts']['total_ep'] = 635
    train_j, train_data = run_exp(
        "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635", train_args=args, num_epochs=635)

    # TODO: retrain
    retrain_args = copy.deepcopy(args)
    retrain_args["retrain_checkpoint"] = train_job_avg_ckpt['base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635']
    train_j, train_data = run_exp(
        "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1",
        train_args=retrain_args, num_epochs=435, epoch_wise_filter=None)

    # for lm_type in ['lstm', 'trafo']:
    #     run_lm_fusion(
    #         lm_type=lm_type, exp_name="base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1", epoch='avg',
    #         test_set_names=['dev-clean', 'dev-other'],
    #         lm_scales=[0.0, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46],
    #         train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=32,
    #     )
    #     mini_lstm_j = train_mini_lstm(
    #         exp_name='base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1',
    #         checkpoint=train_job_avg_ckpt['base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1'],
    #         args=args, num_epochs=40, w_drop=True)
    #     mini_lstm_best_ckpt = get_best_checkpoint(mini_lstm_j, key='dev_score')
    #     # lstm other: 0.56, 0.38
    #     # lstm clean: 0.48, 0.38 | 0.48, 0.42 | 0.48, 0.4 | 0.5, 0.4
    #     # trafo other: 0.52, 0.44 | 0.52, 0.4 | 0.54, 0.42 | 0.54, 0.44 | 0.5, 0.42 | 0.5, 0.4
    #     # trafo clean: 0.48, 0.44
    #
    #     # beam size 40 seems the best.
    #     for beam_size in [40, 42, 44, 46, 48]:
    #         run_lm_fusion(
    #             lm_type='trafo', exp_name="base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1", epoch='avg',
    #             test_set_names=['dev-clean', 'dev-other'],
    #             lm_scales=[0.54, 0.56, 0.58],
    #             prior_scales=[0.42, 0.44, 0.46],
    #             prior_type='mini_lstm', mini_lstm_ckpt=mini_lstm_best_ckpt,
    #             train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=beam_size,
    #         )
    #
    #     run_lm_fusion(
    #         lm_type='trafo', exp_name="base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1", epoch='avg',
    #         test_set_names=['test-other'],
    #         lm_scales=[0.56],
    #         prior_scales=[0.44],
    #         prior_type='mini_lstm', mini_lstm_ckpt=mini_lstm_best_ckpt,
    #         train_job=train_j, train_data=train_data, feature_net=log10_net_10ms, args=args, beam_size=40,
    #     )

    # TODO: more retraining
    # for lr in [8e-4, 5e-4, 3e-4]:
    #     for lr_decay in [0.9, 0.95]:
    #         for min_lr in [None, 1e-6, 1e-5]:
    #             for const in [5]:
    #                 retrain_args = copy.deepcopy(args)
    #                 retrain_args['lr_decay'] = lr_decay
    #                 retrain_args['min_lr'] = min_lr
    #                 retrain_args['lr'] = lr
    #                 if const:
    #                     retrain_args['learning_rates_list'] = [lr] * (const * 20)
    #                 retrain_args["retrain_checkpoint"] = train_job_avg_ckpt['base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635']
    #                 name = f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_retrain1_lr{lr}_decay{lr_decay}_const{const}"
    #                 if min_lr:
    #                     name += f'_minLR{min_lr}'
    #                 run_exp(name, train_args=retrain_args, num_epochs=400, epoch_wise_filter=None)


    # ------------------------------------------------------- #

    args = copy.deepcopy(oclr_args)
    args['decoder_args'].dropout = 0.1
    args['decoder_args'].att_dropout = 0.1
    run_exp("base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_drop-0.1_attDrop-0.1", train_args=args, num_epochs=435)

    retrain_args = copy.deepcopy(args)
    retrain_args["retrain_checkpoint"] = train_job_avg_ckpt['base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_drop-0.1_attDrop-0.1']
    run_exp(
        "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_drop-0.1_attDrop-0.1_retrain1",
        train_args=retrain_args, num_epochs=435, epoch_wise_filter=None)

    args = copy.deepcopy(oclr_args)
    args['decoder_args'].dropout = 0.1
    args['decoder_args'].att_dropout = 0.1
    args['decoder_args'].att_num_heads = 8
    run_exp("base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_drop-0.1_attDrop-0.1_AttH-8", train_args=args, num_epochs=435)

    retrain_args = copy.deepcopy(args)
    retrain_args["retrain_checkpoint"] = train_job_avg_ckpt['base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_drop-0.1_attDrop-0.1_AttH-8']
    run_exp(
        "base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_drop-0.1_attDrop-0.1_AttH-8_retrain1",
        train_args=retrain_args, num_epochs=435, epoch_wise_filter=None)

    args = copy.deepcopy(oclr_args)
    args['oclr_opts']['cycle_ep'] = 285
    args['oclr_opts']['total_ep'] = 635
    args['encoder_args'].use_sqrd_relu = True
    args['decoder_args'].att_dropout = 0.1
    args['decoder_args'].dropout = 0.1
    run_exp("base_conf_12l_lstm_1l_conv6_sqrdReLU_OCLR_cyc285_635_drop-0.1_attDrop-0.1", train_args=args, num_epochs=635)

    # TODO: retrain base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009
    for lr in [8e-4, 5e-4]:
        retrain_args = copy.deepcopy(oclr_args)
        retrain_args['lr'] = lr
        retrain_args['lr_decay'] = 0.95
        retrain_args['retrain_checkpoint'] = train_job_avg_ckpt['base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009']
        run_exp(
            f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009_retrain1_lr{lr}_decay0.95",
            train_args=retrain_args, num_epochs=400)

    retrain_args = copy.deepcopy(oclr_args)
    retrain_args['learning_rates_list'] = [8e-4] * 100 + list(numpy.linspace(8e-4, 1e-6, 300))
    retrain_args['lr_decay'] = 0.95
    retrain_args['retrain_checkpoint'] = train_job_avg_ckpt[
        'base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009']
    run_exp(
        f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009_retrain1_const100_linDecay300-1e-6_decay0.95",
        train_args=retrain_args, num_epochs=400)

    for lr in [8e-4, 5e-4]:
        retrain_args = copy.deepcopy(oclr_args)
        retrain_args['learning_rates_list'] = list(numpy.linspace(lr, 1e-6, 400))
        retrain_args['lr_decay'] = 0.95
        retrain_args['retrain_checkpoint'] = train_job_avg_ckpt[
            'base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009']
        run_exp(
            f"base_conf_12l_lstm_1l_conv6_OCLR_sqrdReLU_cyc285_ep635_peakLR0.0009_retrain1_lr{lr}_linDecay400-1e-6_decay0.95",
            train_args=retrain_args, num_epochs=400)


    oclr_args = copy.deepcopy(conformer_dec_exp_args)
    oclr_args["oclr_opts"] = {
        "peak_lr": 8e-4,
        "final_lr": 1e-6,
        "cycle_ep": 195,
        "total_ep": 435,  # 20 epochs
        "n_step": 1700,
    }
    args = copy.deepcopy(oclr_args)
    args['encoder_args'].input_layer = 'conv-6'
    args['encoder_args'].use_sqrd_relu = True
    args['decoder_args'].use_sqrd_relu = True
    #run_exp('base_conf_12l_conf_6l_OCLR_sqrdReLU_cyc195_ep435', train_args=args, num_epochs=435)
