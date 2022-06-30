import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, search_single


from .attention_asr_config import create_config, ConformerEncoderArgs, TransformerDecoderArgs, RNNDecoderArgs

from .zeineldeen_helpers.models.lm.transformer_lm import TransformerLM

def conformer_baseline():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root_datasets = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="c2b31983bfc1049c7ce1549d7ffed153c9d4a443").out_repository
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                commit="d92b4701d3730040cc6834ef20f5398a6347187d").out_repository

    prefix_name = "experiments/librispeech/librispeech_100_attention/conformer_2022"

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    training_datasets = build_training_datasets(returnn_exe, returnn_root_datasets, prefix_name, bpe_size=2000)

    # build testing datasets
    test_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other", "test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(testset, returnn_python_exe=returnn_exe, returnn_root=returnn_root_datasets, output_path=prefix_name)

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

    conformer_enc_args.ctc_loss_scale = 3.0 / 7

    trafo_dec_args = TransformerDecoderArgs(
        num_layers=6, embed_dropout=0.1, label_smoothing=0.1,
        apply_embed_weight=True,
        pos_enc='rel',
        ff_init=fairseq_ff_init,
        mhsa_init=fairseq_mhsa_init,
        mhsa_out_init=fairseq_ff_init
    )

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
    training_args['pretrain_opts'] = {'variant': 4}
    training_args['pretrain_reps'] = 6

    # ---------------------------------------------------------
    # LM Settings
    transf_lm_net = TransformerLM(
        source='prev:output', num_layers=24, vocab_size=2051, use_as_ext_lm=True, prefix_name='lm_')
    transf_lm_net.create_network()
    transf_lm_opts = {
        'lm_subnet': transf_lm_net.network.get_net(),
        'lm_output_prob_name': 'lm_output',
        'is_recurrent': True,
        'preload_from_files': {
            'lm_model': {
                'filename': '/work/asr4/zeineldeen/setups-data/librispeech/2021-02-21--lm-bpe/dependencies/lm_models/transf/epoch.016',
                'prefix': 'lm_'
            }
        },
        'name': 'trafo',
    }

    # ---------------------------------------------------------
    # Initial experiment
    name = 'base_conformer_12l_transformer_6l'
    exp_prefix = prefix_name + "/" + name

    args = copy.deepcopy({**training_args, "encoder_args": conformer_enc_args, "decoder_args": trafo_dec_args})
    args['name'] = name

    returnn_config = create_config(training_datasets=training_datasets, **args)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

    # returnn_config = create_config(training_datasets=training_datasets, **args)
    # returnn_config.config["store_tf_profile"] = True
    # search_single(exp_prefix + "/profile",
    #               returnn_config,
    #               train_job.out_checkpoints[250],
    #               test_dataset_tuples["dev-other"][0],
    #               test_dataset_tuples["dev-other"][1],
    #               returnn_exe,
    #               returnn_root,
    #               mem_rqmt=16)

    ext_lm_search_args = copy.deepcopy(args)
    ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

    for lm_scale in [0.2, 0.32, 0.4]:
        search_args = copy.deepcopy(ext_lm_search_args)
        search_args['ext_lm_opts']['lm_scale'] = lm_scale
        returnn_config = create_config(training_datasets=training_datasets, **search_args)
        search_single(exp_prefix + "/default_last_ext_lm_%.2f" % lm_scale,
                      returnn_config,
                      train_job.out_checkpoints[250],
                      test_dataset_tuples["dev-other"][0],
                      test_dataset_tuples["dev-other"][1],
                      returnn_exe,
                      returnn_root)

    for lm_scale in [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]:
        search_args = copy.deepcopy(ext_lm_search_args)
        search_args['ext_lm_opts']['lm_scale'] = lm_scale
        search_args['beam_size'] = 32
        search_args['batch_size'] = 4000
        returnn_config = create_config(training_datasets=training_datasets, **search_args)
        search_single(exp_prefix + "/default_last_ext_lm_%.2f_bs32" % lm_scale,
                      returnn_config,
                      train_job.out_checkpoints[250],
                      test_dataset_tuples["dev-other"][0],
                      test_dataset_tuples["dev-other"][1],
                      returnn_exe,
                      returnn_root)
        #search(exp_prefix + "/default_last_ext_lm_%.1f" % lm_scale, returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)

        # ---------------------------------------------------------

    # Initial experiment
    name = 'base_conformer_12l_lstm_1l'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_conformer_enc_args.ctc_loss_scale = 1.0
    training_args = copy.deepcopy(training_args)

    # pretraining
    training_args['pretrain_opts'] = {'variant': 3}
    training_args['pretrain_reps'] = 5

    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy({**training_args, "encoder_args": local_conformer_enc_args, "decoder_args": rnn_dec_args})
    args['name'] = name
    args['with_staged_network'] = True
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="cc7d1ab95560910c92b5a43e4f2717370c796154").out_repository


    returnn_config = create_config(training_datasets=training_datasets, **args)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

    ext_lm_search_args = copy.deepcopy(args)
    ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

    for lm_scale in [0.2, 0.32, 0.4]:
        search_args = copy.deepcopy(ext_lm_search_args)
        search_args['ext_lm_opts']['lm_scale'] = lm_scale
        returnn_config = create_config(training_datasets=training_datasets, **search_args)
        search_single(exp_prefix + "/default_last_ext_lm_%.2f" % lm_scale,
                      returnn_config,
                      train_job.out_checkpoints[250],
                      test_dataset_tuples["dev-other"][0],
                      test_dataset_tuples["dev-other"][1],
                      returnn_exe,
                      returnn_root)

    #for lm_scale in [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]:
    #    search_args = copy.deepcopy(ext_lm_search_args)
    #    search_args['ext_lm_opts']['lm_scale'] = lm_scale
    #    search_args['beam_size'] = 32
    #    search_args['batch_size'] = 4000
    #    returnn_config = create_config(training_datasets=training_datasets, **search_args)
    #    search_single(exp_prefix + "/default_last_ext_lm_%.2f_bs32" % lm_scale,
    #                  returnn_config,
    #                  train_job.out_checkpoints[250],
    #                  test_dataset_tuples["dev-other"][0],
    #                  test_dataset_tuples["dev-other"][1],
    #                  returnn_exe,
    #                  returnn_root)


    # timing experiment
    name = 'base_conformer_12l_convsub_lstm_1l'
    local_conformer_enc_args = copy.deepcopy(conformer_enc_args)
    local_conformer_enc_args.input_layer = "conv"
    local_conformer_enc_args.ctc_loss_scale = 1.0
    training_args = copy.deepcopy(training_args)

    # pretraining
    training_args['pretrain_opts'] = {'variant': 3}
    training_args['pretrain_reps'] = 5

    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy({**training_args, "encoder_args": local_conformer_enc_args, "decoder_args": rnn_dec_args})
    args['name'] = name
    args['with_staged_network'] = True
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="cc7d1ab95560910c92b5a43e4f2717370c796154").out_repository


    returnn_config = create_config(training_datasets=training_datasets, **args)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_80", returnn_config, train_job.out_checkpoints[40], test_dataset_tuples, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)

    ext_lm_search_args = copy.deepcopy(args)
    ext_lm_search_args["ext_lm_opts"] = transf_lm_opts

    for lm_scale in [0.4]:
        search_args = copy.deepcopy(ext_lm_search_args)
        search_args['ext_lm_opts']['lm_scale'] = lm_scale
        returnn_config = create_config(training_datasets=training_datasets, **search_args)
        search_single(exp_prefix + "/default_last_ext_lm_%.2f" % lm_scale,
                      returnn_config,
                      train_job.out_checkpoints[250],
                      test_dataset_tuples["dev-other"][0],
                      test_dataset_tuples["dev-other"][1],
                      returnn_exe,
                      returnn_root)