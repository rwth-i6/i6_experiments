import copy

import numpy

from sisyphus import tk

from i6_core.tools import CloneGitRepositoryJob
from i6_core.returnn import ReturnnConfig

from .pipeline import \
    build_training_datasets, build_test_dataset, training, search, get_best_checkpoint, search_single


from .config_01_transfo_decoder import create_config

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



    fairseq_ff_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"  # limit = sqrt(6 / (fan_in + fan_out))
    fairseq_mhsa_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)"  # limit = sqrt(6 * 0.5 / (fan_in + fan_out)) = sqrt(3 / (fan_in + fan_out))
    fairseq_inits = {
        'ff_init':  fairseq_ff_init,
        'mhsa_init': fairseq_mhsa_init,
        'mhsa_out_init': fairseq_ff_init,
        'conv_module_init': fairseq_ff_init
    }

    # ------------------------------------------------------------------------------------------------------------------- #

    conformer_enc_args = dict(
        enc_layers=12, input_layer='lstm-6', att_num_heads=8, ff_dim=2048, enc_key_dim=512, conv_kernel_size=32,
        pos_enc='rel', dropout=0.1, att_dropout=0.1, l2=0.0001)

    rnn_dec_args = dict(dec_layers=6, embed_dropout=0.1, label_smoothing=0.1)

    default_args = dict(**conformer_enc_args, **rnn_dec_args)

    best_args = copy.deepcopy(default_args)

    # LR scheduling
    best_args['const_lr'] = [42, 100]  # use const LR during pretraining
    best_args['wup_start_lr'] = 0.0002
    best_args['wup'] = 20

    best_args['apply_embed_weight'] = True
    best_args['ctc_loss_scale'] = 3.0 / 7
    best_args['speed_pert'] = True
    best_args['l2'] = 0.0001
    best_args['self_att_l2'] = 0.0
    best_args['dec_pos_enc'] = 'rel'

    # overwrite BN params
    best_args['batch_norm_opts'] = {
        'momentum': 0.1,
        'epsilon': 1e-3,
        'update_sample_only_in_training': False,
        'delay_sample_update': False
    }

    # pretraining
    best_args['pretrain_opts'] = {'variant': 4}
    best_args['pretrain_reps'] = 6

    # fairseq init
    best_args.update(fairseq_inits)

    # Initial experiment
    name = 'base_conformer_12l_transformer_6l'
    exp_prefix = prefix_name + "/" + name
    args = copy.deepcopy(best_args)
    args['name'] = name

    returnn_config = create_config(training_datasets=training_datasets, **args)
    train_job = training(exp_prefix, returnn_config, returnn_exe, returnn_root)
    search(exp_prefix + "/default_last", returnn_config, train_job.out_checkpoints[250], test_dataset_tuples, returnn_exe, returnn_root)
    #search(exp_prefix + "/default_best", returnn_config, get_best_checkpoint(train_job, output_path=exp_prefix), test_dataset_tuples, returnn_exe, returnn_root_search)

    ext_lm_search_args = copy.deepcopy(args)
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

