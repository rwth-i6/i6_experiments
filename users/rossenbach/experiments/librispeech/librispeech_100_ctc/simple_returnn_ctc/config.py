from sisyphus import tk

from i6_experiments.common.setups.rasr.util import RasrInitArgs

from .ctc_system import CtcRecognitionArgs


def get_legacy_network():
    legacy_network = {
        'bwd_lstm_1': {'L2': 0.01, 'class': 'rec', 'direction': -1, 'dropout': 0.1, 'from': 'source', 'n_out': 512, 'unit': 'nativelstm2'},
        'bwd_lstm_2': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': -1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_1', 'bwd_lstm_1'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'bwd_lstm_3': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': -1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_2', 'bwd_lstm_2'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'bwd_lstm_4': {'L2': 0.01, 'class': 'rec', 'direction': -1, 'dropout': 0.1, 'from': 'max_pool_3', 'n_out': 512, 'unit': 'nativelstm2'},
        'bwd_lstm_5': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': -1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_4', 'bwd_lstm_4'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'bwd_lstm_6': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': -1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_5', 'bwd_lstm_5'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'fwd_lstm_1': {'L2': 0.01, 'class': 'rec', 'direction': 1, 'dropout': 0.1, 'from': 'source', 'n_out': 512, 'unit': 'nativelstm2'},
        'fwd_lstm_2': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': 1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_1', 'bwd_lstm_1'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'fwd_lstm_3': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': 1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_2', 'bwd_lstm_2'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'fwd_lstm_4': {'L2': 0.01, 'class': 'rec', 'direction': 1, 'dropout': 0.1, 'from': 'max_pool_3', 'n_out': 512, 'unit': 'nativelstm2'},
        'fwd_lstm_5': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': 1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_4', 'bwd_lstm_4'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'fwd_lstm_6': { 'L2': 0.01,
                        'class': 'rec',
                        'direction': 1,
                        'dropout': 0.1,
                        'from': ['fwd_lstm_5', 'bwd_lstm_5'],
                        'n_out': 512,
                        'unit': 'nativelstm2'},
        'max_pool_3': {'class': 'pool', 'from': ['fwd_lstm_3', 'bwd_lstm_3'], 'mode': 'max', 'padding': 'same', 'pool_size': (2,), 'trainable': False},
        'output': { 'class': 'softmax',
                    'from': ['fwd_lstm_6', 'bwd_lstm_6'],
                    'n_out': 139,
                    'target': None},
        'log_output':{'class': 'activation', 'from': 'output_0', 'activation': 'log'},
        #'source': {'class': 'eval', 'eval': "self.network.get_config().typed_value('_specaugment_eval_func')(source(0, as_data=True), network=self.network)"}
        'source': {'class': 'copy', 'from': ["data"]}
    }
    return legacy_network


config = {
    'batch_size' : 20000,
    'max_seqs'   : 128,
    'batching'   : 'random',

    # optimization #
    'learning_rates'     : [0.001]*30,
    'gradient_clip'     : 0,
    'gradient_noise'    : 0.1, # together with l2 and dropout for overfit

    # Note: (default 1e-8) likely not too much impact
    'optimizer': {'class': 'nadam', 'epsilon': 1e-8},

    # let it stop and adjust in time
    # Note: for inf or nan, sth. is too big (e.g. lr warm up)
    'stop_on_nonfinite_train_score' : False,

    'learning_rate_control'        : 'newbob_multi_epoch',
    'newbob_multi_num_epochs'      : 3,
    'newbob_multi_update_interval' : 1,
    'newbob_learning_rate_decay'   : 0.9,

    'learning_rate_control_relative_error_relative_lr' : True,
    'learning_rate_control_min_num_epochs_per_new_lr'  : 3,

    'extern_data': {'data': {'shape': (None, 50), 'available_for_inference': True, 'dim': 50}},
}


rasr_args = RasrInitArgs(
    costa_args={
        'eval_recordings': True,
        'eval_lm': False
    },
    am_args={
        'states_per_phone'   : 1, # single state
        'state_tying'        : 'monophone-eow', # different class for final phoneme of a word
        'tdp_transition'     : (0, 0, 'infinity', 0), # skip on infinity
        'tdp_silence'        : (0, 0, 'infinity', 0),
    },
    feature_extraction_args={
        'gt': {
            'gt_options': {
                'minfreq'   : 100,
                'maxfreq'   : 7500,
                'channels'  : 50,
                'do_specint': False
            }
        }
    },
)


def get_default_recog_args():
    recog_args = CtcRecognitionArgs(
        eval_epochs=[40, 80, 120, 160, 180, 200],
        lm_scales=[1.1],
        recog_args={
            'feature_flow': 'gt',
            'lm_lookahead': True, # use lookahead, using the lm for pruning partial words
            'lookahead_options': {
                'history-limit': 1,
                'cache-size-low': 2000,
                'cache-size-high': 3000,
                'scale': None, # use lm scale also for lookahead
            }, # the lookahead rasr options
            #'create_lattice': True, # write lattice cache files
            'eval_single_best': True, # show the evaluation of the best path in lattice in the log (model score)
            'eval_best_in_lattice': True, # show the evaluation of the best path in lattice in the log (oracle)
            #'best_path_algo': 'bellman-ford',  # options: bellman-ford, dijkstra
            #'fill_empty_segments': False, # insert dummy when transcription output is empty
            'rtf': 30, # time estimation for jobs
            'mem': 8, # memory for jobs
            'use_gpu': False, # True makes no sense
            'label_unit'      : 'phoneme',
            'label_tree_args' : { 'skip_silence'   : True, # no silence in tree
                                  'lexicon_config' : {'filename': create_eow_lexicon(delete_empty_orth=True),
                                                      'normalize_pronunciation': False,} # adjust eow-monophone
                                  },
            'label_scorer_type': 'precomputed-log-posterior',
            'label_scorer_args' : { 'scale'      : 1.0,
                                    'usePrior'   : True,
                                    'priorScale' : 0.5,
                                    'extraArgs'  : {'blank-label-index' : 0,
                                                    'reduction_factors': 2,}
                                    },
            'lm_gc_job_mem': 16,
        },
        search_parameters={
            'label-pruning': 16,
            'label-pruning-limit': 20000,
            'word-end-pruning': 0.5,
            'word-end-pruning-limit': 20000,
            # keep alternative paths in the lattice or not
            'create-lattice': True,
            'optimize-lattice': False,
        }
    )
    return recog_args