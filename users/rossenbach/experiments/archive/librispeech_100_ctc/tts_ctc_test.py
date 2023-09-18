import copy
import numpy
from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.rasr.util import RasrDataInput, RasrInitArgs

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict, \
    get_corpus_object_dict, get_arpa_lm_dict
from i6_experiments.users.rossenbach.lexicon.modification import AddBoundaryMarkerToLexiconJob

from .ctc_system import CtcSystem, CtcRecognitionArgs
from .hacky_tts_ctc_system import HackyTTSCTCSystem
from .ctc_network import BLSTMCTCModel, get_network, legacy_network, get_ctctts_network
from .specaugment_clean_v2 import SpecAugmentSettings, get_funcs

from i6_experiments.users.rossenbach.lexicon.modification import EnsureSilenceFirst


def create_regular_lexicon(delete_empty_orth=False):
    ls100_bliss_lexicon = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=True, add_unknown_phoneme_and_mapping=False)['train-clean-100']
    ls100_bliss_lexicon = EnsureSilenceFirst(ls100_bliss_lexicon, delete_empty_orth=delete_empty_orth).out_lexicon
    return ls100_bliss_lexicon


def create_eow_lexicon(delete_empty_orth=False):
    ls100_bliss_lexicon = get_g2p_augmented_bliss_lexicon_dict(use_stress_marker=True, add_unknown_phoneme_and_mapping=False)['train-clean-100']
    add_boundary_marker_job = AddBoundaryMarkerToLexiconJob(
        bliss_lexicon=ls100_bliss_lexicon,
        add_eow=True,
        add_sow=False
    )
    ls100_eow_bliss_lexicon = add_boundary_marker_job.out_lexicon
    ls100_eow_bliss_lexicon = EnsureSilenceFirst(ls100_eow_bliss_lexicon, delete_empty_orth=delete_empty_orth).out_lexicon
    return ls100_eow_bliss_lexicon


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
    default_mixture_scorer_args={"scale": 0.3}  # TODO is this needed for ctc?
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


def get_corpus_data_inputs(delete_empty_orth=False):
    """

    :return:
    """

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", output_prefix="corpora")

    lm = {
        "filename": get_arpa_lm_dict()['4gram'],
        'type': "ARPA",
        'scale': 10,
    }
    lexicon = {
        'filename': create_regular_lexicon(delete_empty_orth=delete_empty_orth),
        'normalize_pronunciation': False,
    }

    eow_lexicon = {
        'filename': create_regular_lexicon(delete_empty_orth=delete_empty_orth),
        'normalize_pronunciation': False,
    }


    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs['train-clean-100'] = RasrDataInput(
        corpus_object=corpus_object_dict['train-clean-100'],
        concurrent=10,
        lexicon=lexicon,
        lm=lm,
    )

    for dev_key in ['dev-clean', 'dev-other']:
        dev_data_inputs[dev_key] = RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=10,
            lexicon=eow_lexicon,
            lm=lm
        )

    test_data_inputs['test-clean'] = RasrDataInput(
        corpus_object=corpus_object_dict['test-clean'],
        concurrent=10,
        lexicon=eow_lexicon,
        lm=lm,
    )

    return train_data_inputs, dev_data_inputs, test_data_inputs


def get_returnn_config(
        use_legacy_network=False,
        feature_dropout=False,
        stronger_specaug=False,
        use_tts=0,
        dropout=0.1):
    config = {
        'batch_size' : 20000,
        'max_seqs'   : 128,
        'batching'   : 'random',

        # optimization #
        'learning_rates'     : list(numpy.linspace(0.0001, 0.001, num=10)) + [0.001]*20,
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

    post_config = {
        'use_tensorflow': True,
        'tf_log_memory_usage': True,
        'cleanup_old_models': True,
        'log_batch_size': True,
        'debug_print_layer_output_template': True,
    }

    specaugment_settings = SpecAugmentSettings(
        min_frame_masks=0,
        max_mask_each_n_frames=22 if stronger_specaug else 200,
        max_frames_per_mask=15 if stronger_specaug else 5,
        min_feature_masks=0,
        max_feature_masks=1,
        max_features_per_mask=5
    )

    if use_tts:
        network_func = get_ctctts_network
        config['extern_data']['speaker_name'] = {'shape': (None,), 'available_for_inference': False, 'dim': 251, 'sparse': True}
    else:
        network_func = get_network

    if use_legacy_network:
        config['network'] = legacy_network
        return ReturnnConfig(config, post_config,
                             python_prolog=get_funcs(), hash_full_python_code=True)
    else:
        network1 = network_func(4, 512, [1, 1, 1], 139, dropout=0.1, l2=0.001, specaugment_settings=None, feature_dropout=False, tts_loss_scale=0.0)
        network2 = network_func(4, 512, [1, 1, 1], 139, dropout=0.1, l2=0.05, specaugment_settings=None, feature_dropout=False, tts_loss_scale=0.0)
        network3 = network_func(4, 512, [1, 1, 1], 139, dropout=dropout, l2=0.01, specaugment_settings=None, feature_dropout=feature_dropout, tts_loss_scale=0.0)
        network4 = network_func(4, 512, [1, 1, 1], 139, dropout=dropout, l2=0.01, specaugment_settings=specaugment_settings, feature_dropout=feature_dropout, tts_loss_scale=0.0)

        staged_network_dict = {
            1: network1,
            3: network2,
            5: network3,
            7: network4
        }

        if use_tts:
            network5 = network_func(4, 512, [1, 1, 1], 139, dropout=dropout, l2=0.01, specaugment_settings=specaugment_settings, feature_dropout=feature_dropout, tts_loss_scale=0.1)
            network6 = network_func(4, 512, [1, 1, 1], 139, dropout=dropout, l2=0.01, specaugment_settings=specaugment_settings, feature_dropout=feature_dropout, tts_loss_scale=use_tts)
            staged_network_dict[9] = network5
            staged_network_dict[10] = network6

        return ReturnnConfig(config, post_config, staged_network_dict=staged_network_dict,
                             python_prolog=get_funcs(), hash_full_python_code=True)


def get_default_training_args():
    returnn_exe = tk.Path("/u/rossenbach/bin/returnn_tf2.3_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="7efd41f470c74fe70fc75bec1383bca6da81fbc1").out_repository
    train_args  = {
        'partition_epochs'   : {'train': 3, 'dev': 1},
        'num_epochs'         : 200,
        # only keep every n epoch (RETURNN) #
        'save_interval'      : 1,
        # additional clean up (Sisyphus) Best epoch is always kept #
        'keep_epochs'        : [40, 80, 120, 160, 170, 180, 190, 200],
        'device'             : 'gpu',
        'time_rqmt'          : 168, # maximum one week
        'mem_rqmt'           : 15,
        'cpu_rqmt'           : 4,
        #'qsub_rqmt'          : '-l qname=!*980*',
        'log_verbosity'      : 5,
        'use_python_control' : True,
        'returnn_python_exe': returnn_exe,
        'returnn_root': returnn_root,
    }
    return train_args



def ctc_test_speaker_loss():
    ctc_lexicon = create_regular_lexicon()
    recog_args = get_default_recog_args()
    tk.register_output("experiments/librispeech_100_ctc/ctc_lexicon.xml", ctc_lexicon)

    # common training and recog args
    training_args = get_default_training_args()
    training_args = copy.deepcopy(training_args)
    training_args['keep_epochs'] = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]
    training_args['num_epochs'] = 250
    recog_args = copy.deepcopy(recog_args)
    recog_args.eval_epochs = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]

    # baseline with subsampling 0:
    system = CtcSystem(
        returnn_config=get_returnn_config(feature_dropout=True, stronger_specaug=True, dropout=0.1),
        default_training_args=training_args,
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs(delete_empty_orth=True)

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_nosub"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", "train", "recog"))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""

    # Test with feature dropout
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        commit="6dc85907ee92a874973c01eee2219abf6a21d853").out_repository

    for tts_scale in [0.5, 1.0, 5.0, 10.0]:
        training_args = copy.deepcopy(training_args)
        training_args['add_speaker_map'] = True
        training_args['returnn_root'] = returnn_root
        recog_args = copy.deepcopy(recog_args)
        recog_args.compile_exec = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh")
        recog_args.blas_lib = tk.Path("/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so")
        recog_args.eval_epochs = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]
        system = HackyTTSCTCSystem(
            returnn_config=get_returnn_config(feature_dropout=True, stronger_specaug=True, dropout=0.1, use_tts=tts_scale),
            default_training_args=training_args,
            recognition_args=recog_args,
            rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
            rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
        )
        train_data, dev_data, test_data = get_corpus_data_inputs(delete_empty_orth=True)

        gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_speaker_scale_%.1f" % tts_scale
        system.init_system(
            rasr_init_args=rasr_args,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data
        )
        system.run(("extract", "train", "recog"))

        test_align_args = {
            'label_unit'      : 'phoneme',
            'label_tree_args' : { 'skip_silence'   : True, # no silence in tree
                                  'lexicon_config' : {'filename': create_regular_lexicon(delete_empty_orth=True),
                                                      'normalize_pronunciation': False,} # adjust eow-monophone
                                  },
            'label_scorer_type': 'precomputed-log-posterior',
            'label_scorer_args' : { 'scale'      : 1.0,
                                    'usePrior'   : True,
                                    'priorScale' : 0.5,
                                    'extraArgs'  : {'blank-label-index' : 0,
                                                    'reduction_factors': 2,}
                                    },

            "register_output": True,
        }

        system.nn_align(
            "align",
            "train-clean-100",
            flow="gt",
            tf_checkpoint=system.tf_checkpoints["default"][250],
            pronunciation_scale=1.0,
            alignment_options={
                'label-pruning'      : 50,
                'label-pruning-limit': 100000
            },
            **test_align_args
        )
        gs.ALIAS_AND_OUTPUT_SUBDIR = ""

