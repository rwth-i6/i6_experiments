import copy
from sisyphus import gs, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.rasr.util import RasrDataInput, RasrInitArgs

from i6_experiments.common.datasets.librispeech import get_g2p_augmented_bliss_lexicon_dict,\
    get_corpus_object_dict, get_arpa_lm_dict
from i6_experiments.users.rossenbach.lexicon.modification import AddBoundaryMarkerToLexiconJob

from .ctc_system import CtcSystem, CtcRecognitionArgs
from .ctc_network import BLSTMCTCModel, get_network, legacy_network
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
            lexicon=lexicon,
            lm=lm
        )

    test_data_inputs['test-clean'] = RasrDataInput(
        corpus_object=corpus_object_dict['test-clean'],
        concurrent=10,
        lexicon=lexicon,
        lm=lm,
    )

    return train_data_inputs, dev_data_inputs, test_data_inputs


def get_returnn_config(
        use_legacy_network=False,
        feature_dropout=False,
        stronger_specaug=False,
        use_dimtags=False,
        subsampling=2,
        dropout=0.1):
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

    if use_dimtags:
        from .ctc_network_dimtag import get_network as get_network_dimtag
        network_func = get_network_dimtag
    else:
        network_func = get_network

    prolog = get_funcs()
    if use_legacy_network:
        config['network'] = legacy_network
        return ReturnnConfig(config, post_config,
                             python_prolog=get_funcs(), hash_full_python_code=True)
    if use_dimtags:
        from returnn_common import nn
        from .ctc_network_dimtag import _map
        time_dim = nn.SpatialDim("time")
        in_dim = nn.FeatureDim("input", dimension=50)
        ext_data = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
        extern_data = {'data': ext_data}

        extern_data = {
            data_key: {
                key: getattr(data, key)
                for key in [*data.get_kwargs(include_special_axes=False).keys(), "available_for_inference"]
                if key not in {"name"}}
            for (data_key, data) in extern_data.items()}


        dim_tags_proxy = nn.ReturnnDimTagsProxy()
        ed_config = dim_tags_proxy.collect_dim_tags_and_transform_config(extern_data)
        ed_config = _map(ed_config)

        network1 = network_func(dim_tags_proxy, ext_data, time_dim, 4, 512, [1, 1, subsampling], 139, dropout=0.1, l2=0.001, specaugment_settings=None, feature_dropout=False)
        network2 = network_func(dim_tags_proxy, ext_data, time_dim, 5, 512, [1, 1, subsampling], 139, dropout=0.1, l2=0.05, specaugment_settings=None, feature_dropout=False)
        network3 = network_func(dim_tags_proxy, ext_data, time_dim, 6, 512, [1, 1, subsampling], 139, dropout=dropout, l2=0.01, specaugment_settings=None, feature_dropout=feature_dropout)
        network4 = network_func(dim_tags_proxy, ext_data, time_dim, 6, 512, [1, 1, subsampling], 139, dropout=dropout, l2=0.01, specaugment_settings=specaugment_settings, feature_dropout=feature_dropout)

        staged_network_dict = {
            1: network1,
            2: network2,
            3: network3,
            4: network4
        }

        config["extern_data"] = ed_config
        prolog.append("from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n%s" % dim_tags_proxy.py_code_str())

        if "prolog" in network1:
            config["behavior_version"] = 12

            for key, entry in network1["extern_data"].items():
                assert key in config["extern_data"]
                config["extern_data"][key]["dim_tags"] = entry["dim_tags"]

        return ReturnnConfig(config, post_config, staged_network_dict=staged_network_dict,
                             python_prolog=prolog, hash_full_python_code=True)

    else:
        network1 = network_func(4, 512, [1, 1, subsampling], 139, dropout=0.1, l2=0.001, specaugment_settings=None, feature_dropout=False)
        network2 = network_func(5, 512, [1, 1, subsampling], 139, dropout=0.1, l2=0.05, specaugment_settings=None, feature_dropout=False)
        network3 = network_func(6, 512, [1, 1, subsampling], 139, dropout=dropout, l2=0.01, specaugment_settings=None, feature_dropout=feature_dropout)
        network4 = network_func(6, 512, [1, 1, subsampling], 139, dropout=dropout, l2=0.01, specaugment_settings=specaugment_settings, feature_dropout=feature_dropout)

        staged_network_dict = {
            1: network1,
            2: network2,
            3: network3,
            4: network4
        }

        return ReturnnConfig(config, post_config, staged_network_dict=staged_network_dict,
                             python_prolog=prolog, hash_full_python_code=True)


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


def ctc_test():
    ctc_lexicon = create_regular_lexicon()
    recog_args = get_default_recog_args()
    tk.register_output("experiments/librispeech_100_ctc/ctc_lexicon.xml", ctc_lexicon)


    system = CtcSystem(
        returnn_config=get_returnn_config(),
        default_training_args=get_default_training_args(),
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", "train", "recog"))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def ctc_test_no_empty_orth():
    ctc_lexicon = create_regular_lexicon()
    recog_args = get_default_recog_args()
    tk.register_output("experiments/librispeech_100_ctc/ctc_lexicon.xml", ctc_lexicon)

    # Test with feature dropout
    system = CtcSystem(
        returnn_config=get_returnn_config(feature_dropout=True, stronger_specaug=True),
        default_training_args=get_default_training_args(),
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs(delete_empty_orth=True)

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_no_empty_orth_featdrop_v3"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", "train", "recog"))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


    # Test with feature dropout

    training_args = get_default_training_args()
    training_args = copy.deepcopy(training_args)
    training_args['keep_epochs'] = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]
    training_args['num_epochs'] = 250
    recog_args = copy.deepcopy(recog_args)
    recog_args.eval_epochs = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]
    recog_args.lm_scales = [1.1, 1.4]
    system = CtcSystem(
        returnn_config=get_returnn_config(feature_dropout=True, stronger_specaug=True, dropout=0.2),
        default_training_args=training_args,
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs(delete_empty_orth=True)

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_no_empty_orth_featdrop_v4"
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


def ctc_test_hdf():
    ctc_lexicon = create_regular_lexicon()
    recog_args = get_default_recog_args()
    tk.register_output("experiments/librispeech_100_ctc/ctc_lexicon.xml", ctc_lexicon)
    # Test with feature dropout

    training_args = get_default_training_args()
    training_args = copy.deepcopy(training_args)
    training_args['keep_epochs'] = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]
    training_args['num_epochs'] = 250
    training_args['use_hdf'] = True
    recog_args = copy.deepcopy(recog_args)
    recog_args.eval_epochs = [40, 80, 120, 160, 200, 210, 220, 230, 240, 250]
    system = CtcSystem(
        returnn_config=get_returnn_config(feature_dropout=True, stronger_specaug=True, dropout=0.2),
        default_training_args=training_args,
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs(delete_empty_orth=True)

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_hdf"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", "train", "recog"))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def ctc_test_dimtag():
    ctc_lexicon = create_regular_lexicon()
    tk.register_output("experiments/librispeech_100_ctc/ctc_lexicon.xml", ctc_lexicon)

    recog_args = get_default_recog_args()
    recog_args = copy.deepcopy(recog_args)
    training_args = copy.deepcopy(get_default_training_args())
    training_args["returnn_root"] = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                                          commit="d030cdeb573a4cbe5504bce5cd48d275a9ff5d7f").out_repository
    system = CtcSystem(
        returnn_config=get_returnn_config(use_dimtags=True),
        default_training_args=training_args,
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs(delete_empty_orth=True)

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_dimtag"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", "train", "recog"))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""


def ctc_test_legacy_network():
    ctc_lexicon = create_regular_lexicon()
    tk.register_output("experiments/librispeech_100_ctc/ctc_lexicon.xml", ctc_lexicon)
    recog_args = get_default_recog_args()

    system = CtcSystem(
        returnn_config=get_returnn_config(use_legacy_network=True),
        default_training_args=get_default_training_args(),
        recognition_args=recog_args,
        rasr_python_home='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1',
        rasr_python_exe='/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python',
    )
    train_data, dev_data, test_data = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = "experiments/librispeech/librispeech_100_ctc/ctc_test_legacy_network"
    system.init_system(
        rasr_init_args=rasr_args,
        train_data=train_data,
        dev_data=dev_data,
        test_data=test_data
    )
    system.run(("extract", "train", "recog"))
    gs.ALIAS_AND_OUTPUT_SUBDIR = ""

