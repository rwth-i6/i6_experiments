import copy
import time

from sisyphus import gs, tk, Path

from i6_core.features.filterbank import filter_width_from_channels
from i6_core.corpus.stats import ExtractOovWordsFromCorpusJob

from i6_experiments.common.setups.hybrid.util import RasrInitArgs, RasrDataInput
from i6_experiments.common.setups.hybrid import gmm_system
from i6_experiments.common.datasets.librispeech import get_corpus_object_dict, get_bliss_lexicon, get_arpa_lm_dict


def get_init_args():
    """
    :return:
    :rtype: hybrid.GmmInitArgs
    """
    am_args = {
        'state_tying': "monophone",
        'states_per_phone': 3,
        'state_repetitions': 1,
        'across_word_model': True,
        'early_recombination': False,
        'tdp_scale': 1.0,
        'tdp_transition': (3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit
        'tdp_silence': (0.0, 3.0, "infinity", 20.0),
        'tying_type': "global",
        'nonword_phones': "",
        'tdp_nonword': (0.0, 3.0, "infinity", 6.0)  # only used when tying_type = global-and-nonword
    }

    costa_args = {
        'eval_recordings': True,
        'eval_lm': True
    }

    feature_extraction_args = {
        'mfcc': {
            'num_deriv': 2,
            'num_features': None,  # confusing name: number of max features, above number -> clipped
            'mfcc_options': {
                'warping_function': "mel",
                'filter_width': filter_width_from_channels(channels=21, warping_function="mel", f_max=8000), # 21 is legacy behavior
                'normalize': True,
                'normalization_options': None,
                'without_samples': False,
                'samples_options': {
                    'audio_format': "wav",
                    'dc_detection': True,
                },
                'cepstrum_options': {
                    'normalize': False,
                    'outputs': 16, # this is the actual output feature dimension
                    'add_epsilon': False,
                },
                'fft_options': None,
            }
        },
        'energy': {
            'energy_options': {
                'without_samples': False,
                'samples_options': {
                    'audio_format': "wav",
                    'dc_detection': True
                },
                'fft_options': {},
            }
        }
    }

    return RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args)


def get_monophone_args():
    linear_alignment_args = {
        'minimum_segment_length': 0,
        'maximum_segment_length': 6000,
        'iterations': 20,
        'penalty': 0,
        'minimum_speech_proportion': .7,
        'save_alignment': False,
        'keep_accumulators': False,
        'extra_merge_args': None,
        'extra_config': None,
        'extra_post_config': None,
    }

    monophone_training_args = {
        'feature_flow': 'mfcc+deriv', # +norm
        'feature_energy_flow': 'energy,mfcc+deriv', # +norm
        'align_iter': 75,
        'splits': 10,
        'accs_per_split': 2,
    }

    monophone_recognition_args = {
        # GmmSystem.recognition() args:
        'iters': [10],
        'lm_scales': [10],
        'optimize_am_lm_scale': True,
        # meta.System.recog() args:
        'feature_flow': 'mfcc+deriv', # +norm
        'pronunciation_scale': 1.0,
        'lm_lookahead': True,
        'lookahead_options': None,
        'create_lattice': True,
        'eval_single_best': True,
        'eval_best_in_lattice': True,
        'search_parameters': {
            'beam-pruning': 14.0,
            'beam-pruning-limit': 100000,
            'word-end-pruning': 0.5,
            'word-end-pruning-limit': 15000
        },
        'parallelize_conversion': False,
        'lattice_to_ctm_kwargs': {},
        'rtf': 10,
        'mem': 8,
        'use_gpu': False,
    }

    return gmm_system.GmmMonophoneArgs(
        linear_alignment_args,
        monophone_training_args,
        monophone_recognition_args
    )


def get_corpus_data_inputs():
    """

    :return:
    """

    corpus_object_dict = get_corpus_object_dict(audio_format="wav", subdir_prefix="corpora")

    lm = {
        "filename": get_arpa_lm_dict()['4gram'],
        'type': "ARPA",
        'scale': 10,
    }
    lexicon = {
        'filename': get_bliss_lexicon(),
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


def run_baseline_training():

    train, dev, test = get_corpus_data_inputs()

    gs.ALIAS_AND_OUTPUT_SUBDIR = 'GMM-Mono-New-Baseline'
    system = gmm_system.GmmSystem()
    start = time.time()
    system.init_system(hybrid_init_args=get_init_args(),
                       gmm_monophone_args=get_monophone_args(),
                       gmm_triphone_args=None,
                       gmm_vtln_args=None,
                       gmm_sat_args=None,
                       gmm_vtln_sat_args=None,
                       train_data=train,
                       dev_data=dev,
                       test_data=test)
    print("init_system took: %.1f" % (time.time()-start))
    start = time.time()
    system.run(["extract", "mono"])
    print("run took: %.1f" % (time.time()-start))


    # gs.ALIAS_AND_OUTPUT_SUBDIR = 'GMM-Mono-Old-Baseline'
    # system = gmm_system.GmmSystem()
    # mono_args = get_monophone_args()
    # mono_args.monophone_recognition_args['lm_scale'] = 7.0
    # mono_args.monophone_recognition_args['search_parameters']['beam-pruning'] = 16.0
    # mono_args.monophone_recognition_args['rtf'] = 20
    # system.init_system(hybrid_init_args=get_init_args(),
    #                gmm_monophone_args=mono_args,
    #                gmm_triphone_args=None,
    #                gmm_vtln_args=None,
    #                gmm_sat_args=None,
    #                gmm_vtln_sat_args=None,
    #                train_data=train,
    #                dev_data=dev,
    #                test_data=test)
    # system.run(["extract", "mono"])
    gs.ALIAS_AND_OUTPUT_SUBDIR = ''

