__all__ = [
    "RasrDataInput",
    "RasrInitArgs",
    "GmmMonophoneArgs",
    "GmmCartArgs",
    "GmmTriphoneArgs",
    "GmmVtlnArgs",
    "GmmSatArgs",
    "GmmVtlnSatArgs",
    "ForcedAlignmentArgs",
    "ReturnnRasrDataInput",
    "NnArgs",
    "RasrSteps",
]


from collections import OrderedDict
from typing import Dict, List, Optional, Type, Union

from sisyphus import tk

import i6_core.meta as meta
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.cart.questions import BasicCartQuestions, PythonCartQuestions
from i6_core.util import MultiPath


class RasrDataInput:
    """
    this class holds the data information for a rasr gmm setup:
    - corpus
    - lexicon
    - lm
    """

    def __init__(
        self,
        corpus_object: meta.CorpusObject,
        lexicon: dict,
        lm: Optional[dict] = None,
        concurrent: int = 10,
    ):
        """
        :param corpus_object: corpus_file: Path, audio_dir: Path, audio_format: str, duration: float
        :param lexicon: file: Path, normalize_pronunciation: bool
        :param lm: filename: Path, type: str, scale: float
        :param concurrent: concurrency for gmm hmm pipeline
        """
        self.corpus_object = corpus_object
        self.lexicon = lexicon
        self.lm = lm
        self.concurrent = concurrent


# when getting an zero weights error, set:
#
# extra_config = sprint.SprintConfig()
# extra_config.allow_zero_weights = True
# {accumulate,split,align}_extra_args = {'extra_config': extra_config}
#
# '{accumulate,split,align}_extra_rqmt': {'mem': 10, 'time': 8},
#
# vtln align time = 8
#
# if not using the run function -> name and corpus almost always to be added


class RasrInitArgs:
    """
    feature extraction, AM information
    """

    def __init__(
        self,
        costa_args: dict,
        am_args: dict,
        feature_extraction_args: dict,
        default_mixture_scorer_args: dict,
        scorer: Optional[str] = None,
    ):
        """
        ##################################################
        :param costa_args: {
            'eval_recordings': True,
            'eval_lm': True
        }
        ##################################################
        :param am_args: {
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
        ##################################################
        :param feature_extraction_args:
            'mfcc': {
                'num_deriv': 2,
                'num_features': None,  # confusing name: number of max features, above number -> clipped
                'mfcc_options': {
                    'warping_function': "mel",
                    'filter_width': 268.258,  # 80
                    'normalize': True,
                    'normalization_options': None,
                    'without_samples': False,
                    'samples_options': {
                        'audio_format': "wav",
                        'dc_detection': True,
                    },
                'cepstrum_options': {
                    'normalize': False,
                    'outputs': 16,
                    'add_epsilon': False,
                },
                'fft_options': None,
                }
            }
            'gt': {
                'minfreq': 100,
                'maxfreq': 7500,
                'channels': 50,
                'warp_freqbreak': None,  # 3700
                'tempint_type': 'hanning',
                'tempint_shift': .01,
                'tempint_length': .025,
                'flush_before_gap': True,
                'do_specint': False,
                'specint_type': 'hanning',
                'specint_shift': 4,
                'specint_length': 9,
                'normalize': True,
                'preemphasis': True,
                'legacy_scaling': False,
                'without_samples': False,
                'samples_options': {
                    'audio_format': "wav",
                    'dc_detection': True
                },
                'normalization_options': {},
            }
            'fb': {
                'warping_function': "mel",
                'filter_width': 80,
                'normalize': True,
                'normalization_options': None,
                'without_samples': False,
                'samples_options': {
                    'audio_format': "wav",
                    'dc_detection': True
                },
                'fft_options': None,
                'apply_log': True,
                'add_epsilon': False,
            }
            'energy': {
                'without_samples': False,
                'samples_options': {
                    'audio_format': "wav",
                    'dc_detection': True
                    },
                'fft_options': {},
            }
        ##################################################
        :param default_mixture_scorer_args:
        {"scale": 0.3}
        ##################################################
        :param scorer:
        "kaldi", "sclite", default is sclite
        ##################################################
        """
        self.costa_args = costa_args
        self.default_mixture_scorer_args = default_mixture_scorer_args
        self.scorer = scorer
        self.am_args = am_args
        self.feature_extraction_args = feature_extraction_args


class GmmMonophoneArgs:
    def __init__(
        self,
        linear_alignment_args: dict,
        training_args: dict,
        recognition_args: dict,
        sdm_args: dict,
    ):
        """
        ##################################################
        :param linear_alignment_args: {
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
        ##################################################
        :param training_args: {
            'feature_flow': 'mfcc+deriv+norm',
            'feature_energy_flow': 'energy,mfcc+deriv+norm',
            'align_iter': 75,
            'splits': 10,
            'accs_per_split': 2,
        }
        ##################################################
        :param recognition_args: {
            'eval_iter': [7, 8, 9, 10]
            'pronunciation_scales': [1.0]
            'lm_scales': [9.0, 9.25, 9.50, 9.75, 10.0, 10.25, 10.50]
            'recog_args': {
                'feature_flow': dev_corpus_name,
                'pronunciation_scale': pronunciation_scale,
                'lm_scale': lm_scale,
                'lm_lookahead': True,
                'lookahead_options': None,
                'create_lattice': True,
                'eval_single_best': True,
                'eval_best_in_lattice': True,
                'search_parameters': {
                    'beam_pruning': 14.0,
                    'beam-pruning-limit': 100000,
                    'word-end-pruning': 0.5,
                    'word-end-pruning-limit': 15000
                },
                'best_path_algo': 'bellman-ford',  # options: bellman-ford, dijkstra
                'fill_empty_segments': False,
                'scorer': recog.Sclite,
                'scorer_args': {'ref': create_corpora.stm_files['dev-other']},
                'scorer_hyp_args': "hyp",
                'rtf': 30,
                'mem': 8,
                'use_gpu': False,
            }
        }
        ##################################################
        """
        self.linear_alignment_args = linear_alignment_args
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.sdm_args = sdm_args


class GmmCartArgs:
    def __init__(
        self,
        cart_questions: Union[Type[BasicCartQuestions], PythonCartQuestions, tk.Path],
        cart_lda_args: dict,
    ):
        """
        ##################################################
        :param cart_questions:

        BasicCartQuestions(**{
            'phoneme_path': "",
            'max_leaves': 9001
            'min_obs': 500
        })
        OR
        PythonCartQuestions(**{
            'phonemes': list[str]
            'steps': cart_steps
            'max_leaves':  9001
            'hmm_states': 3
        })
        OR
        Path("")
        :param cart_lda_args: {
            'initial_flow': 'mfcc+deriv+norm',  # is feature_flow from monophone training
            'context_flow': 'mfcc',
            'context_size': 9,
            'num_dim': 48,
            'num_iter': 2,
            'eigenvalue_args': {},
            'generalized_eigenvalue_args': {'all': {'verification_tolerance': 1e15}},
            'alignment': "train_mono",  # if using run function not needed
        }
        ##################################################
        """
        self.cart_questions = cart_questions
        self.cart_lda_args = cart_lda_args


class GmmTriphoneArgs:
    def __init__(
        self,
        training_args: dict,
        recognition_args: dict,
        sdm_args: Optional[dict] = None,
    ):
        """
        ##################################################
        :param training_args: {
            'feature_flow': 'mfcc+context+lda',
            'splits': 10,
            'accs_per_split': 2,
            'initial_alignment': "train_mono",  # if using run function not needed
        }
        ##################################################
        :param recognition_args:
        ##################################################
        :param sdm_args: {
            'feature_flow': "mfcc+context+lda",
            'alignment': "train_tri",   # if using run function not needed
        }
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.sdm_args = sdm_args


class GmmVtlnArgs:
    def __init__(
        self,
        training_args: dict,
        recognition_args: dict,
        sdm_args: Optional[dict] = None,
    ):
        """
        ##################################################
        :param training_args: {
            'feature_flow': {
                'base_flow': 'uncached_mfcc',
                'context_size': 9,
                'lda_matrix': "{corpus_name}_{mono}"  # if using run function not needed
            }
            'warp_mix': {
                'splits': 8,
                'accs_per_split': 2,
                'feature_scorer': "estimate_mixtures_sdm.tri"  # if using run function not needed
            }
            'train': {
                'splits': 10,
                'accs_per_split': 2,
                'initial_alignment': "train_tri",  # if using run function not needed
                'feature_flow': "mfcc+context+lda+vtln",
            }
        ##################################################
        :param recognition_args:
        ##################################################
        :param sdm_args: {
            'feature_flow': "mfcc+context+lda+vtln",
            'alignment': "train_vtln",  # if using run function not needed
        }
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.sdm_args = sdm_args


class GmmSatArgs:
    def __init__(
        self,
        training_args: dict,
        recognition_args: dict,
        sdm_args: Optional[dict] = None,
    ):
        """
        ##################################################
        :param training_args: {
            'feature_cache': 'mfcc+context+lda',
            'cache_regex': '^mfcc.*$',
            'splits': 10,
            'accs_per_split': 2,
            'mixtures': "estimate_mixtures_sdm.tri",  # if using run function not needed
            'align_keep_values': {7: tk.gs.JOB_DEFAULT_KEEP_VALUE},
            'feature_flow': "mfcc+context+lda",
            'alignment': "train_tri",   # if using run function not needed
        }
        ##################################################
        :param recognition_args:
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.sdm_args = sdm_args


class GmmVtlnSatArgs:
    def __init__(
        self,
        training_args: dict,
        recognition_args: dict,
        sdm_args: Optional[dict] = None,
    ):
        """
        ##################################################
        :param training_args: {
            'feature_cache': "mfcc+context+lda+vtln",
            'cache_regex': '^.*\\+vtln$',
            'mixtures': "estimate_mixtures_sdm.vtln"  # if using run function not needed
            'splits': 10,
            'accs_per_split': 2,
            'feature_flow': "mfcc+context+lda+vtln",
            'alignment': "train_vtln",  # if using run function not needed
        }
        ##################################################
        :param recognition_args:
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.sdm_args = sdm_args


class ForcedAlignmentArgs:
    def __init__(self, name, target_corpus_key, flow, feature_scorer):
        self.name = name
        self.target_corpus_key = target_corpus_key
        self.flow = flow
        self.feature_scorer = feature_scorer


class ReturnnRasrDataInput(RasrDataInput):
    def __init__(
        self,
        segment_path: Optional[Union[tk.Path, str]],
        cart_tree: Optional[Union[tk.Path, str]],
        alignments: Optional[
            Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute]
        ],
        features: Optional[
            Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute]
        ],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.segment_path = segment_path
        self.cart_tree = cart_tree
        self.alignments = alignments
        self.features = features


class NnArgs:
    def __init__(
        self,
        returnn_configs: Dict[str, returnn.ReturnnConfig],
        training_args: Optional[Dict] = None,
        recognition_args: Optional[Dict[str, Dict]] = None,
        rescoring_args: Optional[Dict[str, Dict]] = None,
    ):
        """
        ##################################################
        :param training_args:
        ##################################################
        :param recognition_args:
        ##################################################
        :param rescoring_args:
        ##################################################
        """
        self.returnn_configs = returnn_configs
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.rescoring_args = rescoring_args


class RasrSteps:
    def __init__(self):
        self._step_names_args = OrderedDict()

    def add_step(self, name, arg):
        self._step_names_args[name] = arg

    def get_step_iter(self):
        return self._step_names_args.items()

    def get_step_names_as_list(self):
        return list(self._step_names_args.keys())

    def get_args_via_idx(self, idx):
        return list(self._step_names_args.values())[idx]
