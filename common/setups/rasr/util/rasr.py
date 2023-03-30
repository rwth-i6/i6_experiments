__all__ = [
    "RasrDataInput",
    "RasrInitArgs",
    "RescoreArgs",
    "RasrSteps",
]


from collections import OrderedDict
from typing import Dict, Optional
from sisyphus import tk

import i6_core.meta as meta


class RasrDataInput:
    """
    this class holds the data information for a rasr gmm setup:
    - corpus object
    - lexicon: file, normalization
    - lm: file, scale, type
    - concurrency
    """

    def __init__(
        self,
        corpus_object: meta.CorpusObject,
        lexicon: dict,
        lm: Optional[dict] = None,
        concurrent: int = 10,
        stm: Optional[tk.Path] = None,
        glm: Optional[tk.Path] = None,
    ):
        """
        :param corpus_object: corpus_file: Path, audio_dir: Path, audio_format: str, duration: float
        :param lexicon: file: Path, normalize_pronunciation: bool
        :param lm: filename: Path, type: str, scale: float
        :param concurrent: concurrency for gmm hmm pipeline
        :param stm: optional stm file for evaluation
        :param glm: optional glm file for evaluation
        """
        self.corpus_object = corpus_object
        self.lexicon = lexicon
        self.lm = lm
        self.concurrent = concurrent
        self.stm = stm
        self.glm = glm


class RasrInitArgs:
    """
    Class holds general information for the complete pipeline.
    These values can/will change during the training process.

    - acoustic modeling information: monophone, triphone, hmm states, ...
    - feature extraction: MFCC, GT, ...
    - corpus statistics settings
    - scorer settings
    """

    def __init__(
        self,
        costa_args: dict,
        am_args: dict,
        feature_extraction_args: dict,
        scorer: Optional[str] = None,
        scorer_args: Optional[Dict] = None,
        stm_args: Optional[Dict] = None,
    ):
        """
        ##################################################
        :param costa_args: {
            'eval_recordings': True,
            'eval_lm': True
        }
        ##################################################
        :param am_args: {
            'state_tying': "monophone", # "monophone", "cart", "monophon-eow", "dense-tying"
            'states_per_phone': 3, # hidden states per phone
            'state_repetitions': 1, # minimum state repetitions
            'across_word_model': True, # phoneme context is across words
            'early_recombination': False, # TODO:
            'tdp_scale': 1.0, # global scale for all tdp scores
            'tdp_transition': (3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit as negative log-probs
            'tdp_silence': (0.0, 3.0, "infinity", 20.0), # negative log-probs for silence
            'tying_type': "global", # TODO:
            'nonword_phones': "", # e.g. [noise]
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
        :param scorer:
        "kaldi", "sclite", default is sclite
        ##################################################
        :param scorer_args:
        ##################################################
        :param stm_args: arguments to influence stm creation from bliss corpus
        ##################################################
        """
        self.costa_args = costa_args
        self.scorer = scorer
        self.scorer_args = scorer_args
        self.am_args = am_args
        self.feature_extraction_args = feature_extraction_args
        self.stm_args = stm_args


class RescoreArgs:
    """
    TODO: docstring
    """

    def __init__(
        self,
        rescoring_args: Optional[Dict[str, Dict]] = None,
    ):
        """
        :param rescoring_args:
        ##################################################
        """
        self.rescoring_args = rescoring_args


class RasrSteps:
    """
    TODO: docstring
    """

    def __init__(self):
        self._step_names_args = OrderedDict()

    def add_step(self, name, arg):
        self._step_names_args[name] = arg

    def get_step_iter(self):
        return self._step_names_args.items()

    def get_step_names_as_list(self):
        return list(self._step_names_args.keys())

    def get_non_gmm_steps_as_list(self):
        """
        Returns all steps that do not produce new mixtures/alignments
        """
        return ["forced_align"]

    def get_gmm_steps_names_as_list(self):
        """
        Returns all steps that return new mixtures/alignments
        Is the inverse of `get_non_gmm_steps_as_list`
        """
        step_names = list(
            filter(
                lambda x: not any(x.startswith(step) for step in self.get_non_gmm_steps_as_list()),
                self.get_step_names_as_list(),
            )
        )
        return step_names

    def get_args_via_idx(self, idx):
        return list(self._step_names_args.values())[idx]

    def get_prev_gmm_step(self, idx):
        """
        returns the previous gmm step based on given index
        """
        step_names = list(
            filter(
                lambda x: not any(x.startswith(step) for step in self.get_non_gmm_steps_as_list()),
                self.get_step_names_as_list()[:idx],
            )
        )
        return step_names[-1]
