__all__ = [
    "GmmMonophoneArgs",
    "GmmCartArgs",
    "GmmTriphoneArgs",
    "GmmVtlnArgs",
    "PrevCtm",
    "GmmSatArgs",
    "GmmVtlnSatArgs",
    "ForcedAlignmentArgs",
    "RecognitionArgs",
    "OutputArgs",
    "GmmOutput",
]

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

from sisyphus import tk

import i6_core.rasr as rasr

from i6_core.cart.questions import BasicCartQuestions, PythonCartQuestions
from i6_core.util import MultiPath

from .nn import ReturnnRasrDataInput


class GmmMonophoneArgs:
    def __init__(
        self,
        linear_alignment_args: dict,
        training_args: dict,
        recognition_args: dict,
        test_recognition_args: Optional[dict] = None,
        sdm_args: Optional[dict] = None,
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
            'align_iter': 75, # single density EM iteration steps
            'splits': 10, # number of density splits
            'accs_per_split': 2, # re-estimations of sample assignments between splits?
        }
        ##################################################
        :param recognition_args: {
            'eval_iter': [7, 8, 9, 10] # iterations to evaluate corresponding to the "splits" iterations
            'pronunciation_scales': [1.0] # scales the pronunciation props (which are simply 1.0 in most cases), only relevant then when using "normalize-pronunciations"
            'lm_scales': [9.0, 9.25, 9.50, 9.75, 10.0, 10.25, 10.50] # obviously
            'recog_args': {
                'feature_flow': dev_corpus_name,
                'pronunciation_scale': pronunciation_scale, # scale for the pronunciation score,
                    only relevant when normalize_pronunciation=True for the lexicon or there are different
                    scores in the lexicon itself
                'lm_scale': lm_scale, # language model scale
                'lm_lookahead': True, # use lookahead, using the lm for pruning partial words
                'lookahead_options': None, # TODO:
                'create_lattice': True, # write lattice cache files
                'eval_single_best': True, # show the evaluation of the best path in lattice in the log (model score)
                'eval_best_in_lattice': True, # show the evaluation of the best path in lattice in the log (oracle)
                'search_parameters': {
                    'beam_pruning': 14.0, # prob ratio of best path compared to pruned path
                    'beam-pruning-limit': 100000, # maximum number of paths
                    'word-end-pruning': 0.5, # pruning ratio at the end of completed words
                    'word-end-pruning-limit': 15000 # maximum number of paths at completed words
                },
                'best_path_algo': 'bellman-ford',  # options: bellman-ford, dijkstra
                'fill_empty_segments': False, # insert dummy when transcription output is empty
                'rtf': 30, # time estimation for jobs, which will be time=duration/concurrent*rtf
                'mem': 8, # memory for jobs
                'use_gpu': False, # True makes no sense
            }
        }

        when getting an zero weights error, set:

        extra_config = sprint.SprintConfig()
        extra_config.allow_zero_weights = True
        {accumulate,split,align}_extra_args = {'extra_config': extra_config}
        ##################################################
        :param test_recognition_args:
        decoding parameters might change depending on whether dev or test sets are decoded.
        setting test_recognition_args to None means no test recognition!
        test_recognition_args UPDATES recognition_args
        ##################################################
        :param sdm_args: TODO:
        ##################################################
        """
        self.linear_alignment_args = linear_alignment_args
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args
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
        test_recognition_args: Optional[dict] = None,
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

        '{accumulate,split,align}_extra_rqmt': {'mem': 10, 'time': 8},
        ##################################################
        :param recognition_args:
        ##################################################
        :param test_recognition_args:
        decoding parameters might change depending on whether dev or test sets are decoded.
        setting test_recognition_args to None means no test recognition!
        test_recognition_args UPDATES recognition_args
        ##################################################
        :param sdm_args: {
            'feature_flow': "mfcc+context+lda",
            'alignment': "train_tri",   # if using run function not needed
        }
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args
        self.sdm_args = sdm_args


class GmmVtlnArgs:
    def __init__(
        self,
        training_args: Dict[str, Union[str, int, Dict]],
        recognition_args: dict,
        test_recognition_args: Optional[dict] = None,
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

        vtln align time = 8
        ##################################################
        :param recognition_args:
        ##################################################
        :param test_recognition_args:
        decoding parameters might change depending on whether dev or test sets are decoded.
        setting test_recognition_args to None means no test recognition!
        test_recognition_args UPDATES recognition_args
        ##################################################
        :param sdm_args: {
            'feature_flow': "mfcc+context+lda+vtln",
            'alignment': "train_vtln",  # if using run function not needed
        }
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args
        self.sdm_args = sdm_args


@dataclass
class PrevCtm:
    """
    Defines a previous ctm output from a recognition which is used to extract the
    speaker alignments for SAT adaptation for the dev/train sets.

    :param prev_step_key: name of the step, e.g. "tri"
    :param pronunciation_scale: one of the existing pronunciation scales
    :param lm_scale: one of the existing lm-scales
    :param iteration: one of the existing GMM iterations
    :param optimized_lm: use automatically optimized lm.scale which might differ from param lm_scale
    """

    prev_step_key: str
    pronunciation_scale: Union[float, int]
    lm_scale: Union[float, int]
    iteration: int
    optimized_lm: bool


class GmmSatArgs:
    def __init__(
        self,
        training_args: dict,
        recognition_args: dict,
        test_recognition_args: Optional[dict] = None,
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
        :param test_recognition_args:
        decoding parameters might change depending on whether dev or test sets are decoded.
        setting test_recognition_args to None means no test recognition!
        test_recognition_args UPDATES recognition_args
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args
        self.sdm_args = sdm_args


class GmmVtlnSatArgs:
    def __init__(
        self,
        training_args: dict,
        recognition_args: dict,
        test_recognition_args: Optional[dict] = None,
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
        :param test_recognition_args:
        decoding parameters might change depending on whether dev or test sets are decoded.
        setting test_recognition_args to None means no test recognition!
        test_recognition_args UPDATES recognition_args
        ##################################################
        """
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args
        self.sdm_args = sdm_args


class ForcedAlignmentArgs:
    """
    parameters for forced alignment on the target corpus
    """

    def __init__(
        self,
        name: str,
        target_corpus_key: str,
        flow: Union[str, List[str], Tuple[str], rasr.FlagDependentFlowAttribute],
        feature_scorer: Union[str, List[str], Tuple[str], rasr.FeatureScorer],
    ):
        """
        :param name: experiment name
        :param target_corpus_key: target corpus
        :param flow: feature flow
        :param feature_scorer: feature scorer (normally trained on different corpus)
        """
        self.name = name
        self.target_corpus_key = target_corpus_key
        self.flow = flow
        self.feature_scorer = feature_scorer


class RecognitionArgs:
    """
    stand alone recognition
    """

    def __init__(self, name, recognition_args):
        """
        :param name: recognition name
        :param recognition_args: recognition arguments. further doc: GmmMonophoneArgs.recognition_args
        """
        self.name = name
        self.recognition_args = recognition_args


class OutputArgs:
    """
    defines which output should be generated for the GMM pipeline
    """

    def __init__(self, name):
        """
        :param name: name the outputs
        """
        self.name = name
        self.corpus_type_mapping = {}
        self.extract_features = []

    def define_corpus_type(self, corpus_key, corpus_type):
        """
        Defines a mapping from corpus_key to corpus_type (train, dev, test).
        This defines how the output is structured/selected.

        :param corpus_key: any corpus key previously defined. see GmmSystem.init_system
        :param corpus_type: train, dev or test
        :return:
        """
        self.corpus_type_mapping[corpus_key] = corpus_type

    def add_feature_to_extract(self, feature_key):
        """
        add feature keys for extraction and output

        :param feature_key: for example mfcc or gt. see RasrInitArgs.feature_extraction_args
        :return:
        """
        self.extract_features.append(feature_key)


class GmmOutput:
    """
    holds all the information generated as output to the GMM pipeline
    """

    def __init__(self):
        self.crp: Optional[rasr.CommonRasrParameters] = None
        self.acoustic_mixtures: Optional[tk.Path] = None
        self.feature_scorers: Dict[str, Type[rasr.FeatureScorer]] = {}
        self.feature_flows: Dict[str, rasr.FlowNetwork] = {}
        self.features: Dict[
            str, Union[tk.Path, MultiPath, rasr.FlagDependentFlowAttribute]
        ] = {}
        self.alignments: Optional[
            Union[tk.Path, MultiPath, rasr.FlagDependentFlowAttribute]
        ] = None

    def as_returnn_rasr_data_input(
        self,
        name: str = "init",
        *,
        feature_flow_key: str = "gt",
        shuffle_data: bool = True,
    ):
        """
        dumps stored GMM pipeline output/file/information for ReturnnRasrTraining

        :param name:
        :param feature_flow_key:
        :param shuffle_data:
        :return:
        :rtype: ReturnnRasrDataInput
        """
        return ReturnnRasrDataInput(
            name=name,
            crp=copy.deepcopy(self.crp),
            alignments=self.alignments,
            feature_flow=self.feature_flows[feature_flow_key],
            features=self.features[feature_flow_key],
            acoustic_mixtures=self.acoustic_mixtures,
            feature_scorers=self.feature_scorers,
            shuffle_data=shuffle_data,
        )
