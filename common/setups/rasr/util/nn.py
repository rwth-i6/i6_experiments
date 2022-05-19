__all__ = ["ReturnnRasrDataInput", "OggZipHdfDataInput", "HybridArgs", "NnRecogArgs"]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from sisyphus import tk

import i6_core.am as am
import i6_core.meta as meta
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath

from .rasr import RasrDataInput


class ReturnnRasrDataInput:
    """
    Holds the data for ReturnnRasrTrainingJob.
    """

    def __init__(
        self,
        name: str,
        crp: Optional[rasr.CommonRasrParameters] = None,
        alignments: Optional[
            Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute]
        ] = None,
        feature_flow: Optional[rasr.FlowNetwork] = None,
        features: Optional[
            Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute]
        ] = None,
        acoustic_mixtures: Optional[Union[tk.Path, str]] = None,
        feature_scorers: Optional[Dict[str, Type[rasr.FeatureScorer]]] = None,
        shuffle_data: bool = True,
        **kwargs,
    ):
        self.name = name
        self.crp = crp
        self.alignments = alignments
        self.feature_flow = feature_flow
        self.features = features
        self.acoustic_mixtures = acoustic_mixtures
        self.feature_scorers = feature_scorers
        self.shuffle_data = shuffle_data

    @staticmethod
    def get_data_dict():
        return {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": "sprint-executables/nn-trainer",
            "sprintConfigStr": "",
            "suppress_load_seqs_print": True,
        }

    def build_crp(
        self,
        am_args,
        corpus_object,
        concurrent,
        segment_path,
        lexicon_args,
        cart_tree_path=None,
        allophone_file=None,
        lm_args=None,
    ):
        """
        constructs and returns a CommonRasrParameters from the given settings and files
        """
        crp = rasr.CommonRasrParameters()
        rasr.crp_add_default_output(crp)
        crp.acoustic_model_config = am.acoustic_model_config(**am_args)
        rasr.crp_set_corpus(crp, corpus_object)
        crp.concurrent = concurrent
        crp.segment_path = segment_path

        crp.lexicon_config = rasr.RasrConfig()
        crp.lexicon_config.file = lexicon_args["filename"]
        crp.lexicon_config.normalize_pronunciation = lexicon_args[
            "normalize_pronunciation"
        ]

        if "add_from_lexicon" in lexicon_args:
            crp.acoustic_model_config.allophones.add_from_lexicon = lexicon_args[
                "add_from_lexicon"
            ]
        if "add_all" in lexicon_args:
            crp.acoustic_model_config.allophones.add_all = lexicon_args["add_all"]

        if cart_tree_path is not None:
            crp.acoustic_model_config.state_tying.type = "cart"
            crp.acoustic_model_config.state_tying.file = cart_tree_path

        if lm_args is not None:
            crp.language_model_config = rasr.RasrConfig()
            crp.language_model_config.type = lm_args["type"]
            crp.language_model_config.file = lm_args["filename"]
            crp.language_model_config.scale = lm_args["scale"]

        if allophone_file is not None:
            crp.acoustic_model_config.allophones.add_from_file = allophone_file

        self.crp = crp

    def update_crp_with(
        self,
        *,
        corpus_file: Optional[tk.Path] = None,
        audio_dir: Optional[Union[str, tk.Path]] = None,
        corpus_duration: Optional[int] = None,
        segment_path: Optional[Union[str, tk.Path]] = None,
        concurrent: Optional[int] = None,
        shuffle_data: bool = True,
    ):
        if corpus_file is not None:
            self.crp.corpus_config.file = corpus_file
        if audio_dir is not None:
            self.crp.corpus_config.audio_dir = audio_dir
        if corpus_duration is not None:
            self.crp.corpus_duration = corpus_duration
        if segment_path is not None:
            self.crp.segment_path = segment_path
        if concurrent is not None:
            self.crp.concurrent = concurrent

        if self.shuffle_data or shuffle_data:
            self.crp.corpus_config.segment_order_shuffle = True
            self.crp.corpus_config.segment_order_sort_by_time_length = True
            self.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384

    def get_crp(self, **kwargs):
        """
        constructs and returns a CommonRasrParameters from the given settings and files
        :rtype CommonRasrParameters:
        """
        if self.crp is None:
            self.build_crp(**kwargs)

        if self.shuffle_data:
            self.crp.corpus_config.segment_order_shuffle = True
            self.crp.corpus_config.segment_order_sort_by_time_length = True
            self.crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384

        return self.crp


class OggZipHdfDataInput:
    def __init__(
        self,
        oggzip_files: tk.Path,
        alignments: tk.Path,
        context_window: Dict,
        audio: Dict,
        targets: str,
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
    ):
        """
        :param oggzip_files:
        :param alignments:
        :param context_window: {"classes": 1, "data": 242}
        :param audio: e.g. {"features": "raw", "sample_rate": 16000} for raw waveform input with a sample rate of 16 kHz
        :param partition_epoch:
        :param seq_ordering:
        :param targets:
        """
        self.oggzip_files = oggzip_files
        self.alignments = alignments
        self.context_window = context_window
        self.audio = audio
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.targets = targets

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "context_window": self.context_window,
            "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
            "datasets": {
                "hdf": {
                    "class": "HDFDataset",
                    "files": [self.alignments.get_path()],
                    "use_cache_manager": True,
                },
                "ogg": {
                    "class": "OggZipDataset",
                    "audio": self.audio,
                    "partition_epoch": self.partition_epoch,
                    "path": tuple(self.oggzip_files.get_path()),
                    "seq_ordering": self.seq_ordering,
                    "targets": self.targets,
                    "use_cache_manager": True,
                },
            },
            "seq_order_control_dataset": "ogg",
        }


class HybridArgs:
    def __init__(
        self,
        returnn_training_configs: Dict[str, returnn.ReturnnConfig],
        returnn_recognition_configs: Dict[str, returnn.ReturnnConfig],
        training_args: Dict[str, Any],
        recognition_args: Dict[str, Dict[str, Dict]],
        test_recognition_args: Optional[Dict[str, Dict[str, Dict]]] = None,
    ):
        """
        ##################################################
        :param returnn_training_configs
        ##################################################
        :param returnn_recognition_configs
            If a config is not found here, the corresponding training config is used
        ##################################################
        :param training_args:
        ##################################################
        :param recognition_args:
        ##################################################
        :param test_recognition_args:
        ##################################################
        """
        self.returnn_training_configs = returnn_training_configs
        self.returnn_recognition_configs = returnn_recognition_configs
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args


@dataclass()
class NnRecogArgs:
    name: str
    returnn_config: returnn.ReturnnConfig
    checkpoints: Dict[int, returnn.Checkpoint]
    acoustic_mixture_path: tk.Path
    prior_scales: List[float]
    pronunciation_scales: List[float]
    lm_scales: List[float]
    optimize_am_lm_scale: bool
    feature_flow_key: str
    search_parameters: Dict
    lm_lookahead: bool
    lattice_to_ctm_kwargs: Dict
    parallelize_conversion: bool
    rtf: int
    mem: int
    lookahead_options: Optional[Dict] = None
    epochs: Optional[List[int]] = None
