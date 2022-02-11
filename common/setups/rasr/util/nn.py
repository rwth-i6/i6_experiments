__all__ = [
    "ReturnnRasrDataInput",
    "OggZipHdfDataInput",
    "NnArgs",
]

from typing import Any, Dict, Optional, Type, Union

from sisyphus import tk

import i6_core.am as am
import i6_core.meta as meta
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath

from .rasr import RasrDataInput


class ReturnnRasrDataInput(RasrDataInput):
    """
    Holds the data for ReturnnRasrTrainingJob.
    """

    def __init__(
        self,
        name: str,
        cart_tree: Union[tk.Path, str],
        alignments: Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute],
        corpus_object: meta.CorpusObject,
        lexicon: dict,
        lm: Optional[dict] = None,
        concurrent: int = 1,
        crp: Optional[rasr.CommonRasrParameters] = None,
        am_args: Optional[Dict] = None,
        feature_flow: Optional[rasr.FlowNetwork] = None,
        features: Optional[
            Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute]
        ] = None,
        segment_path: Optional[Union[tk.Path, str]] = None,
        allophone_file: Optional[Union[tk.Path, str]] = None,
        acoustic_mixtures: Optional[Union[tk.Path, str]] = None,
        feature_scorers: Optional[Dict[str, Type[rasr.FeatureScorer]]] = None,
        shuffle_data: bool = True,
        **kwargs,
    ):
        super().__init__(
            corpus_object=corpus_object,
            lexicon=lexicon,
            lm=lm,
            concurrent=concurrent,
        )
        self.name = name
        # from RasrDataInput: CorpusObject, Lexicon: dict, LM: dict, Concurrency: int
        self.cart_tree = cart_tree
        self.alignments = alignments
        self.crp = crp
        self.am_args = am_args
        self.feature_flow = feature_flow
        self.features = features
        self.segment_path = segment_path
        self.allophone_file = allophone_file
        self.acoustic_mixtures = acoustic_mixtures
        self.feature_scorers = feature_scorers
        self.shuffle_data = shuffle_data

    def get_data_dict(self):
        return {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": "sprint-executables/nn-trainer",
            "sprintConfigStr": "",
            "suppress_load_seqs_print": True,
        }

    def get_crp(self):
        """
        constructs and returns a CommonRasrParameters from the given settings and files
        :rtype CommonRasrParameters:
        """
        if self.crp is not None:
            return self.crp
        crp = rasr.CommonRasrParameters()
        rasr.crp_add_default_output(crp)
        crp.acoustic_model_config = am.acoustic_model_config(**self.am_args)
        rasr.crp_set_corpus(crp, self.corpus_object)
        crp.concurrent = self.concurrent
        crp.segment_path = self.segment_path

        if self.shuffle_data:
            crp.corpus_config.segment_order_shuffle = True
            crp.corpus_config.segment_order_sort_by_time_length = True
            crp.corpus_config.segment_order_sort_by_time_length_chunk_size = 384

        crp.lexicon_config = rasr.RasrConfig()
        crp.lexicon_config.file = self.lexicon["filename"]
        crp.lexicon_config.normalize_pronunciation = self.lexicon[
            "normalize_pronunciation"
        ]

        if "add_from_lexicon" in self.lexicon:
            crp.acoustic_model_config.allophones.add_from_lexicon = self.lexicon[
                "add_from_lexicon"
            ]
        if "add_all" in self.lexicon:
            crp.acoustic_model_config.allophones.add_all = self.lexicon["add_all"]

        if self.cart_tree is not None:
            crp.acoustic_model_config.state_tying.type = "cart"
            crp.acoustic_model_config.state_tying.file = self.cart_tree

        if self.lm is not None:
            crp.language_model_config = rasr.RasrConfig()
            crp.language_model_config.type = self.lm["type"]
            crp.language_model_config.file = self.lm["filename"]
            crp.language_model_config.scale = self.lm["scale"]

        if self.allophone_file is not None:
            crp.acoustic_model_config.allophones.add_from_file = self.allophone_file

        return crp


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


class NnArgs:
    def __init__(
        self,
        returnn_configs: Dict[str, returnn.ReturnnConfig],
        training_args: Dict[str, Any],
        recognition_args: Dict[str, Dict[str, Dict]],
        test_recognition_args: Optional[Dict[str, Dict[str, Dict]]] = None,
    ):
        """
        ##################################################
        :param returnn_configs
        ##################################################
        :param training_args:
        ##################################################
        :param recognition_args:
        ##################################################
        :param test_recognition_args:
        ##################################################
        """
        self.returnn_configs = returnn_configs
        self.training_args = training_args
        self.recognition_args = recognition_args
        self.test_recognition_args = test_recognition_args
