__all__ = [
    "RasrTrainingArgs",
    "ReturnnRasrDataInput",
    "OggZipHdfDataInput",
    "HybridArgs",
    "NnRecogArgs",
    "NnForcedAlignArgs",
]

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

import i6_core.am as am
import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.util import MultiPath

from .rasr import RasrDataInput

RasrCacheTypes = Union[tk.Path, str, MultiPath, rasr.FlagDependentFlowAttribute]


@dataclass(frozen=True)
class RasrTrainingArgs:
    """
    Options for writing a RASR training config. See `ReturnnRasrTrainingJob`.
    Most of them may be disregarded, i.e. the defaults can be left untouched.
    """

    partition_epochs: Optional[int] = None
    num_classes: Optional[int] = None
    disregarded_classes: Optional[tk.Path] = None
    class_label_file: Optional[tk.Path] = None
    buffer_size: int = 200 * 1024
    extra_rasr_config: Optional[rasr.RasrConfig] = None
    extra_rasr_post_config: Optional[rasr.RasrConfig] = None
    use_python_control: bool = True


class ReturnnRasrDataInput:
    """
    Holds the data for ReturnnRasrTrainingJob.
    """

    def __init__(
        self,
        name: str,
        crp: Optional[rasr.CommonRasrParameters] = None,
        alignments: Optional[RasrCacheTypes] = None,
        feature_flow: Optional[Union[rasr.FlowNetwork, Dict[str, rasr.FlowNetwork]]] = None,
        features: Optional[Union[RasrCacheTypes, Dict[str, RasrCacheTypes]]] = None,
        acoustic_mixtures: Optional[Union[tk.Path, str]] = None,
        feature_scorers: Optional[Dict[str, Type[rasr.FeatureScorer]]] = None,
        shuffle_data: bool = True,
        stm: Optional[tk.Path] = None,
        glm: Optional[tk.Path] = None,
        rasr_training_args: Optional[RasrTrainingArgs] = None,
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
        self.stm = stm
        self.glm = glm
        self.rasr_training_args = rasr_training_args or {}

    def get_training_feature_flow_file(self) -> tk.Path:
        feature_flow = returnn.ReturnnRasrTrainingJob.create_flow(self.feature_flow, self.alignments)
        write_feature_flow = rasr.WriteFlowNetworkJob(feature_flow)
        return write_feature_flow.out_flow_file

    def get_training_rasr_config_file(self) -> tk.Path:
        config, post_config = returnn.ReturnnRasrTrainingJob.create_config(
            self.crp, self.alignments, **asdict(self.rasr_training_args)
        )
        config.neural_network_trainer.feature_extraction.file = self.get_training_feature_flow_file()
        write_rasr_config = rasr.WriteRasrConfigJob(config, post_config)
        return write_rasr_config.out_config

    def get_data_dict(self) -> Dict[str, Union[str, DelayedFormat, tk.Path]]:
        """Returns the data dict for the ExternSprintDataset to be used in a training ReturnnConfig."""
        config_file = self.get_training_rasr_config_file()
        config_str = DelayedFormat("--config={} --*.LOGFILE=nn-trainer.{}.log --*.TASK=1", config_file, self.name)
        dataset = {
            "class": "ExternSprintDataset",
            "sprintTrainerExecPath": rasr.RasrCommand.select_exe(self.crp.nn_trainer_exe, "nn-trainer"),
            "sprintConfigStr": config_str,
        }
        partition_epochs = self.rasr_training_args.partition_epochs
        if partition_epochs is not None:
            dataset["partitionEpoch"] = partition_epochs
        return dataset

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
        crp.lexicon_config.normalize_pronunciation = lexicon_args["normalize_pronunciation"]

        if "add_from_lexicon" in lexicon_args:
            crp.acoustic_model_config.allophones.add_from_lexicon = lexicon_args["add_from_lexicon"]
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

    def get_crp(self, **kwargs) -> rasr.CommonRasrParameters:
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
        oggzip_files: List[tk.Path],
        alignments: List[tk.Path],
        audio: Dict,
        partition_epoch: int = 1,
        seq_ordering: str = "laplace:.1000",
        meta_args: Optional[Dict[str, Any]] = None,
        ogg_args: Optional[Dict[str, Any]] = None,
        hdf_args: Optional[Dict[str, Any]] = None,
        acoustic_mixtures: Optional[tk.Path] = None,
    ):
        """
        :param oggzip_files: zipped ogg files which contain the audio
        :param alignments: hdf files which contain dumped RASR alignments
        :param audio: e.g. {"features": "raw", "sample_rate": 16000} for raw waveform input with a sample rate of 16 kHz
        :param partition_epoch: if >1, split the full dataset into multiple sub-epochs
        :param seq_ordering: sort the sequences in the dataset, e.g. "random" or "laplace:.100"
        :param meta_args: parameters for the `MetaDataset`
        :param ogg_args: parameters for the `OggZipDataset`
        :param hdf_args: parameters for the `HdfDataset`
        :param acoustic_mixtures: path to a RASR acoustic mixture file (used in System classes, not RETURNN training)
        """
        self.oggzip_files = oggzip_files
        self.alignments = alignments
        self.audio = audio
        self.partition_epoch = partition_epoch
        self.seq_ordering = seq_ordering
        self.meta_args = meta_args
        self.ogg_args = ogg_args
        self.hdf_args = hdf_args
        self.acoustic_mixtures = acoustic_mixtures

    def get_data_dict(self):
        return {
            "class": "MetaDataset",
            "data_map": {"classes": ("hdf", "classes"), "data": ("ogg", "data")},
            "datasets": {
                "hdf": {
                    "class": "HDFDataset",
                    "files": self.alignments,
                    "use_cache_manager": True,
                    **(self.hdf_args or {}),
                },
                "ogg": {
                    "class": "OggZipDataset",
                    "audio": self.audio,
                    "partition_epoch": self.partition_epoch,
                    "path": self.oggzip_files,
                    "seq_ordering": self.seq_ordering,
                    "use_cache_manager": True,
                    **(self.ogg_args or {}),
                },
            },
            "seq_order_control_dataset": "ogg",
            **(self.meta_args or {}),
        }


# Attribute names are invalid identifiers, therefore use old syntax
SearchParameters = TypedDict(
    "SearchParameters",
    {
        "beam-pruning": float,
        "beam-pruning-limit": float,
        "lm-state-pruning": Optional[float],
        "word-end-pruning": float,
        "word-end-pruning-limit": float,
    },
)


class LookaheadOptions(TypedDict):
    cache_high: Optional[int]
    cache_low: Optional[int]
    history_limit: Optional[int]
    laziness: Optional[int]
    minimum_representation: Optional[int]
    tree_cutoff: Optional[int]


class LatticeToCtmArgs(TypedDict):
    best_path_algo: Optional[str]
    encoding: Optional[str]
    extra_config: Optional[Any]
    extra_post_config: Optional[Any]
    fill_empty_segments: Optional[bool]


class NnRecogArgs(TypedDict):
    acoustic_mixture_path: Optional[tk.Path]
    checkpoints: Optional[Dict[int, returnn.Checkpoint]]
    create_lattice: Optional[bool]
    epochs: Optional[List[int]]
    eval_best_in_lattice: Optional[bool]
    eval_single_best: Optional[bool]
    feature_flow_key: str
    lattice_to_ctm_kwargs: Optional[LatticeToCtmArgs]
    lm_lookahead: bool
    lm_scales: List[float]
    lookahead_options: Optional[LookaheadOptions]
    mem: int
    name: str
    optimize_am_lm_scale: bool
    parallelize_conversion: Optional[bool]
    prior_scales: List[float]
    pronunciation_scales: List[float]
    returnn_config: Optional[returnn.ReturnnConfig]
    rtf: int
    search_parameters: Optional[SearchParameters]
    use_gpu: Optional[bool]


KeyedRecogArgsType = Dict[str, Union[Dict[str, Any], NnRecogArgs]]


class EpochPartitioning(TypedDict):
    dev: int
    train: int


class NnTrainingArgs(TypedDict):
    buffer_size: Optional[int]
    class_label_file: Optional[tk.Path]
    cpu_rqmt: Optional[int]
    device: Optional[str]
    disregarded_classes: Optional[Any]
    extra_rasr_config: Optional[rasr.RasrConfig]
    extra_rasr_post_config: Optional[rasr.RasrConfig]
    horovod_num_processes: Optional[int]
    keep_epochs: Optional[bool]
    log_verbosity: Optional[int]
    mem_rqmt: Optional[int]
    num_classes: int
    num_epochs: int
    partition_epochs: Optional[EpochPartitioning]
    save_interval: Optional[int]
    time_rqmt: Optional[int]
    use_python_control: Optional[bool]


class HybridArgs:
    def __init__(
        self,
        returnn_training_configs: Dict[str, returnn.ReturnnConfig],
        returnn_recognition_configs: Dict[str, returnn.ReturnnConfig],
        training_args: Union[Dict[str, Any], NnTrainingArgs],
        recognition_args: KeyedRecogArgsType,
        test_recognition_args: Optional[KeyedRecogArgsType] = None,
    ):
        """
        ##################################################
        :param returnn_training_configs
            RETURNN config keyed by training corpus.
        ##################################################
        :param returnn_recognition_configs
            If a config is not found here, the corresponding training config is used
        ##################################################
        :param training_args:
        ##################################################
        :param recognition_args:
            Configuration for recognition on dev corpora.
        ##################################################
        :param test_recognition_args:
            Additional configuration for recognition on test corpora. Merged with recognition_args.
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
    native_ops: Optional[List[str]] = None


class NnForcedAlignArgs(TypedDict):
    name: str
    target_corpus_keys: List[str]
    feature_scorer_corpus_key: str
    scorer_model_key: Union[str, List[str], Tuple[str], rasr.FeatureScorer]
    epoch: int
    base_flow_key: str
    tf_flow_key: str
    dump_alignment: bool
