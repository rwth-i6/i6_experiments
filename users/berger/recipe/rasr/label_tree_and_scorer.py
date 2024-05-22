__all__ = ["LabelTree", "LabelScorer"]

from typing import Any, Dict, Optional

from i6_core import rasr
from i6_experiments.users.berger import helpers
from sisyphus import tk, setup_path

assert __package__ is not None
Path = setup_path(__package__)


class LabelTree:
    def __init__(
        self,
        label_unit: str,
        lexicon_config: helpers.LexiconConfig,
        use_transition_penalty: bool = False,
        root_transition: Dict = {},
        default_transition: Dict = {},
        special_transition: Dict = {},
        special_transition_labels: Optional[Any] = None,
        skip_silence: bool = False,
    ):
        self.post_config = rasr.RasrConfig()

        self.lexicon_config = rasr.RasrConfig()
        self.lexicon_config.file = lexicon_config.filename
        self.lexicon_config.normalize_pronunciation = lexicon_config.normalize_pronunciation

        self.config = rasr.RasrConfig()
        self.config.label_unit = label_unit
        if use_transition_penalty:
            self.config.use_transition_penalty = True
            for key, value in root_transition:
                self.config.root_transition[key] = value
            for key, value in default_transition:
                self.config.default_transition[key] = value
            for key, value in special_transition:
                self.config.special_transition[key] = value

            if special_transition_labels is not None:
                self.config.special_transition_labels = special_transition_labels

        if skip_silence:
            self.config.skip_silence = True
            if label_unit == "hmm":  # we need the ci wordend states
                self.config.hmm_state_tree.add_ci_transitions = True

    def apply_config(
        self,
        path: str,
        config: rasr.RasrConfig,
        post_config: Optional[rasr.RasrConfig] = None,
    ):
        config[path]._update(self.config)
        if post_config is not None:
            post_config[path]._update(self.post_config)


# These are implemented #
valid_scorer_types = [
    "precomputed-log-posterior",
    "tf-attention",
    "tf-rnn-transducer",
    "tf-ffnn-transducer",
    "onnx-ffnn-transducer",
]


class LabelScorer:
    def __init__(
        self,
        scorer_type: str,
        scale: float = 1.0,
        label_file: Optional[tk.Path] = None,
        num_classes: Optional[int] = None,
        use_prior: bool = False,
        prior_scale: float = 0.6,
        prior_file: Optional[tk.Path] = None,
        extra_args: Optional[Dict] = None,
    ):
        self.config = rasr.RasrConfig()
        self.post_config = rasr.RasrConfig()

        assert scorer_type in valid_scorer_types
        self.config.label_scorer_type = scorer_type
        self.config.scale = scale

        if label_file is not None:
            self.config.label_file = label_file
        else:
            assert num_classes is not None, "no label-file nor number-of-classes given"

        if num_classes is not None:
            self.config.number_of_classes = num_classes

        if use_prior:
            assert prior_file is not None, "prior is activated but no prior file provided"
            self.config.use_prior = True
            self.config.prior_file = prior_file
            self.config.priori_scale = prior_scale

        # sprint key values #
        if extra_args is not None:
            for key, value in extra_args.items():
                self.config[key.replace("_", "-")] = value

    @property
    def scorer_type(self):
        return self.config.label_scorer_type

    @property
    def scale(self):
        return self.config.scale

    @property
    def label_file(self):
        if self.config._get("label-file") is not None:
            return self.config.label_file
        return None

    @property
    def num_classes(self):
        if self.config._get("number-of-classes") is not None:
            return self.config.number_of_classes
        return None

    @property
    def use_prior(self):
        if self.config._get("use-prior") is not None:
            return self.config["use-prior"]
        return False

    @property
    def prior_scale(self):
        if self.config._get("priori-scale") is not None:
            return self.config["priori-scale"]
        return 1.0

    @property
    def prior_file(self):
        if self.config._get("prior-file") is not None:
            return self.config["prior-file"]
        return None

    @property
    def extra_args(self):
        return {
            key: val
            for key, val in self.config._items()
            if key not in [
                "label-scorer-type",
                "scale",
                "label-file",
                "number-of-classes",
                "use-prior",
                "priori-scale",
                "prior-file",
            ]
        }

    def apply_config(
        self,
        path: str,
        config: rasr.RasrConfig,
        post_config: Optional[rasr.RasrConfig] = None,
    ):
        config[path]._update(self.config)
        if post_config is not None:
            post_config[path]._update(self.post_config)

    def set_loader_config(self, loader_config: rasr.RasrConfig):
        self.config.loader = loader_config

    def set_input_config(self, input_config: Optional[rasr.RasrConfig] = None):
        if input_config is not None:
            self.config.feature_input_map = input_config
            return

        # Create feature input map
        self.config.feature_input_map = rasr.RasrConfig()
        self.config.feature_input_map.info_0.param_name = "feature"
        self.config.feature_input_map.info_0.tensor_name = "extern_data/placeholders/data/data"
        self.config.feature_input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/data/data_dim0_size"
