__all__ = ['LabelTree', 'LabelScorer']

from sisyphus import *
Path = setup_path(__package__)
from i6_core.rasr.config import *

class LabelTree:
    def __init__(self, label_unit, lexicon_config=None, **kwargs):
        self.config         = RasrConfig()
        self.post_config    = RasrConfig()
        config = lexicon_config
        lexicon_config = RasrConfig()
        lexicon_config.file = config['filename']
        lexicon_config.normalize_pronunciation = config['normalize_pronunciation']
        self.lexicon_config = lexicon_config # allow overwrite csp lexicon

        self.config.label_unit = label_unit
        if kwargs.get('use_transition_penalty', False):
            self.config.use_transition_penalty = True
            for name in ['root_transition', 'default_transition', 'special_transition']:
                args = kwargs.get(name, None)
                if args is None: continue
                for key, value in args.items():
                    self.config[name.replace('_','-')][key] = value
            special = kwargs.get('special_transition_labels', None)
            if special is not None:
                self.config.special_transition_labels = special
        if kwargs.get('skip_silence', False):
            self.config.skip_silence = True
            if label_unit == 'hmm': # we need the ci wordend states
                self.config.hmm_state_tree.add_ci_transitions = True

    def apply_config(self, path, config, post_config=None):
        config[path]._update(self.config)
        if post_config is not None:
            post_config[path]._update(self.post_config)

# These are impelmented #
valid_scorer_types = ['precomputed-log-posterior', 'tf-attention', 'tf-rnn-transducer', 'tf-ffnn-transducer']

class LabelScorer:
    def __init__(self, scorerType, scale=1.0, labelFile=None, numClasses=None, usePrior=False, priorScale=0.6, priorFile=None, extraArgs={}):
        self.config            = RasrConfig()
        self.post_config       = RasrConfig()

        assert scorerType in valid_scorer_types
        self.config.label_scorer_type = scorerType
        self.config.scale = scale

        if labelFile is not None:
            self.config.label_file = labelFile
        else: assert numClasses is not None, 'no label-file nor number-of-classes given'

        if numClasses is not None:
            self.config.number_of_classes = numClasses

        if usePrior:
            assert priorFile is not None, "prior is activated but no prior file provided"
            self.config.use_prior = True
            self.config.prior_file = priorFile
            self.config.priori_scale = priorScale

        # sprint key values #
        for key in extraArgs:
            self.config[key.replace('_','-')] = extraArgs[key]

    def apply_config(self, path, config, post_config=None):
        config[path]._update(self.config)
        if post_config is not None:
            post_config[path]._update(self.post_config)

    @classmethod
    def need_tf_flow(cls, scorerType):
        return scorerType == 'precomputed-log-posterior'

    def set_loader_config(self, loaderConfig):
        if loaderConfig is not None:
            self.config.loader = loaderConfig

    def set_input_config(self, inputConfig=None):
        if inputConfig is not None:
            self.config.feature_input_map = inputConfig
        else: # default
            self.config.feature_input_map = RasrConfig()
            self.config.feature_input_map.info_0.param_name             = 'feature'
            self.config.feature_input_map.info_0.tensor_name            = 'extern_data/placeholders/data/data'
            self.config.feature_input_map.info_0.seq_length_tensor_name = 'extern_data/placeholders/data/data_dim0_size'