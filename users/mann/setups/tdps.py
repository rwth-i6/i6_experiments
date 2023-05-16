import copy

import numpy as np

from i6_core import rasr
from i6_core.am import config

from enum import Enum
from collections import namedtuple
from itertools import product

transition = namedtuple("Transitions", ["fwd", "loop"])

from functools import wraps
def maybe_delayed(f):
    @wraps(f)
    def f_wrapped(v):
        # if isinstance(v, Variable):
        if hasattr(v, "function"):
            return v.function(f) 
        return f(v)
    return f_wrapped

def negate(x):
    return -x

def nlog(x):
    return -np.log(x)

def compose(f, g):
    def composition(x):
        return f(g(x))
    return composition

class compose:
    def __init__(self, f, g):
        self.f, self.g = f, g
        self.__name__ = "compose"
    
    def __call__(self, x):
        return self.f(self.g(x))

def log(x):
    return np.log(x)

def exp(x):
    return np.exp(x)

def norm_fwd(p):
    return (p / np.sum(p, keepdims=True))[0]

def make_array(x):
    return np.array(x)

@maybe_delayed
def assert_bounded(x):
    assert 0 <= x <= 1
    return x

class Transition:
    __slots__ = "values"

    def __init__(self, values):
        self.values = values
    
    @classmethod
    def _make(cls, transitions):
        return cls(transitions)
    
    @classmethod
    def from_fwd_prob(cls, fwd):
        fwd = assert_bounded(fwd)
        return cls([fwd, 1-fwd])
    
    @property
    def fwd(self):
        return self.values[0]

    @property
    def loop(self):
        return self.values[1]
    
    def __eq__(self, other):
        return self.values == other.values
    
    def __str__(self):
        return "Transitions(fwd={}, loop={})".format(self.fwd, self.loop)
    
    def to_weights(self):
        return type(self)(-np.log(self.values))
    
    def to_rasr_config(self):
        transition = self.to_weights()
        config = rasr.RasrConfig()
        config.forward = transition.fwd
        config.loop = transition.loop
        return config

transition = Transition


class ParameterMode(Enum):
    PROB = 0
    LOG_PROB = 1
    WEIGHT = 2

class SimpleTransitionModel:
    __slots__ = "speech", "silence", "_mode"

    def __init__(self,
        speech: transition,
        silence: transition,
        mode: ParameterMode=ParameterMode.PROB
    ):
        assert isinstance(speech , transition)
        assert isinstance(silence, transition)
        self.speech = speech
        self.silence = silence
        self._mode = mode
    
    @property
    def mode(self):
        return self._mode.name
    
    def __sis_state__(self):
        return {
            "speech": self.speech.values,
            "silence": self.silence.values,
            "mode": self.mode
        }
    
    def is_weights(self):
        return self._mode is ParameterMode.WEIGHT

    def is_log_probs(self):
        return self._mode is ParameterMode.LOG_PROB

    def is_probs(self):
        return self._mode is ParameterMode.PROB
    
    def evaluate(self):
        return type(self)(
            Transition(self.speech.values.get()),
            Transition(self.silence.values.get()),
            self._mode
        )

    @classmethod
    def from_weights(
        cls, speech, silence
    ):
        return cls(
            transition._make( speech),
            transition._make(silence),
            ParameterMode.WEIGHT
        )
    
    @classmethod
    def from_zeros(cls):
        return cls.from_weights([0.0, 0.0], [0.0, 0.0])
    
    @classmethod
    def from_fwd_probs(
        cls, speech_fwd, silence_fwd
    ):
        return cls(
            transition.from_fwd_prob(speech_fwd),
            transition.from_fwd_prob(silence_fwd),
            ParameterMode.PROB
        )

    @classmethod
    def from_log_probs(
        cls, speech, silence
    ):
        return cls(
            transition._make(speech),
            transition._make(silence),
            ParameterMode.LOG_PROB
        )
    
    def to_weights(self):
        if self.is_weights():
            return self
        f = (lambda x: -x) if self.is_log_probs() else (lambda x: -np.log(x))
        f = make_array
        if self.is_probs():
            f = compose(log, f)
        elif not self.is_log_probs():
            raise TypeError("Parameter mode is not correct")
        f = compose(negate, f)
        f = maybe_delayed(f)
        return self.from_weights(
            f(self.speech.values), f(self.silence.values)
        )
    
    def to_log_probs(self):
        if self.is_log_probs():
            return self
        assert self.is_probs() or self.is_weights()
        f = maybe_delayed(log if self.is_probs() else negate)
        return self.from_log_probs(f(self.speech.values), f(self.silence.values))

    def to_probs(self):
        if self._mode is ParameterMode.PROB:
            return self
        f = make_array
        if self.is_weights():
            f = compose(negate, f)
        elif not self.is_log_probs():
            raise TypeError("Parameter mode is not correct")
        f = compose(norm_fwd, compose(exp, f))
        f = maybe_delayed(f)
        return self.from_fwd_probs(f(self.speech.values), f(self.silence.values))

    def apply_to_am_config(self, am_config):
        am_config.tdp[      "*"]["forward"] = self.speech.fwd
        am_config.tdp[      "*"][   "loop"] = self.speech.loop
        am_config.tdp["silence"]["forward"] = self.silence.fwd
        am_config.tdp["silence"][   "loop"] = self.silence.loop
    
    def __eq__(self, other):
        if not type(other) == type(self):
            return false
        return (
            self.speech == other.speech
            and self.silence == other.silence
            and self._mode == other._mode
        )
        
    
    def __str__(self):
        res = "Speech {}, Silence {}, Mode: {}".format(self.speech, self.silence, self._mode)
        return res


class TransitionParameterApplicator:
    def __init__(self, system):
        self.system = system

    def apply(self, transition_model: SimpleTransitionModel, corpus: str, **_ignored):
        transition_model = transition_model.to_weights()
        csp = self.system.csp
        amc = csp[corpus].acoustic_model_config
        if amc is csp["base"].acoustic_model_config:
            csp[corpus].acoustic_model_config \
                = copy.deepcopy(amc)
        amc = self.system.csp[corpus].acoustic_model_config
        transition_model.apply_to_am_config(amc)


class ExitPenaltyApplicator:
    def __init__(self, system, default_corpus="train"):
        self.system = system
        self.default_corpus = default_corpus
    
    def apply(self, silence_exit_penalty, corpus=None, **_ignored):
        corpus = corpus or self.default_corpus
        csp = self.system.csp
        amc = csp[corpus].acoustic_model_config
        if amc is csp["base"].acoustic_model_config:
            csp[corpus].acoustic_model_config \
                = copy.deepcopy(amc)
        amc = csp[corpus].acoustic_model_config
        amc.tdp["silence"].exit = silence_exit_penalty
        
class SkipApplicator:
    def __init__(self, system, default_corpus="train"):
        self.system = system
        self.default_corpus = default_corpus
    
    def apply(self, skip_penalty, corpus=None, **_ignored):
        corpus = corpus or self.default_corpus
        csp = self.system.csp
        amc = csp[corpus].acoustic_model_config
        if amc is csp["base"].acoustic_model_config:
            csp[corpus].acoustic_model_config \
                = copy.deepcopy(amc)
        amc = csp[corpus].acoustic_model_config
        amc.tdp["*"].skip = skip_penalty

class CombinedModel:
    """
    Container class that holds a transition model
    and a silence exit penalty. Primary purpose is to transform
    weights into the corresponding RASR form with the method
    to_buggy_weights.
    In the future, this class will likely be deprecated.
    """
    def __init__(self, transition_model, silence_exit, speech_skip=None, skip_normed=False):
        self.transition_model = transition_model
        self.silence_exit = silence_exit
        self.speech_skip = speech_skip
        self.skip_normed = skip_normed
    
    @classmethod
    def from_fwd_probs(cls, speech_fwd, silence_fwd, silence_exit, speech_skip=None):
        transition_model = SimpleTransitionModel.from_fwd_probs(speech_fwd, silence_fwd)
        return cls(transition_model, silence_exit, speech_skip=speech_skip)
    
    @classmethod
    def zeros(cls, silence_exit=0.0, speech_skip=None):
        transition_model = SimpleTransitionModel.from_weights([0.0, 0.0], [0.0, 0.0])
        return cls(transition_model, silence_exit, speech_skip=speech_skip)
    
    @classmethod
    def legacy(cls):
        return cls.from_weights(0.0, 3.0, 3.0, 0.0, 20.0, 30.0)
    
    def adjust(self, silence_exit=None, speech_skip=None):
        # assert all(key in self.__dict__ for key in kwargs)
        self.__dict__.update({
            key: value for key, value in locals().items() if key != "self" and value is not None
        })
        return self
    
    @classmethod
    def from_weights(cls, speech_fwd, speech_loop, silence_fwd, silence_loop, silence_exit, speech_skip=None, skip_normed=False):
        transition_model = SimpleTransitionModel.from_weights([speech_fwd, speech_loop], [silence_fwd, silence_loop])
        return cls(transition_model, silence_exit, speech_skip=speech_skip, skip_normed=skip_normed)
    
    def to_acoustic_model_extra_config(self):
        config = rasr.RasrConfig() 
        self.transition_model.to_weights().apply_to_am_config(config)
        config.tdp.silence.exit = self.silence_exit
        config.tdp["*"].skip = self.speech_skip
        config.tdp["*"].exit = 0.0
        return config
    
    def to_acoustic_model_config(self):
        config = rasr.RasrConfig() 
        assert isinstance(config, rasr.RasrConfig)
        config.tdp.scale = 1.0
        config.tdp["*"].loop = None
        config.tdp["*"].forward = None
        config.tdp["*"].skip = "infinity"
        config.tdp["*"].exit = 0.0
        config.tdp.silence.loop = None
        config.tdp.silence.forward = None
        config.tdp.silence.skip = "infinity"
        config.tdp.silence.exit = self.silence_exit
        config.tdp["entry-m1"].loop = "infinity"
        config.tdp["entry-m2"].loop = "infinity"
        self.transition_model.to_weights().apply_to_am_config(config)

        if self.skip_normed and self.speech_skip is not None:
            fwd = config.tdp["*"].forward
            config.tdp["*"].forward = fwd * (1 - self.speech_skip)
            config.tdp["*"].skip = fwd * self.speech_skip
        elif self.speech_skip:
            config.tdp["*"].skip = self.speech_skip
        assert isinstance(config, rasr.RasrConfig)
        return config
    
    def to_buggy_weights(self):
        transition_model = self.transition_model.to_weights()
        silence_fwd = transition_model.silence.fwd
        speech_fwd  = transition_model.speech.fwd
        silence_exit = self.silence_exit + silence_fwd - speech_fwd
        return type(self)(transition_model, silence_exit)

    def __str__(self):
        return str(self.transition_model) + " Silence exit: {}".format(self.silence_exit)


class CombinedModelApplicator:
    def __init__(self, system):
        self.system = system
    
    def apply(self, model: CombinedModel, corpus="train", **_ignored):
        csp = self.system.csp
        # copy acoustic model config to change value
        # exclusive to specified corpus
        amc = csp[corpus].acoustic_model_config
        if amc is csp["base"].acoustic_model_config:
            csp[corpus].acoustic_model_config = copy.deepcopy(amc)
        amc = csp[corpus].acoustic_model_config
        model.transition_model.apply_to_am_config(amc)
        amc.tdp["silence"].exit = model.silence_exit
        