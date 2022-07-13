import copy
import functools

import numpy as np

from i6_core import returnn
import i6_experiments.users.mann.nn.bw as bw

class WrapEpochValue(returnn.CodeWrapper):
    def __init__(self, f):
        super().__init__("WrapEpochValue({})".format(f))

def eval_code_wrapper(wrapper: returnn.CodeWrapper):
    def WrapEpochValue(l):
        return l(0)
    return eval(wrapper.code)

@functools.singledispatch
def remove_pretrain(config):
    cls = config.__class__
    tp = type(config)
    raise ValueError(
        "Config must be of instance dict or bw.ScaleConfig"
        + " instead found {}".format(cls)
    )

@remove_pretrain.register(dict)
def _(config: dict):
    assert not isinstance(config, bw.ScaleConfig)
    # delete general args
    for k in list(config.keys()):
        if k.startswith("pretrain"):
            del config[k]
    config.pop("extra_python", None)

@remove_pretrain.register(bw.ScaleConfig)
def _(config: bw.ScaleConfig):
    # delete general args
    for k in list(config.keys()):
        if k.startswith("pretrain"):
            del config[k]
    # filter out WrapEpochValues
    prolog = getattr(config, "python_prolog", None)
    if prolog:
        config.python_prolog = tuple(filter(
            lambda s: not isinstance(s, str) or "Pretrain" not in s, prolog
        ))
    for scale in ["prior", "am"]:
        attr = f"{scale}_scale"
        value = getattr(config, attr)
        if isinstance(value, returnn.CodeWrapper):
            value = eval_code_wrapper(value)
            setattr(config, attr, value)

class ExponentialScaleWarmup:
    """ Class for setting various scales and generating setter methods
    to use with TuningSystem. For now focusses on generating an exponential
    pretrain schedule interpolating between the given initial and final
    acoustic model scale. """
    def __init__(self, initial_am, final_am, final_epoch,
            rel_prior_scale=1.0, prior_scale=None, absolute_scale=None):
        assert rel_prior_scale is None or prior_scale is None
        self.initial_am = initial_am
        self.final_am = final_am
        self.final_epoch = final_epoch
        self.rel_prior_scale = rel_prior_scale
        self.abs_prior_scale = prior_scale
        self.absolute_scale = absolute_scale
        self.pretrain = True
    
    def is_prior_relative(self):
        return self.rel_prior_scale is not None

    @classmethod
    def parse_legacy_config(cls, config):
        final_am = config.am_scale
        final_prior = config.prior_scale
        rel_prior_scale = final_prior / final_am
        def WrapEpochValue(l):
            return l(1)
        if "extra_python" in config:
            code = config["extra_python"]
        else:
            code = config.extra_python_code
        assert code != "", "Config does not seem to contain pretrain code"
        am_pretrain_code = next(l for l in code.split("\n") if "am_scale" in l)
        initial_am = eval(am_pretrain_code.split("=")[-1])
        final_epoch = config["pretrain_repetitions"]["final"]
        return cls(initial_am, final_am, final_epoch, rel_prior_scale)

    @classmethod
    def init_from_config(cls, config):
        final_am = config.am_scale
        final_prior = config.prior_scale
        try:
            rel_prior_scale = final_prior / final_am
        except TypeError:
            final_am = eval_code_wrapper(final_am)
            final_prior = eval_code_wrapper(final_prior)
        def WrapEpochValue(l):
            return l(1)
        if "extra_python" in config:
            code = config["extra_python"]
        else:
            code = config.extra_python_code
        assert code != "", "Config does not seem to contain pretrain code"
        am_pretrain_code = next(l for l in code.split("\n") if "am_scale" in l)
        initial_am = eval(am_pretrain_code.split("=")[-1])
        final_epoch = config["pretrain_repetitions"]["final"]
        return cls(initial_am, final_am, final_epoch, rel_prior_scale)
    
    @property
    def prior_scale(self):
        if self.abs_prior_scale is not None:
            return self.prior_scale
        return self.final_am * self.rel_prior_scale
    
    @prior_scale.setter
    def prior_scale(self, value):
        if self.abs_prior_scale is not None:
            self.prior_scale = value
        self.rel_prior_scale = value
    
    @property
    def k_scale(self):
        if self.absolute_scale is None:
            return 1.0
        return self.absolute_scale
    
    def generate_exp_am_schedule(self, precision=6):
        l = np.log(self.final_am / self.initial_am) / (self.final_epoch - 1)
        f = lambda x: round(self.initial_am * np.exp(l * x), precision)
        return list(map(f, range(self.final_epoch)))

    @staticmethod
    def wev_template(sched, k_scale=None, prior_scale=None):
        buf = "WrapEpochValue(lambda epoch: "
        if k_scale is not None and k_scale != 1.0:
            buf += "{k:.2f} * ".format(k=k_scale)
        if prior_scale is not None and prior_scale != 1.0:
            buf += "{pr:.2f} * ".format(pr=prior_scale)
        buf += "{sched}[epoch - 1])".format(sched=sched)
        return returnn.CodeWrapper(buf)
    
    def has_absolute_scale(self):
        return isinstance(self.absolute_scale, (float, int)) and self.absolute_scale != 1.0

    @staticmethod
    def scaled_am_pretrain_code(am_scales, rel_prior_scale=1.0, k_scale=1.0):
        if k_scale is None:
            k_scale = 1.0
        pretrain_code = """from Pretrain import WrapEpochValue
network['combine_prior']['eval_locals']['am_scale']    = WrapEpochValue(lambda epoch: {k:.2f} * {ams}[epoch - 1])
network['combine_prior']['eval_locals']['prior_scale'] = WrapEpochValue(lambda epoch: {k:.2f} * {prs:.2f} * {ams}[epoch - 1])
locals().update(**config)""".format(ams=am_scales, prs=prior_scale, k=k_scale) 
        return pretrain_code
    
    def get_legacy_pretrain_code(self, am_scales):
        assert self.rel_prior_scale and not self.abs_prior_scale
        pretrain_code = """from Pretrain import WrapEpochValue
network['combine_prior']['eval_locals']['am_scale']    = WrapEpochValue(lambda epoch: {k:.2f} * {ams}[epoch - 1])
network['combine_prior']['eval_locals']['prior_scale'] = WrapEpochValue(lambda epoch: {k:.2f} * {prs:.2f} * {ams}[epoch - 1])
locals().update(**config)""".format(ams=am_scales, prs=self.rel_prior_scale, k=self.k_scale) 
        return pretrain_code


    def set_config(self, config, legacy=False, override_scales=False):
        assert isinstance(config, returnn.ReturnnConfig)
        if override_scales:
            config.prior_scale = self.prior_scale
            config.am_scale = self.final_am
        assert -1e5 < config.prior_scale - self.prior_scale < 1e5, "Jump from final prior scale to default static prior scale"
        assert config.am_scale == self.final_am, "Jump from final am scale to default static am scale"
        schedule = self.generate_exp_am_schedule()
        if legacy:
            assert config.extra_python_code == ""
            # code = self.scaled_am_pretrain_code(schedule, prior_scale=self.prior_scale, k_scale=self.absolute_scale)
            code = self.get_legacy_pretrain_code(schedule)
            if config.get("extra_python", False):
                config["extra_python"] = code
            else:
                config.extra_python_code = code
            config['pretrain_repetitions']['final'] = self.final_epoch
            return
        config.python_prolog = getattr(config, "python_prolog", ()) + ("from Pretrain import WrapEpochValue",)
        config.prior_scale = ExponentialScaleWarmup.wev_template(schedule, self.absolute_scale, self.rel_prior_scale)
        config.am_scale = ExponentialScaleWarmup.wev_template(schedule, self.absolute_scale)
        if self.has_absolute_scale():
            config.tdp_scale *= self.absolute_scale
        config.config['pretrain'] = {'repetitions': {'default': 0, 'final': self.final_epoch}, "construction_algo": "no_network_modifications"}
        # config['pretrain_repetitions']['final'] = self.final_epoch
        

from collections import UserDict
class PretrainConfigHolder(UserDict):
    # def __init__(self, rel_prior_scale=1.0):
    #     self.rel_prior_scale = rel_prior_scale
    def __init__(
        self,
        config,
        rel_prior_scale=1.0,
        initial_am=0.01,
        final_epoch=5,
        init_from_config=False,
        legacy=False,
        **warmup_args,
    ):
        self.warmup = ExponentialScaleWarmup(
            initial_am,
            config.am_scale,
            final_epoch,
            rel_prior_scale,
            **warmup_args
        )
        self.config = config
        self.legacy = legacy
        super().__init__(config.config)
    
    @property
    def prior_scale(self):
        value = self.config.prior_scale
        assert value == self.esw.prior_scale
        return value
    
    @prior_scale.setter
    def prior_scale(self, prior_scale):
        config_value = prior_scale
        if self.warmup.is_prior_relative():
            config_value *= self.config.am_scale
        self.config.prior_scale = config_value
        self.warmup.prior_scale = prior_scale

    @classmethod
    def copy_from_config(
            cls,
            config,
            rel_prior_scale=1.0,
            initial_am=0.01,
            final_epoch=5,
        ):
        config = copy.deepcopy(config)
        assert isinstance(config, bw.ScaleConfig)
        config.__class__ = cls
        config.rel_prior_scale = rel_prior_scale
        config.initial_am = initial_am
        config.final_epoch = final_epoch
        return config
    
    @classmethod
    def copy_init_from_config(
        cls, config, legacy=True
    ):
        assert legacy
        # config = copy.deepcopy(config)
        config = bw.ScaleConfig.from_config(config)
        # config.make_prior_relative()
        esw = ExponentialScaleWarmup.parse_legacy_config(config)
        res = cls(config, legacy=legacy)
        res.warmup = esw
        return res
    
    def build(self):
        config = copy.deepcopy(self.config)
        self.warmup.set_config(config, self.legacy)
        if self.legacy:
            return config.config
        return config
