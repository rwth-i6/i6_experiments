from sisyphus import *
from sisyphus.delayed_ops import DelayedBase
from sisyphus.tools import try_get

from recipe import sprint
from . import helpers
from .helpers import TuningSystem

from collections import OrderedDict

import asyncio

def split_str(s):
    return s.split("-")

def get_default_dict(kwarg):
    if kwarg is None:
        return {}
    return kwarg

class DelayedMin(DelayedBase):
    def __init__(self, args: list):
        self.args = args
    
    def get(self):
        return min(try_get(v) for v in self.args)

class DelayedArgmin(DelayedBase):
    def __init__(self, mapping: dict):
        self.mapping = mapping
    
    def get(self):
        m = self.mapping
        return min(m, key=lambda k: try_get(m.get(k)))

class Schedule:
    def __init__(self):
        self.setters = OrderedDict()
        self.params = OrderedDict()

    def register(self, name, setter, params):
        self.setters[name] = setter
        self.params[name] = params

class Tuner:
    SAMPLE_SIZE = 7
    DEFAULT_PRIOR_SCALES = [0.1, 0.2, 0.5, 0.7, 1.0]

    def __init__(
            self,
            system,
            prior_scales=None,
            prior_tdp_sample_size=SAMPLE_SIZE,
            sample_seed=0
        ):
        self.system = system
        self.ts = TuningSystem(system, {})
        self.prior_scales = self.DEFAULT_PRIOR_SCALES if not prior_scales else prior_scales
        self.results = {}

        import random
        from itertools import product
        random.seed(sample_seed)
        self.samples = random.sample(
            list(product(
                [0.3, 0.4, 0.5, 0.6, 0.7],
                [0.1, 0.5, 0.8, 1.0]
            )),
            prior_tdp_sample_size
        )

    def tune_recog(
            self,
            name,
            recognition_args=None,
            scorer_args=None,
            **nn_and_recog_args,
            ):
        recognition_args = recognition_args or {}
        scorer_args = scorer_args or {}
        nn_and_recog_args.setdefault("compile_crnn_config", name)
        def set_recog(recognition_args, scorer_args, scales):
            prior_scale, tdp_scale = scales[0], scales[1]
            recognition_args["extra_config"]["flf-lattice-tool.network.recognizer.acoustic-model.tdp"].scale = tdp_scale
            scorer_args["prior_scale"] = prior_scale

        extra_config = sprint.SprintConfig()
        extra_config["flf-lattice-tool.network.recognizer.recognizer.acoustic-lookahead-temporal-approximation-scale"] = 10
        scale_tuning_name = f"{name}_recog_scales"
        
        self.ts.tune_parameter(
            name=scale_tuning_name,
            crnn_config=None,
            parameters=self.samples,
            transformation=set_recog,
            recognition_args={
                "extra_config": extra_config,
                "search_parameters": { "beam-pruning": 14.0 }
            },
            optimize=False,
            scorer_args={},
            procedure=helpers.Recog(160, f"{name}"),
            **nn_and_recog_args
        )
        tuned_scorer_args = {}
        tuned_recog_args = {"extra_config": sprint.SprintConfig()}
        opt_scales_str = DelayedArgmin({p: v[160] for p, v in self.ts.summary_data[scale_tuning_name].items()})
        opt_scales = opt_scales_str.function(split_str)
        set_recog(tuned_recog_args, tuned_scorer_args, opt_scales)
        # tuned_name = f"{name}"
        nn_and_recog_args.setdefault("training_args", {})
        self.system.decode(
            name=name,
            epoch=160,
            recog_name=f"{name}_tuned",
            crnn_config=None,
            recognition_args={
                **recognition_args,
                **tuned_recog_args,
            },
            scorer_args={
                **scorer_args,
                **tuned_scorer_args
            },
            # training_args={},
            **nn_and_recog_args,
        )
    

    async def tune_prior(
            self,
            name,
            config,
            recog=True,
            override_scales=True,
            legacy=False,
            **nn_and_recog_args,
        ):
        from recipe.crnn.helpers.mann import pretrain, interface

        assert isinstance(config, pretrain.PretrainConfigHolder)

        def set_prior_scale(crnn_config, prior_scale):
            assert isinstance(crnn_config, pretrain.PretrainConfigHolder)
            crnn_config.prior_scale = prior_scale
        
        prior_scales = self.prior_scales
        
        self.ts.tune_parameter(
            name=name,
            crnn_config=config,
            parameters=prior_scales,
            transformation=set_prior_scale,
            delayed_build=True,
            **nn_and_recog_args
        )

        if not recog:
            return
        
        # set prior scale
        results = {p: v[160] for p, v in self.ts.wers[name].items()}
        await tk.async_run(results)

        # opt_prior_scale = DelayedArgmin({p: v[160] for p, v in ts.wers[name].items()})
        opt_prior_scale = min(results, key=results.get)

        def set_recog(recognition_args, scorer_args, scales):
            prior_scale, tdp_scale = scales
            recognition_args["extra_config"]["flf-lattice-tool.network.recognizer.acoustic-model.tdp"].scale = tdp_scale
            scorer_args["prior_scale"] = prior_scale

        def tune_recog_scales(prior_scale):
            extra_config = sprint.SprintConfig()
            extra_config["flf-lattice-tool.network.recognizer.recognizer.acoustic-lookahead-temporal-approximation-scale"] = 10
            
            self.ts.tune_parameter(
                name=f"{name}_recog_scales",
                crnn_config=None,
                parameters=self.samples,
                transformation=set_recog,
                recognition_args={
                    "extra_config": extra_config,
                    "search_parameters": { "beam-pruning": 14.0 }
                },
                optimize=False,
                scorer_args={},
                procedure=helpers.Recog(160, f"{name}-{prior_scale}"),
                **nn_and_recog_args
            )
        
        tune_recog_scales(opt_prior_scale)
        orig_name = name
        name = f"{name}_recog_scales"
        results = {p: v[160] for p, v in self.ts.summary_data[name].items()}
        await tk.async_run(results)

        opt_scales = min(results, key=results.get)
        tuned_scorer_args = {}
        tuned_recog_args = {"extra_config": sprint.SprintConfig()}
        set_recog(tuned_recog_args, tuned_scorer_args, opt_scales.split("-"))


        tuned_name = f"{orig_name}-{opt_prior_scale}"
        self.system.decode(
            name=tuned_name,
            epoch=160,
            recog_name=f"{orig_name}_tuned",
            crnn_config=None,
            recognition_args=tuned_recog_args,
            scorer_args=tuned_scorer_args,
            training_args={},
            **nn_and_recog_args,
        )
        self.results[orig_name] = {
            "WER": self.system.jobs["dev"]["scorer_crnn-" + tuned_name + "-160-optlm"].wer
        }
    
    def make_report(self, fname, **report_args):
        import os
        from recipe.setups.mann.reports import GenericReport
        tk.register_report(
            os.path.join(fname, "tuned_wers"),
            GenericReport(
                cols=["WER"],
                values=self.results,
                **report_args
            )
        )


