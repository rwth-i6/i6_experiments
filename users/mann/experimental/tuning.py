from sisyphus import *
from sisyphus.delayed_ops import DelayedBase
from sisyphus.tools import try_get

from i6_core import rasr
from i6_core.util import instanciate_delayed
from . import helpers
from .helpers import TuningSystem
from i6_experiments.users.mann.setups.nn_system import RecognitionConfig
from i6_experiments.users.mann.setups.reports import maybe_get
from i6_experiments.users.mann.setups.nn_system.base_system import NotSpecified

import os
import copy
from collections import OrderedDict, namedtuple, UserList, ChainMap
from typing import Callable, Union, Tuple, List
from inspect import signature, Parameter, _empty
from tabulate import tabulate

import asyncio
import itertools

Auto = object()

def split_str(s):
    return s.split("-")

def get_default_dict(kwarg):
    if kwarg is None:
        return {}
    return kwarg

def determine_signature(func: Callable) -> List[str]:
    return [k for k, p in signature(func).parameters.items() if p.default == _empty][:-1]

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


class Setter:
    def __init__(self, transformation, signature):
        self.transformation = transformation
        self.signature = signature
    
    @classmethod
    def from_func(cls, func):
        return cls(func, determine_signature(func))
    
    def map_args(self, **arg_map):
        def wrapper(f):
            pass
        pass
    
    def to_binary_setter(self):
        def func(*args):
            if args[-1] is True:
                return self.transformation(*args[:-1])
        return type(self)(func, self.signature) 
    
    def __call__(self, *args, **kwargs):
        self.transformation(*args, **kwargs)


class NamedValue(
    namedtuple("BaseNamedValue", ["name", "value", "param"])
):
    @property
    def children(self):
        try:
            return self.param.children[self.value].params
        except TypeError:
            return None
        # try:
        #     return self.param.children[self.value].values()
        # except AttributeError:
        #     return None
        # except KeyError:
        #     return None
    
    def to_str(self):
        if self.value is False:
            return ""
        if self.value is True:
            return self.name
        return "{}={}".format(self.name, self.value)

class ScheduleParameter:
    def __init__(
        self,
        name,
        setter,
        params=None,
        parent=None,
        children=None,
    ):
        self.name = name
        self.setter = setter
        self.params = [NamedValue(name, value, self) for value in params]
        self.parent = parent
        self.children = children

class ParameterPath(UserList):
    def __str__(self):
        return ".".join(v.to_str() for v in self)
    

class Schedule:
    def __init__(self):
        self.params = OrderedDict()

    def register(self, name, setter, params):
        self.params[name] = params
    
    def add_root(self, name, setter, params):
        self.root = ScheduleParameter(name, setter, params) 
        self.params[name] = self.root
        return self

    def add_child(self, parent, name, setter, params=None, yesno=False):
        if yesno:
            setter = setter.to_binary_setter()
            params = [False, True]
        value = None
        if isinstance(parent, tuple):
            parent, value = parent
        parent = self.params[parent]
        param = ScheduleParameter(name, setter, params, parent=parent)
        self.params[name] = param
        if value is None:
            parent.children = {value.value: param for value in parent.params}
        else:
            parent.children = {value: param}
        return self
    
    def add_node(self, name, setter, parent=None, params=None, yesno=False):
        if parent is None:
            if yesno:
                setter = setter.to_binary_setter()
                params = [False, True]
            return self.add_root(name, setter, params)
        return self.add_child(parent, name, setter, params=params, yesno=yesno)

    def get_transformation(self):
        def setter(param_path: ParameterPath, **config_pointers):
            for named_value in param_path:
                named_value.param.setter(named_value.value, **config_pointers)
        return setter
    
    def get_parameters(self):
        """Collect all parameter paths from root to leaves via depth-first search."""
        value_paths = []
        stack = [iter(self.root.params)]
        stacktrace = []
        discovered = set()
        while stack:
            print(stack)
            try:
                value: NamedValue = next(stack[-1])
                print(value)
            except StopIteration:
                stack.pop()
                stacktrace.pop()
                continue
            stacktrace.append(value)
            if value.children is None:
                value_paths.append(ParameterPath(stacktrace))
            else:
                stack.append(iter(value.children))
        return value_paths

def dump_table(output_dir, name, data, headers, optimum=None):
    output_path = os.path.join("output", output_dir, "summary", name)
    if not os.path.isdir(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    print("Dump summary {}".format(output_path))
    table_data = instanciate_delayed(data)
    with open(output_path, "w") as f:
        print("write table")
        f.write(tabulate(data, headers=headers, tablefmt="presto", floatfmt=".2f"))
        if optimum is not None:
            print("write opt")
            f.write("\noptimum: tdp, prior = {}\n".format(optimum))
    
class RecognitionTuner:
    TDP_SCALES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    TDP_REDUCED = [0.1, 0.5, 1.0]
    PRIOR_SCALES = [0.1, 0.3, 0.5, 0.7]

    BASE_CONFIG = RecognitionConfig(
        beam_pruning=16,
        beam_pruning_threshold=NotSpecified,
        altas=2.0,
    )

    class Setter:
        def __init__(self, args):
            self.args = {}
        
        def set_args(self, **kwargs):
            self.args.update(kwargs)
            return self
        
    class TdpPriorSetter(Setter):
        def __call__(self, tdp, prior):
            self.args["recognition_args"] = None

    def __init__(
        self,
        system,
        tdp_scales=None,
        prior_scales=None,
        base_config=None,
        all_scales=None,
    ):
        # assert isinstance(system, RecognitionSystem)
        assert (tdp_scales and prior_scales) != bool(all_scales)
        self.system = system
        self.tdp_scales = tdp_scales or self.TDP_SCALES
        self.prior_scales = prior_scales or self.PRIOR_SCALES
        self._all_scales = all_scales
        self.base_config = base_config or self.BASE_CONFIG
    
    @property
    def all_scales(self):
        if self._all_scales:
            return self._all_scales
        return itertools.product(self.tdp_scales, self.prior_scales)
    
    def print_report(self, name, data):
        table_data = []
        sorted_priors = sorted(self.prior_scales)
        for tdp in sorted(self.tdp_scales):
            row = [tdp]
            for prior in sorted_priors:
                wer = data[(tdp, prior)]
                row.append(wer)
            table_data.append(row)
        
        tk.register_callback(
            dump_table,
            output_dir=self.output_dir,
            name=name + "_tuning.txt",
            data=table_data,
            headers=["TDP \\ Prior"] + sorted_priors,
            optimum=self.calc_optimum(data),
        )
    
    def calc_optimum(self, data):
        return DelayedArgmin(data)

    def run_decode(
        self,
        name,
        recognition_config,
        epoch,
        decoding_args,
        returnn_config,
        exp_config,
        params,
        quick=False,
        extra_suffix="",
        **kwargs
    ):
        tdp, prior = params
        optimize = True
        recognition_config = recognition_config.replace(
            tdp_scale=tdp,
            prior_scale=prior,
        )
        recog_name="{}{}.tuned".format(name, extra_suffix)
        if quick:
            recognition_config = recognition_config.replace(
                beam_pruning=self.base_config.beam_pruning,
                altas=self.base_config.altas,
                beam_pruning_threshold=self.base_config.beam_pruning_threshold,
            )
            optimize = False
            recog_name="{}{}.tune_recog.tdp-{}.prior-{}".format(name, extra_suffix, tdp, prior)
        self.system.run_decode(
            name,
            epoch=epoch,
            recog_name=recog_name,
            optimize=optimize,
            recognition_args=recognition_config.to_dict(prefix="tune_recog/" if quick else ""),
            crnn_config=returnn_config,
            exp_config=exp_config,
            **kwargs,
        )
        return recog_name

    async def tune(
        self,
        name,
        epoch,
        recognition_config,
        exp_config,
        decoding_args=None,
        optimum=None,
        returnn_config=None,
        extra_suffix="",
        prior_suffix=None,
        print_report=False,
    ):
        prior = False
        decoding_args = copy.deepcopy(decoding_args)
        if exp_config.reestimate_prior == "CRNN":
            exp_config = exp_config.replace(
                reestimate_prior=False
            ).extend(
                scorer_args={"prior_file": name}
            )
        data = {}
        for mscale in self.all_scales:
            recog_name = self.run_decode(
                name,
                recognition_config=recognition_config,
                epoch=epoch,
                decoding_args=decoding_args,
                returnn_config=returnn_config,
                exp_config=exp_config,
                params=mscale,
                quick=True,
                extra_suffix=extra_suffix,
            )
            data[mscale] = self.system.get_wer(recog_name, epoch, optlm=False, precise=True, prior=prior_suffix)
        
        if print_report:
            self.print_report(name, data)
        
        if not optimum:
            return
        
        sync = True
        if optimum == "async":
            sync = False

        if not isinstance(optimum, tuple):
            optimum = self.calc_optimum(data) 
        
        if not sync:
            await tk.async_run(optimum)

        # hopefully ready
        optimum = optimum.get()

        recog_name = self.run_decode(
            name,
            recognition_config=recognition_config,
            epoch=epoch,
            decoding_args=decoding_args,
            returnn_config=returnn_config,
            exp_config=exp_config,
            params=optimum,
        )

    def tune(
        self,
        name,
        epoch,
        recognition_config,
        exp_config,
        decoding_args=None,
        optimum=None,
        returnn_config=None,
        extra_suffix="",
        prior_suffix=None,
        print_report=False,
    ):
        prior = False
        decoding_args = copy.deepcopy(decoding_args)
        if exp_config.reestimate_prior == "CRNN":
            exp_config = exp_config.replace(
                reestimate_prior=False
            ).extend(
                scorer_args={"prior_file": name}
            )
        data = {}
        for mscale in self.all_scales:
            recog_name = self.run_decode(
                name,
                recognition_config=recognition_config,
                epoch=epoch,
                decoding_args=decoding_args,
                returnn_config=returnn_config,
                exp_config=exp_config,
                params=mscale,
                quick=True,
                extra_suffix=extra_suffix,
            )
            data[mscale] = self.system.get_wer(recog_name, epoch, optlm=False, precise=True, prior=prior_suffix)
        
        if print_report:
            self.print_report(name, epoch, data)
        
        if not optimum:
            return
        
        if not isinstance(optimum, tuple):
            optimum = self.calc_optimum(data) 
            optimum = optimum.get()
        
        # hopefully ready
        # optimum = optimum.get()

        recog_name = self.run_decode(
            name,
            recognition_config=recognition_config,
            epoch=epoch,
            decoding_args=decoding_args,
            returnn_config=returnn_config,
            exp_config=exp_config,
            params=optimum,
            extra_suffix=extra_suffix,
        )
        return self.system.get_wer(recog_name, epoch, optlm=True, precise=False, prior=prior_suffix)

    async def tune_async(
        self,
        name,
        epoch,
        recognition_config,
        exp_config,
        decoding_args=None,
        optimum="async",
        returnn_config=None,
        extra_suffix="",
        prior_suffix=None,
        print_report=False,
        force_existing_prior=False,
    ):
        prior = False
        decoding_args = copy.deepcopy(decoding_args)
        if exp_config.reestimate_prior == "CRNN" and force_existing_prior:
            exp_config = exp_config.replace(
                reestimate_prior=False
            ).extend(
                scorer_args={"prior_file": name}
            )
        data = {}
        for mscale in self.all_scales:
            recog_name = self.run_decode(
                name,
                recognition_config=recognition_config,
                epoch=epoch,
                decoding_args=decoding_args,
                returnn_config=returnn_config,
                exp_config=exp_config,
                params=mscale,
                quick=True,
                extra_suffix=extra_suffix,
            )
            data[mscale] = self.system.get_wer(recog_name, epoch, optlm=False, precise=True, prior=prior_suffix)
        
        if print_report:
            self.print_report(name + extra_suffix, data)
        
        if not optimum:
            return
        
        sync = True
        if optimum == "async":
            sync = False

        if not isinstance(optimum, tuple):
            optimum = self.calc_optimum(data)
        
        if not sync:
            await tk.async_run(optimum)

        # hopefully ready
        optimum = optimum.get()

        recog_name = self.run_decode(
            name,
            recognition_config=recognition_config,
            epoch=epoch,
            decoding_args=decoding_args,
            returnn_config=returnn_config,
            exp_config=exp_config,
            params=optimum,
            extra_suffix=extra_suffix,
        )



class FactoredHybridTuner(RecognitionTuner):
    FWD_LOOP_SCALES = [0.1, 0.3, 0.5, 0.7, 1.0]

    def __init__(
        self,
        system,
		tdp_scales=None,
		prior_scales=None,
		fwd_loop_scales=None,
		all_scales=None,
		base_config=None
    ):
        assert (tdp_scales and prior_scales and fwd_loop_scales) != bool(all_scales)
        super().__init__(system, tdp_scales, prior_scales, base_config, all_scales=all_scales)
        self.fwd_loop_scales = fwd_loop_scales or self.FWD_LOOP_SCALES
    
    @property
    def all_scales(self):
        if self._all_scales:
            return self._all_scales
        return itertools.product(self.tdp_scales, self.prior_scales, self.fwd_loop_scales)
    
    def run_decode(
        self,
        name,
        recognition_config,
        epoch,
        decoding_args,
        returnn_config,
        exp_config,
        params,
        quick=False,
        extra_suffix="",
        **kwargs
    ):
        tdp, prior, fwd_loop = params
        recognition_config = recognition_config.replace(tdp_scale=tdp)
        recog_name="{}{}.tuned".format(name, extra_suffix)
        optimize = True
        if quick:
            recognition_config = recognition_config.replace(
                beam_pruning=self.base_config.beam_pruning,
                altas=self.base_config.altas,
                beam_pruning_threshold=self.base_config.beam_pruning_threshold,
            )
            optimize = False
            recog_name="{}{}.tune_recog.fwd_loop-{}.tdp-{}.prior-{}".format(name, extra_suffix, fwd_loop, tdp, prior)
        self.system.run_decode(
            name,
            epoch=epoch,
            recog_name=recog_name,
            type="fh",
            optimize=optimize,
            decoding_args=decoding_args,
            recognition_args=recognition_config.to_dict(prefix="tune_recog/" if quick else ""),
            crnn_config=returnn_config,
            extra_suffix="",
            exp_config=exp_config.extend(
                scorer_args={
                    "fwd_loop_scale": fwd_loop,
                    "prior_scale": prior,
                }
            ),
            clean=True,
            **kwargs,
        )
        return recog_name
    
    def print_report(self, name, data):
        def make_table():
            lines = []
            table = []
            for tdp, prior, fwd_loop in self.all_scales:
                table.append([tdp, prior, fwd_loop, maybe_get(data[(tdp, prior, fwd_loop)])])
            lines.append(tabulate(table, headers=["tdp", "prior", "fwd_loop", "WER"], tablefmt="presto"))
            lines.append("")
            lines.append("Optimum found at: {}".format(maybe_get(self.calc_optimum(data))))
            return "\n".join(lines)
        
        tk.register_report(
            os.path.join(self.output_dir, "summary", name + "_tuning.txt"),
            make_table,
        )
    
    async def tune_async(
        self,
        name,
        epoch,
        recognition_config,
        decoding_args,
        exp_config,
        optimum="async",
        returnn_config=None,
        extra_suffix="",
        prior_suffix=None,
        print_report=False,
        force_existing_prior=False,
        flf_tool_exe=None,
    ):
        if flf_tool_exe is not None:
            self.system.crp["dev"] = copy.deepcopy(self.system.crp["dev"])
            self.system.crp["dev"].flf_tool_exe = flf_tool_exe
        prior = False
        decoding_args = copy.deepcopy(decoding_args)
        if exp_config.reestimate_prior == "CRNN" and force_existing_prior:
            exp_config = exp_config.replace(
                reestimate_prior=False
            ).extend(
                scorer_args={"prior_file": name}
            )
        data = {}
        for mscale in self.all_scales:
            recog_name = self.run_decode(
                name,
                recognition_config=recognition_config,
                epoch=epoch,
                decoding_args=decoding_args,
                returnn_config=returnn_config,
                exp_config=exp_config,
                params=mscale,
                quick=True,
                extra_suffix=extra_suffix,
            )
            data[mscale] = self.system.get_wer(recog_name, epoch, optlm=False, precise=True, prior=prior_suffix)
        
        if print_report:
            self.print_report(name + extra_suffix + "-{}".format(epoch), data)
        
        if not optimum:
            return
        
        sync = True
        if optimum == "async":
            sync = False

        if not isinstance(optimum, tuple):
            optimum = self.calc_optimum(data) 
        
        if not sync:
            await tk.async_run(optimum)
            optimum = optimum.get()

        recog_name = self.run_decode(
            name,
            recognition_config=recognition_config,
            epoch=epoch,
            decoding_args=decoding_args,
            returnn_config=returnn_config,
            exp_config=exp_config,
            extra_suffix=extra_suffix,
            params=optimum,
        )


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


