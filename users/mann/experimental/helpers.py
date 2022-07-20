from sisyphus import *

import copy
import operator
import numpy as np

from typing import Callable, Union, Tuple, List
from inspect import signature, Parameter, _empty
from collections.abc import Iterable
from contextlib import contextmanager
from functools import reduce

# import recipe.summary as summary
import i6_core.meta as meta
import i6_core.am as am
# from recipe.mm.tpd import TdpFromAlignment
# from recipe.experimental.mann.alignment_evaluation import MultipleEpochAlignmentStatisticsJob
# from recipe.experimental.mann.sequence_training import add_fastbw_configs
# from recipe.experimental.mann.extractors import LearningRates, AlignmentScore, SummaryJob, TpdSummary

Default = object()
Auto    = object()

def determine_signature(func: Callable) -> List[str]:
    # return list(signature(func).parameters.keys())[:-1]
    return [k for k, p in signature(func).parameters.items() if p.default == _empty][:-1]

class Procedure:
    tag: str = None

class Train(Procedure):
    tag = "train"

class Recog(Procedure):
    tag = "recog"
    def __init__(self, epoch, name):
        self.epoch = epoch
        self.name = name

class TuningSystem:

    NoTransformation = object()

    def __init__(
            self, system, training_args,
            scorer_args={}, recognition_args={},
            alignment_args={}, baselines=None,
            compile_args={},
            epochs=Default
        ):
        self.system            = system
        self.training_args     = copy.deepcopy(training_args)
        self.scorer_args       = copy.deepcopy(  scorer_args)
        self.recognition_args  =             recognition_args
        self.alignment_args    =               alignment_args
        self.compile_args      =                 compile_args
        self.alignment_bundles     = {}
        self.alignments            = {}
        self.configs               = {}
        self.training_arg_branches = {}
        self.am_configs            = {}
        self.summary_data          = {}
        self.optlm_summary_data    = {}
        self.call_args             = {}
        self.wers                  = {}
        self.baselines             = baselines or {}
        self.epochs                = epochs if epochs is not Default else system.default_epochs
    
    @contextmanager
    def system_branch(self, name, ps):
        temp_am = self.system.csp['train'].acoustic_model_config
        self.system.csp['train'].acoustic_model_config = self.am_configs[name][ps]
        yield self.system
        self.system.csp['train'].acoustic_model_config = temp_am

    @contextmanager
    def experiment_branch(self, name, ps):
        temp_am = self.system.csp['train'].acoustic_model_config
        self.system.csp['train'].acoustic_model_config = self.am_configs[name][ps]
        yield self.system, self.training_arg_branches[name][ps], self.configs[name][ps]
        self.system.csp['train'].acoustic_model_config = temp_am

    def duplicate_nn(self, crnn_config, prefix, rg, tuning_param, 
                     summary_data, epochs=[12, 24, 32], summary_epoch=24, 
                     legacy_keep_epoch_handling=True, id_suppression=False,
                     training_args=None,
                     **kwargs):
        assert (rg == 1) == id_suppression, "Cannot save multiple experiments if ids are suppressed"
        if legacy_keep_epoch_handling:
            crnn_config['cleanup_old_models']['keep'] = epochs
        if not training_args:
            training_args = self.training_args
        # recognition_args = kwargs.pop('recognition_args', self.recognition_args)
        # scorer_args = kwargs.pop('scorer_args', self.scorer_args)
        extra_args = {args: getattr(self, args, {}) for args in ["scorer_args", "recognition_args", "compile_args"]}
        extra_args.update(kwargs)
        training_args['num_epochs'] = max(epochs + [training_args.get("num_epochs") or 1])
        for id in range(rg):
            if not id_suppression:
                crnn_config['test_id'] = id
                nn_name_id = prefix + '_id-%d' % id
            else: 
                nn_name_id = prefix

            
            self.system.nn_and_recog(name=nn_name_id,
                                     training_args=training_args,
                                     crnn_config=crnn_config,
                                    #  scorer_args=scorer_args,
                                    #  recognition_args=recognition_args,
                                     epochs=epochs, use_tf_flow=True, **extra_args)
            
            suffix = "-prior" if kwargs.get('reestimate_prior', '') == 'CRNN' else ""
            for key in summary_data:
                for epoch in  epochs:
                    summary_data[key][epoch] = getattr(
                        self._get_scorer(
                            nn_name_id, epoch, optlm=(key in ["optlm", "wers"]),
                            reestimate_prior='reestimate_prior' in kwargs
                        ), "wer", None
                    )
    
    def call_procedure(self, procedure, **kwargs):
        assert isinstance(procedure, Procedure)
        def not_implemented_procedure(**kwargs):
            raise NotImplementedError("Procedure not implemented yet")
        procedures = {
            "train": self.system.nn_and_recog,
            "recog": self.system.decode,
            "align": not_implemented_procedure
        }
        if isinstance(procedure, Recog):
            kwargs["epoch"] = procedure.epoch
            kwargs["recog_name"] = kwargs["name"]
            kwargs["name"] = procedure.name
        procedures[procedure.tag](**kwargs)
                
    def tune_parameter(
            self, name: str, crnn_config: dict, parameters: list, 
            transformation: Union[Callable[[meta.System, dict, any], any], str],
            epochs=Default, signature=Auto, auto_summary=False,
            alignment_epoch=None, alignment_summary=False,
            alignment_epochs=[], alignment_parameters=[],
            correct_param_mapping=False, training_args=None,
            parameter_representation=None, parameter_mapping=None,
            procedure=Train(),
            # dryrun=False,
            **kwargs
        ):
        """ Devises an nn tuning experiment.

        :param name: experiment name
        :param crnn_config: baseline config to be tuned
        :param parameters: list of parameters to be tested
        :param transformation(crnn_config, parameter): transformation
            description of the crnn config; may return a custom list of epochs for evaluation
        :return: None
        """
        crnn_config_base = copy.deepcopy(crnn_config)
        if name not in self.configs:
            self.configs[name]               = {'base': crnn_config_base}
        self.training_arg_branches[name] = {} if not training_args \
            else {'base': copy.deepcopy(training_args)}
        for d in [self.am_configs, self.alignments, self.summary_data, self.optlm_summary_data, self.wers]:
            if name not in d:
                d[name] = {}
        if training_args is None:
            training_args = self.training_args
        if isinstance(transformation, str):
            try:
                transformation = globals()[transformation]
            except KeyError:
                raise ValueError("Could not find function with name {}".format(transformation))
        assert callable(transformation), "Transformation is not callable or " \
            + "no callable function with name {} was found".format(transformation)
        
        if signature is Auto:
            signature = determine_signature(transformation)
        if epochs is Default:
            epochs = self.epochs
        if isinstance(procedure, Recog):
            epochs = [procedure.epoch]

        suffix = "-prior" if 'reestimate_prior' in kwargs and kwargs['reestimate_prior'] == 'CRNN' else ""

        base_kwargs = kwargs

        with tk.block(name + ' tuning'):
            for parameter in parameters:
                crnn_config_branch   = copy.deepcopy(crnn_config_base)
                training_args_branch = copy.deepcopy(training_args)
                alignment_args       = copy.deepcopy(self.alignment_args)
                kwargs = copy.deepcopy(base_kwargs)
                # see if non standard signature present
                args = self._pass_signature(
                    signature,
                    config=crnn_config_branch,
                    alargs=alignment_args,
                    targs=training_args_branch,
                    **kwargs
                )
                # get parameter string or switch roles of mapping
                # ps = self._get_parameter_string(parameter, parameter_mapping=parameters)
                ps = TuningSystem.get_parameter_string(parameter, parameters, parameter_representation)
                if isinstance(parameters, dict):
                    ps, parameter = parameter, ps
                if parameter_mapping is not None:
                    parameter = parameter_mapping(parameter)

                # assign new parameter to config via transformation
                if parameter is not self.NoTransformation:
                    transformation(*args, parameter)
                
                prefix = '{name}-{parameter}'\
                           .format(name=name, parameter=ps)
                
                self.summary_data[name][ps] = {}
                self.optlm_summary_data[name][ps] = {}
                self.wers[name][ps] = {}

                # duplicated nn training
                # self.duplicate_nn(
                #     crnn_config=crnn_config_branch,
                #     prefix=prefix,
                #     rg=rng,
                #     tuning_param=parameter,
                #     summary_data={
                #         "default": self.summary_data[name][ps],
                #         "optlm"  : self.optlm_summary_data[name][ps],
                #         "wers"   : self.wers[name][ps]
                #     },
                #     epochs=epochs, 
                #     legacy_keep_epoch_handling=legacy_keep_epoch_handling,
                #     id_suppression=id_suppression,
                #     training_args=training_args_branch,
                #     **kwargs
                # )

                self.call_procedure(
                    procedure,
                    name=prefix,
                    crnn_config=crnn_config_branch,
                    epochs=epochs, 
                    training_args=training_args_branch,
                    **kwargs
                )

                # self.system.nn_and_recog(
                #     name=prefix,
                #     crnn_config=crnn_config_branch,
                #     epochs=epochs, 
                #     training_args=training_args_branch,
                #     **kwargs
                # )
                summary_data = {
                    "default": self.summary_data[name][ps],
                    "optlm"  : self.optlm_summary_data[name][ps],
                    "wers"   : self.wers[name][ps]
                }
                # suffix = "-prior" if kwargs.get('reestimate_prior', '') == 'CRNN' else ""
                for key in summary_data:
                    for epoch in  epochs:
                        # if key in ["optlm"]
                        try:
                            wer = self._get_scorer(
                                prefix, epoch, optlm=(key in ["optlm", "wers"]),
                                reestimate_prior='reestimate_prior' in kwargs
                            ).out_wer
                        except KeyError:
                            assert key in ["optlm", "wers"], "Something went wrong"
                            wer = None
                        summary_data[key][epoch] = wer
                self.configs[name][ps]               = crnn_config_branch
                self.training_arg_branches[name][ps] = training_args_branch # copy.deepcopy(self.training_args)
                self.am_configs[name][ps]            = copy.deepcopy(self.system.csp['train'].acoustic_model_config)
                
                if alignment_epoch:
                    suffix = "-prior" if 'reestimate_prior' in kwargs and kwargs['reestimate_prior'] == 'CRNN' else ""
                    self.alignment_bundles[ps] = \
                        self.system.nn_align(nn_name='{}_id-0'.format(prefix),
                                             crnn_config=crnn_config_branch,
                                             epoch=alignment_epoch,
                                             scorer_suffix=suffix)

                suffix = "-prior" if 'reestimate_prior' in kwargs and kwargs['reestimate_prior'] == 'CRNN' else ""
                self.alignments[name][ps] = {
                    epoch: (self.system.nn_align(nn_name='{}'.format(prefix), 
                        crnn_config=crnn_config_branch, 
                        epoch=epoch, scorer_suffix=suffix,
                        **alignment_args),
                        self.system.jobs['train']['alignment_{}-{}'.format(prefix, epoch)])
                    for epoch in alignment_epochs
                    if ps in map(self._get_parameter_string, alignment_parameters)
                }
            
            # choose correct identifiers
            if isinstance(parameters, dict) and not correct_param_mapping:
                parameters = parameters.values()
            # make summary automatically on request
            options = {'epoch', 'optlm', False}
            assert auto_summary in options or auto_summary.issubset(options)
            if auto_summary and (auto_summary == 'epoch' or 'epoch' in auto_summary):
                self.summary(name=name, epochs=epochs, parameters=parameters)
            if auto_summary and (auto_summary == 'optlm' or 'optlm' in auto_summary):
                self.summary(name=name, epochs=epochs, parameters=parameters, optlm=True)
            
            if isinstance(alignment_parameters, dict) and not correct_param_mapping:
                alignment_parameters = alignment_parameters.values()
            if alignment_summary:
                assert len(alignment_epochs) > 0
                self.alignment_summary(name, alignment_epochs, alignment_parameters)
            
    def tune_alignment_parameter(self, name, parameter, epoch,
        	tuning_name, tuning_parameters, transformation, 
            scorer_suffix='', signature=['system']):
        self.alignments[tuning_name] = {}
        prefix = '{name}'.format(name=name)
        nn_name = "{}-{}_id-0".format(prefix, parameter)
        # reuse system branch
        with self.system_branch(name, str(parameter)) as s:
            sigmap = {
                'system': s,
                'scorer': s.feature_scorers['train']['{}-{}{}'.format(nn_name, epoch, scorer_suffix)]
            }
            for p in tuning_parameters:
                # apply transformation
                transformation(*[sigmap[sig] for sig in signature], p)
                # get parameter reprensentation
                ps = self._get_parameter_string(p, tuning_parameters)
                # align and save in dictionary
                self.alignments[tuning_name][ps] = {}
                alignment_name = '{}-{}'.format(tuning_name, ps)
                self.alignments[tuning_name][ps][epoch] = s.nn_align(
                    nn_name=nn_name,
                    crnn_config=self.configs[name][parameter],
                    epoch=epoch,
                    scorer_suffix=scorer_suffix,
                    name=alignment_name
                )
                s.jobs['train']['alignment_{}'.format(alignment_name)].add_alias(alignment_name)
                
    def _get_report_dir(self, name, epoch, parameter=None, optlm=True, reestimate_prior=False):
        if parameter is not None:
            name = name + "-{}".format(parameter)
        jname = f'scorer_crnn-{name}-{epoch}'
        if reestimate_prior: jname += '-prior'
        if optlm:            jname += '-optlm'
        scorer = self.system.jobs['dev'][jname]
        return scorer.report_dir

    def _get_scorer(self, name, epoch, parameter=None, optlm=True, reestimate_prior=False):
        if parameter is not None:
            name = name + "-{}".format(parameter)
        jname = f'scorer_crnn-{name}-{epoch}'
        if reestimate_prior: jname += '-prior'
        if optlm:            jname += '-optlm'
        scorer = self.system.jobs['dev'][jname]
        return scorer

    def _pass_signature(self, signature, config, alargs, targs, **kwargs):
        """ Utility funtion for handling custom signatures. """
        mapping = {
            'system':        self.system,
            'targs' :              targs,
            'training_args':       targs,
            'config':             config,
            'crnn_config':        config,
            'scorer':   self.scorer_args, 
            'scorer_args': self.scorer_args,
            'epochs':               None,
            'amargs':    self.am_configs, 
            'alargs':             alargs,
            **kwargs
        }
        return [mapping[s] for s in signature]

    def _get_parameter_string(self, parameter, parameter_mapping=None):
        if isinstance(parameter_mapping, dict):
            return parameter_mapping[parameter]
        if isinstance(parameter, tuple):
            return '-'.join(list(map(str, parameter)))
        else:
            return str(parameter)

    @staticmethod
    def get_parameter_string(parameter, parameters=None, parameter_mapping=None):
        if parameter is TuningSystem.NoTransformation:
            return "none"
        if parameter_mapping is not None:
            return parameter_mapping(parameter)
        if isinstance(parameters, dict):
            return parameters[parameter]
        if isinstance(parameter, tuple):
            return '-'.join(list(map(str, parameter)))
        else:
            return str(parameter)

    def summary(self, name, epochs, parameters, parameter_names=None, epoch_decoupling=False, optlm=True,
                baseline=True, latex=False):
        data = self.summary_data[name]
        ps   = list(map(self._get_parameter_string, parameters))
        id   = 0

        if optlm:
            data = self.optlm_summary_data[name]
        if not parameter_names:
            parameter_names = name
        if baseline:
            baseline_data = {
                name: { epoch: self._get_scorer(name, epoch, optlm=optlm).wer for epoch in epochs}
                for name in self.baselines
            }
            data.update(baseline_data)
        
        summary = SummaryJob(data, "{} \\ epochs".format(parameter_names), latex=latex)
        tk.register_output('{}{}-tuning_summary.txt'.format(name, '-optlm' if optlm else ''), summary.summary)
        if latex:
            tk.register_output('{}{}-tuning_summary.latex.txt'.format(name, '-optlm' if optlm else ''), summary.latex)


        # if epoch_decoupling:
        #     tupled_data = [(data[p][id][se].report_dir, str(e), p) 
        #                    for p in ps
        #                    for se, e in zip(sorted(data[p][id].keys())[::-1], epochs[::-1])]
        # else:
        #     tupled_data = [(data[p][id][e].report_dir, str(e), p) for e in epochs for p in ps]
        #     if baseline:
        #         # print("baselines: {}".format(self.baselines))
        #         tupled_data += [
        #             (self._get_scorer(name, epoch, optlm=optlm).report_dir, str(epoch), name)
        #             for name in self.baselines for epoch in epochs
        #         ]


        # row_names = list(ps)
        # if baseline: row_names += self.baselines
        # sum = summary.ScliteSummary(
        #     data=tupled_data,
        #     header="{} \\ {}".format(parameter_names, "epochs"),
        #     col_names=list(map(str,     epochs)),
        #     row_names=row_names
        #     )
        
        # # register summary as output
        # tk.register_output('{}{}-tuning_summary.txt'.format(name, '-optlm' if optlm else ''), sum.summary)

    def parameter_summary(self, name: str, epoch: Union[dict, int], parameters: Iterable, 
            parameter_map=None, parameter_names=None, optlm=False):
        from operator import itemgetter
        data = self.summary_data[name] if not optlm else self.optlm_summary_data[name]
        if parameter_map is None:           parameter_map = TuningSystem.get_parameter_string
        if isinstance(parameter_map, dict): parameter_map = parameter_map.get
        if isinstance(epoch, int): epoch = {p: epoch for p in parameters}
        tupled_data = [(data[parameter_map(p)][0][epoch[p]], str(p[0]), str(p[1]))
            for p in parameters]
        if not parameter_names:
            parameter_names = name
        sum = summary.ScliteSummary(
            data=tupled_data,
            header="{} \\ {}".format(*parameter_names),
            col_names=list(map(str, sorted(set(map(itemgetter(1), parameters))))),
            row_names=list(map(str, sorted(set(map(itemgetter(0), parameters)))))
            )
        # register summary as output
        tk.register_output('{}-{}-tuning_summary.txt'.format(name, epoch), sum.summary)
    
    def alignment_summary(self, name, epochs, parameters, score=False, tdp=None):
        parameters = list(map(self._get_parameter_string, parameters))
        alignment_bundle = {
            epoch: {
                ps: self.alignments[name][ps][epoch][0] for ps in parameters
            } for epoch in epochs
        }
        m = MultipleEpochAlignmentStatisticsJob(alignment_bundle)
        tk.register_output('stats_alignment_{}.txt'.format(name), m.statistics)

        if score:
            score_bundle = {
                ps: {
                    epoch: AlignmentScore(
                            alignment_logs=self.alignments[name][ps][epoch][1].log_file, 
                            concurrent=self.system.csp['train'].concurrent
                        ).total_score
                    for epoch in epochs
                } for ps in parameters
            }
            score_sum = SummaryJob(score_bundle, '{} \\ epochs'.format(name), variable_key='frame')
            tk.register_output('alignment_scores_{}.txt'.format(name), score_sum.summary)
        
        if tdp:
            tdp_epoch = tdp if isinstance(tdp, int) and not isinstance(tdp, bool) else epochs[-1]
            tdp_bundle = {
                ps: TdpFromAlignment(
                    csp=self.system.csp['train'],
                    alignment=self.alignments[name][ps][tdp_epoch][1],
                    allophones=tk.Path(self.system.default_allophones_file)
                ).transition_prob
                for ps in parameters
            }
            tdp_sum = TpdSummary(tdp_bundle)
            tk.register_output('tdp_alignment_{}.txt'.format(name), tdp_sum.summary)

    

class BWConfig:

    SCALE_PATHS = {
        'am_scale'    : ('network', 'combine_prior', 'eval_locals',    'am_scale'), 
        'prior_scale' : ('network', 'combine_prior', 'eval_locals', 'prior_scale'),
        'tdp_scale'   : ('network', 'fast_bw', 'tdp_scale')
    }

    def __init__(self, crnn_config, training_args=None, am_config=None, override=True):
        self.crnn_config   = copy.deepcopy(crnn_config) if override else crnn_config
        self.training_args = training_args
        self.am_config     =     am_config
        self.k             =           1.0 

    @staticmethod
    def deep_get(dictionary, keys, default=None):
        return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys, dictionary)

    @staticmethod
    def deep_set(dictionary, keys, value):
        d = dictionary
        for key in keys[:-1]:
            d = d[key]
        d[keys[-1]] = value
    
    def deploy(self):
        for scale, path in BWConfig.SCALE_PATHS.items():
            value = self.k * BWConfig.deep_get(self.crnn_config, path)
            BWConfig.deep_set(self.crnn_config, path, value)
        return self.crnn_config

    @property
    def am_scale(self):
        return self.crnn_config['network']['combine_prior']['eval_locals']['am_scale']
    
    @am_scale.setter
    def am_scale(self, value):
        self.crnn_config['network']['combine_prior']['eval_locals']['am_scale'] = value

    @property
    def prior_scale(self):
        return self.crnn_config['network']['combine_prior']['eval_locals']['prior_scale']
    
    @prior_scale.setter
    def prior_scale(self, value):
        self.crnn_config['network']['combine_prior']['eval_locals']['prior_scale'] = value

    @property
    def tdp_scale(self):
        return self.crnn_config['network']['fast_bw']['tdp_scale']
    
    @tdp_scale.setter
    def tdp_scale(self, value):
        self.crnn_config['network']['fast_bw']['tdp_scale'] = value
    
    @property
    def absolute_scale(self):
        return self.k
    
    @absolute_scale.setter
    def absolute_scale(self, value):
        scales = [self.prior_scale, self.am_scale, self.tdp_scale]
        for scale in scales:
            scale = value * scale
    
    def add_am_schedule(self, initial_am, final_epoch):
        self.crnn_config['pretrain'] = 'default'
        self.crnn_config['pretrain_repetitions'] = {'default': 0, 'final': final_epoch}
        generate_exp_am_schedule(initial_am=initial_am, final_am=self.am_scale)


class ScaleSetter:
    """ Class for setting various scales and generating setter methods
    to use with TuningSystem. For now focusses on generating an exponential
    pretrain schedule interpolating between the given initial and final
    acoustic model scale. """
    def __init__(self, initial_am, final_am, final_epoch,
        prior_scale=1.0, absolute_scale=1.0):
        self.initial_am = initial_am
        self.final_am = final_am
        self.final_epoch = final_epoch
        self.prior_scale = prior_scale
        self.absolute_scale = absolute_scale
        self.pretrain = True
    
    def generate_exp_am_schedule(self, precision=6):
        l = np.log(self.final_am / self.initial_am) / (self.final_epoch - 1)
        f = lambda x: round(self.initial_am * np.exp(l * x), precision)
        return list(map(f, range(self.final_epoch)))

    @staticmethod
    def scaled_am_pretrain_code(am_scales, prior_scale=1.0, k_scale=1.0):
        pretrain_code = """from Pretrain import WrapEpochValue
network['combine_prior']['eval_locals']['am_scale']    = WrapEpochValue(lambda epoch: {k:.2f} * {ams}[epoch - 1])
network['combine_prior']['eval_locals']['prior_scale'] = WrapEpochValue(lambda epoch: {k:.2f} * {prs:.2f} * {ams}[epoch - 1])
locals().update(**config)""".format(ams=am_scales, prs=prior_scale, k=k_scale) 
        return pretrain_code

    def set_config(self, training_args, config):
        if self.pretrain:
            am_scales = self.generate_exp_am_schedule()
            training_args['extra_python'] = scaled_am_pretrain_code(am_scales, prior_scale=self.prior_scale, k_scale=self.absolute_scale)
            config['pretrain_repetitions']['final'] = self.final_epoch
        set_prior_scale(config, self.final_am * self.prior_scale * self.absolute_scale)
        set_am_scale(config, self.final_am * self.absolute_scale)
        config["network"]["fast_bw"]["tdp_scale"] *= self.absolute_scale
        # set_tdp_scale(config, tdp_scale * self.)

    def get_final_epoch_setter(self):
        def set_schedule(training_args, crnn_config, final_epoch):
            am_scales = generate_exp_am_schedule(
                initial_am=self.initial_am,
                final_am=self.final_am,
                final_epoch=final_epoch    
            )
            training_args['extra_python'] = scaled_am_pretrain_code(
                am_scales, prior_scale=self.prior_scale,
                k_scale=self.absolute_scale)
            set_config(
                crnn_config, initial_am=self.initial_am, final_am=self.final_am,
                final_epoch=final_epoch, prior=self.prior_scale
            )
        return set_schedule

    def train_prior_setter(self):
        if self.pretrain:
            def setter(targs, conf, prior):
                am_scales = self.generate_exp_am_schedule()
                targs['extra_python'] = scaled_am_pretrain_code(am_scales, prior_scale=prior, k_scale=self.absolute_scale)
                set_prior_scale(conf, self.final_am * prior)
            return setter
        else:
            return set_prior_scale


class PretrainSetter(ScaleSetter):
    construction_algo_lstm_skip1 = """
def construction_algo(idx, net_dict):
    if idx == 2:
        return None
    a = 2 * (idx + 1)
    for i in range(a + 1, 7):
        del net_dict['fwd_{}'.format(i)], net_dict['bwd_{}'.format(i)]
    net_dict['output']['from'] = ['fwd_{}'.format(a), 'bwd_{}'.format(a)]
    return net_dict
pretrain = {'construction_algo': construction_algo}
locals().update(**config)
"""
    construction_algo_lstm_skip2 = """
def construction_algo(idx, net_dict):
    if idx == 1:
        return None
    a = 3 * (idx + 1)
    for i in range(a + 1, 7):
        del net_dict['fwd_{}'.format(i)], net_dict['bwd_{}'.format(i)]
    net_dict['output']['from'] = ['fwd_{}'.format(a), 'bwd_{}'.format(a)]
    return net_dict
pretrain = {'construction_algo': construction_algo}
locals().update(**config)
"""

    construction_algo_ffnn_skip1 = """
def construction_algo(idx, net_dict):
    if idx == 2:
        return None
    a = 2 * (idx + 1)
    for i in range(a + 1, 7):
        del net_dict['hidden{}'.format(i)]
    net_dict['output']['from'] = ['hidden{}'.format(a)]
    return net_dict
pretrain = {'construction_algo': construction_algo}
locals().update(**config)
"""

    construction_algo_ffnn_skip2 = """
def construction_algo(idx, net_dict):
    if idx == 1:
        return None
    a = 3 * (idx + 1)
    for i in range(a + 1, 7):
       del net_dict['hidden{}'.format(i)]
    net_dict['output']['from'] = ['hidden{}'.format(a)]
    return net_dict
pretrain = {'construction_algo': construction_algo}
locals().update(**config)
"""
    algo_mapping = {
        "ffnn": { "skip1": construction_algo_ffnn_skip1, "skip2": construction_algo_ffnn_skip2 },
        "lstm": { "skip1": construction_algo_lstm_skip1, "skip2": construction_algo_lstm_skip2 }
    }

    def __init__(self, *args, nn_type="ffnn", **kwargs):
        super().__init__(*args, **kwargs)
        assert nn_type in PretrainSetter.algo_mapping
        self.nn_type = nn_type

    @staticmethod
    def from_scale_setter(scale_setter):
        return PretrainSetter(
            initial_am=scale_setter.initial_am,
            final_am=scale_setter.final_am,
            final_epoch=scale_setter.final_epoch,
            prior_scale=scale_setter.prior_scale,
            absolute_scale=scale_setter.absolute_scale
        )

    def get_pretrain_setter(self):
        # TODO: make depth variable
        def configure_pretrain(targs, config, parameter):
            config['pretrain']['construction_algo'] = parameter['construction_algo']
            if parameter['construction_algo'] == 'from_output':
                del config['pretrain']['output_layers']
            if parameter.get('construction_algo', 'None').startswith('skip'):
                targs['extra_python'] = \
                    PretrainSetter.algo_mapping[self.nn_type][parameter['construction_algo']]
            elif not parameter['warmup'] and 'extra_python' in targs:
                del targs['extra_python']
            config['pretrain_repetitions'] = parameter['repetitions']
            am_scales = self.generate_exp_am_schedule()
            if parameter['warmup']: # and parameter['repetitions']['final'] == 2:
                last = am_scales[-1]
                am_scales += [last] * 20
            if parameter['warmup']:
                am_pretrain_code = scaled_am_pretrain_code(
                    am_scales=am_scales,
                    prior_scale=self.prior_scale,
                    k_scale=self.absolute_scale
                )
                targs['extra_python'] = am_pretrain_code
            if not parameter['construction_algo']:
                del config['pretrain'], config['pretrain_repetitions']
    
        return configure_pretrain

# setters
def set_am_scale(crnn_config, am):
    crnn_config['network']['combine_prior']['eval_locals']['am_scale'] = am

def set_prior_scale(crnn_config, prior):
    crnn_config['network']['combine_prior']['eval_locals']['prior_scale'] = prior

def set_tdp_scale(crnn_config, tdp):
    crnn_config['network']['fast_bw']['tdp_scale'] = tdp

def generate_exp_am_schedule(initial_am=0.001, final_am=1.0, 
                             final_epoch=10, precision=6):
    l = np.log(final_am / initial_am) / (final_epoch - 1)
    f = lambda x: round(initial_am * np.exp(l * x), precision)
    return list(map(f, range(final_epoch)))

def am_pretrain_code(am_scales, prior_scale=1.0):
    pretrain_code = """from Pretrain import WrapEpochValue
network['combine_prior']['eval_locals']['am_scale']    = WrapEpochValue(lambda epoch: {ams}[epoch - 1])
network['combine_prior']['eval_locals']['prior_scale'] = WrapEpochValue(lambda epoch: {prs:.2f} * {ams}[epoch - 1])
locals().update(**config)""".format(ams=am_scales, prs=prior_scale) 
    return pretrain_code

def scaled_am_pretrain_code(am_scales, prior_scale=1.0, k_scale=1.0):
    pretrain_code = """from Pretrain import WrapEpochValue
network['combine_prior']['eval_locals']['am_scale']    = WrapEpochValue(lambda epoch: {k:.2f} * {ams}[epoch - 1])
network['combine_prior']['eval_locals']['prior_scale'] = WrapEpochValue(lambda epoch: {k:.2f} * {prs:.2f} * {ams}[epoch - 1])
locals().update(**config)""".format(ams=am_scales, prs=prior_scale, k=k_scale) 
    return pretrain_code

def set_config(crnn_config, initial_am=0.001, final_am=1.0,
               final_epoch=10, prior=1.0):
    crnn_config['pretrain_repetitions']['final'] = final_epoch
    crnn_config['network']['combine_prior']['eval_locals']['am_scale'] = final_am
    crnn_config['network']['combine_prior']['eval_locals']['prior_scale'] = final_am

def am_pretrain_schedule_setter(*keywords, **kwargs):
    def set_schedule(training_args, crnn_config, parameter):
        if not isinstance(parameter, tuple):
            assert len(keywords) == 1
            parameter = (parameter,)
        kwargs.update(zip(keywords, parameter))
        am_scales = generate_exp_am_schedule(**kwargs)
        training_args['extra_python'] = am_pretrain_code(am_scales)
        set_config(crnn_config, **kwargs)
    return set_schedule

def prior_pretrain_schedule_setter(training_args, crnn_config, prior):
    am_scales = generate_exp_am_schedule()
    training_args['extra_python'] = am_pretrain_code(am_scales, prior)
    crnn_config['network']['combine_prior']['eval_locals']['prior_scale'] = prior

def passthrough_config(crnn_config, config_as_parameter):
    assert crnn_config == {}, "Make sure base config is empty to avoid ambiguity"
    crnn_config.update(config_as_parameter)

def get_continued_training(
        system: meta.System, 
        training_args: dict, 
        base_config: dict, 
        lr_continuation: Union[float, int, bool],
        base_training: Tuple[str, int], 
        copy_mode: str = 'preload', 
        pretrain: bool = False,
        alignment=None, 
        dryrun: bool = False
        ) -> (dict, dict):
    """
    Outputs training args and crnn config dict for the continuation of a base
    training.
    Returns
    -------
    dict
        new training args
    dict
        new crnn config dict 
    """
    assert copy_mode in ['import', 'preload', None]
    training_args = copy.deepcopy(training_args)
    base_config = copy.deepcopy(base_config)
    if not pretrain: # delete all pretrain stuff
        _ = base_config.pop('pretrain', None), \
            base_config.pop('pretrain_repetitions', None), \
            training_args.pop('extra_python', None)
    if alignment:
        training_args['alignment'] = alignment
    if not base_training:
        return training_args, base_config
    if len(base_training) == 3:
        nn_name, nn_epoch, _ = base_training
    else:
        nn_name, nn_epoch = base_training
    if isinstance(lr_continuation, (int, float)) and not isinstance(lr_continuation, bool):
        base_config.pop('learning_rates', None)
        base_config['learning_rate'] = lr_continuation
    elif lr_continuation: # continue from last learning rate
        lrs = LearningRates(
            system.jobs['train']['train_nn_{}'.format(nn_name)].learning_rates)
        tk.register_output('lrs_{}.txt'.format(nn_name), lrs.learning_rates)
        if not dryrun:
            lr = lrs.learning_rates[nn_epoch-1]
            base_config['learning_rate'] = lr
            base_config.pop('learning_rates', None)
    checkpoint = system.nn_checkpoints['train'][nn_name][nn_epoch]
    if len(base_training) == 3:
        checkpoint = base_training[2]
    if copy_mode == 'import': # copy params 1 to 1
        base_config['import_model_train_epoch1'] = checkpoint
    elif copy_mode == 'preload': # copy only same params
        base_config['preload_from_files'] = {
            nn_name : {
                'filename': checkpoint,
                'init_for_train': True,
                'ignore_missing': True
            }
        }
    return training_args, base_config

def clog(fprob, precision=3):
    forward = - round(np.log(fprob), precision)
    loop = - round(np.log(1 - fprob), precision)
    return loop, forward

def set_system_tdps(system, tdpf):
    # set tdps in system
    system.set_tdps(
        corpus        =                            'base',
        tdp_transition=clog(tdpf[0]) + ('infinity',  0.0),
        tdp_silence   =clog(tdpf[1]) + ('infinity', tdpf[2] if len(tdpf) > 2 else 20.0)
    )

def set_recognition_tdps(system, tdpf):
    # set tdps in system
    system.set_tdps(
        corpus        =                             'dev',
        tdp_transition=clog(tdpf[0]) + ('infinity',  0.0),
        tdp_silence   =clog(tdpf[1]) + ('infinity', 20.0)
    )

def set_recognition_prior(scorer_args, prior):
    scorer_args['prior_scale'] = prior

def set_recognition_tdp_scale(system, tdp_scale):
    system.csp["dev"].acoustic_model_config.tdp.scale = tdp_scale

def set_tdps(system, training_args, crnn_config, tdpf):
    # set tdps in system
    sil_exit = 20.0 if len(tdpf) < 3 else tdpf[2]
    speech_skip = 'infinity' if len(tdpf) < 4 else tdpf[3]
    system.set_tdps(
        corpus        =                            'base',
        tdp_transition=clog(tdpf[0]) + (speech_skip,  0.0),
        tdp_silence   =clog(tdpf[1]) + ('infinity', sil_exit)
    )

    # add bw configs
    additional_sprint_config_files, \
    additional_sprint_post_config_files \
        = add_fastbw_configs(system.csp['train'])
    
    # update training arguments
    training_args.update(
        additional_sprint_config_files     =     additional_sprint_config_files,
        additional_sprint_post_config_files=additional_sprint_post_config_files
    )

def set_fastbw_tdps(training_args, tdpf):
    for config in ["additional_sprint_config_files"]: #, "additional_sprint_post_config_files"]:
        ac_conf = training_args[config]['fastbw']['neural-network-trainer.alignment-fsa-exporter.model-combination.acoustic-model']
        for i, phon in enumerate(['*', 'silence']):
            ac_conf.tdp[phon].loop, ac_conf.tdp[phon].forward = clog(tdpf[i]) 
        if len(tdpf) > 2:
            ac_conf.tdp.silence.exit = tdpf[2]