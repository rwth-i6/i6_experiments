__all__ = ["BaseDecoder"]

import copy
from typing import Dict, List, Optional, Tuple, Type, Union

from sisyphus import tk
from sisyphus.delayed_ops import Delayed, DelayedBase, DelayedFormat

import i6_core.features as features
import i6_core.rasr as rasr
import i6_core.recognition as recog

from i6_core.corpus import CorpusToStmJob, SegmentCorpusJob
from i6_core.meta import CorpusObject

from .config.am_config import Tdp
from .util.decode import (
    RecognitionParameters,
    SearchJobArgs,
    Lattice2CtmArgs,
    StmArgs,
    ScliteScorerArgs,
    OptimizeJobArgs,
)


class BaseDecoder:
    """
    This class provides basic functionality to perform decoding with RASR

    - initialize or set CommonRasrParameters (crp)
    - set the eval corpora
    - perform decoding
    - lm scale optimization and redo decoding with optimized lm scale
    """

    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: "str" = "linux-x86_64-standard",
        compress: bool = False,
        append: bool = False,
        unbuffered: bool = False,
        compress_after_run: bool = True,
        search_job_class: Type[tk.Job] = recog.AdvancedTreeSearchJob,
        scorer_job_class: Type[tk.Job] = recog.ScliteJob,
        alias_output_prefix: str = "",
    ):
        """
        :param rasr_binary_path: path to RASR binary directory
        :param rasr_arch: RASR arch
        :param compress: RASR config setting to compress log files
        :param append: RASR config setting to append to log files
        :param unbuffered: RASR config setting to utilize buffer for writing to log files
        :param compress_after_run: RASR config setting to compress files after the actual run
        :param search_job_class: search job class similar to recog.AdvancedTreeSearchJob
        :param scorer_job_class: scoring job class similar to recog.ScliteJob
        :param alias_output_prefix: set a prefix for alias and output paths
        """
        self.rasr_binary_path = rasr_binary_path
        self.rasr_arch = rasr_arch

        self.search_job_class = search_job_class
        self.scorer_job_class = scorer_job_class

        self.base_crp_name = "base"
        self.crp = {"base": rasr.CommonRasrParameters()}
        rasr.crp_add_default_output(self.crp["base"], compress, append, unbuffered, compress_after_run)
        self.crp["base"].set_executables(rasr_binary_path, rasr_arch)

        self.eval_corpora = []
        self.stm_paths = {}
        self.feature_flows = {}

        self.feature_name_job_mapping = {
            "gt": features.GammatoneJob,
            "mfcc": features.MfccJob,
        }

        # holds the recognition jobs: search, lat2ctm, score, optlm
        # self.jobs[CORPUS_KEY][JOB_TYPE][EXP_NAME] = tk.Job
        self.jobs: Dict[str, Dict[str, Dict[str, Type[tk.Job]]]] = {}

        self.alias_output_prefix = alias_output_prefix

    def init_base_crp(
        self,
        *,
        acoustic_model_config: rasr.RasrConfig,
        lexicon_config: rasr.RasrConfig,
        language_model_config: Optional[rasr.RasrConfig] = None,
        lm_lookahead_config: Optional[rasr.RasrConfig] = None,
        extra_configs: Optional[Dict[str, rasr.RasrConfig]] = None,
        crp_name: Optional[str] = None,
    ):
        """
        creates a (base) crp from RASR configs

        :param acoustic_model_config: acoustic model config as RasrConfig
        :param lexicon_config: lexicon config as RasrConfig
        :param language_model_config: language model config as RasrConfig
        :param lm_lookahead_config: language model lookahead config as RasrConfig
        :param extra_configs: additional configs
        :param crp_name: the name for the created crp. default is "base"
        """
        if crp_name is not None:
            self.crp[crp_name] = copy.deepcopy(self.crp["base"])
            self.base_crp_name = crp_name

        self.crp[self.base_crp_name].acoustic_model_config = acoustic_model_config
        self.crp[self.base_crp_name].lexicon_config = lexicon_config
        self.crp[self.base_crp_name].language_model_config = language_model_config
        self.crp[self.base_crp_name].lm_lookahead_config = lm_lookahead_config

        extra_configs = extra_configs if extra_configs is not None else {}
        for k, v in extra_configs.items():
            self.crp[self.base_crp_name][k] = v

    def set_crp(self, name: str, crp: rasr.CommonRasrParameters):
        """
        sets an already existing crp

        :param name: name of the crp
        :param crp: common RASR parameters
        """
        self.base_crp_name = name
        self.crp[name] = copy.deepcopy(crp)

    def init_eval_datasets(
        self,
        eval_datasets: Dict[str, CorpusObject],
        corpus_durations: Dict[str, float],
        concurrency: Dict[str, int],
        *,
        feature_flows: Optional[Dict[str, rasr.FlowNetwork]] = None,
        feature_extraction: Optional[Tuple[str, Dict]] = None,
        stm_args: Optional[Dict[str, StmArgs]] = None,
        stm_paths: Optional[Dict[str, tk.Path]] = None,
    ):
        """
        initializes the evaluation datasets

        :param eval_datasets: the actual datasets
        :param corpus_durations: corpus durations
        :param concurrency: split decoding in N parts and run in parallel
        :param feature_flows: RASR flow network
        :param feature_extraction: extract new features with the given parameters
        :param stm_args: arguments for the STM creation
        :param stm_paths: use already created STM files
        """
        assert (feature_flows is not None) ^ (feature_extraction is not None)
        assert (stm_args is not None) ^ (stm_paths is not None)

        self.eval_corpora.extend(list(eval_datasets.keys()))
        for corpus_key, corpus_object in eval_datasets.items():
            self.crp[corpus_key] = rasr.CommonRasrParameters(base=self.crp[self.base_crp_name])
            rasr.crp_set_corpus(self.crp[corpus_key], corpus_object)
            self.crp[corpus_key].corpus_duration = corpus_durations[corpus_key]
            self.crp[corpus_key].concurrent = concurrency[corpus_key]
            self.crp[corpus_key].segment_path = SegmentCorpusJob(
                self.crp[corpus_key].corpus_config.file, concurrency[corpus_key]
            ).out_segment_path

            if stm_paths is not None:
                self.stm_paths[corpus_key] = stm_paths[corpus_key]
            else:
                self.stm_paths[corpus_key] = CorpusToStmJob(
                    self.crp[corpus_key].corpus_config.file, **stm_args[corpus_key]
                ).out_stm_path

            if feature_flows is not None:
                self.feature_flows[corpus_key] = feature_flows[corpus_key]
            else:
                name = feature_extraction[0]
                feature_job = self.feature_name_job_mapping[name](self.crp[corpus_key], **feature_extraction[1])
                feature_path = rasr.FlagDependentFlowAttribute(
                    "cache_mode",
                    {
                        "task_dependent": feature_job.out_feature_path[name],
                        "bundle": feature_job.out_feature_bundle[name],
                    },
                )
                self.feature_flows[corpus_key] = features.basic_cache_flow(feature_path)

            for job_type in ["search", "lat2ctm", "score", "optlm"]:
                self.jobs[corpus_key][job_type] = {}

    @staticmethod
    def _get_scales_string(
        am_scale: Union[float, tk.Variable],
        lm_scale: Union[float, tk.Variable],
        prior_scale: float,
        tdp_scale: float,
        tdp_speech: Tdp,
        tdp_silence: Tdp,
        tdp_nonspeech: Optional[Tdp] = None,
        pronunciation_scale: Optional[Union[float, tk.Variable]] = None,
        altas: Optional[float] = None,
    ) -> Union[str, DelayedBase]:
        """
        gets a string which describes the individual recognition and its parameters. used for alias and output.
        """
        out_str = ""

        if isinstance(am_scale, tk.Variable):
            out_str += DelayedFormat("am{}", am_scale)
        else:
            out_str += f"am{am_scale:05.2f}"

        if isinstance(lm_scale, tk.Variable):
            out_str += DelayedFormat("_lm{}", lm_scale)
        else:
            out_str += f"_lm{lm_scale:05.2f}"

        out_str += f"_prior{prior_scale:05.2f}"

        out_str += f"_tdp{tdp_scale}"
        out_str += f"_tdpspeech{tdp_speech}"
        out_str += f"_tdpsilence{tdp_silence}"

        if tdp_nonspeech is not None:
            out_str += f"_tdpnonspeech{tdp_nonspeech}"

        if pronunciation_scale is not None:
            if isinstance(pronunciation_scale, tk.Variable):
                out_str += DelayedFormat("_ps{}", pronunciation_scale)
            else:
                out_str += f"_ps{pronunciation_scale:05.2f}"

        if altas is not None:
            out_str += f"_altas{altas:05.2f}"

        return out_str

    def _set_scales_and_tdps(
        self,
        corpus_key: str,
        am_scale: Union[float, tk.Variable],
        lm_scale: Union[float, tk.Variable],
        prior_scale: float,
        tdp_scale: float,
        tdp_speech: Tdp,
        tdp_silence: Tdp,
        tdp_nonspeech: Optional[Tdp],
        pronunciation_scale: Optional[float],
        altas: Optional[float],
    ) -> Union[str, DelayedBase]:
        """
        sets scales and TDPs for each corpus
        """
        base_corpus_key = corpus_key
        corpus_key += "/"
        corpus_key += self._get_scales_string(
            am_scale=am_scale,
            lm_scale=lm_scale,
            prior_scale=prior_scale,
            tdp_scale=tdp_scale,
            tdp_speech=tdp_speech,
            tdp_silence=tdp_silence,
            tdp_nonspeech=tdp_nonspeech,
            pronunciation_scale=pronunciation_scale,
            altas=altas,
        )

        self.crp[corpus_key] = rasr.CommonRasrParameters(base=self.crp[base_corpus_key])

        if isinstance(lm_scale, tk.Variable):
            self.crp[corpus_key].language_model_config.scale = Delayed(lm_scale)
        else:
            self.crp[corpus_key].language_model_config.scale = lm_scale

        self.crp[corpus_key].acoustic_model_config.tdp.scale = tdp_scale

        self.crp[corpus_key].acoustic_model_config.tdp["*"].loop = tdp_speech.loop
        self.crp[corpus_key].acoustic_model_config.tdp["*"].forward = tdp_speech.forward
        self.crp[corpus_key].acoustic_model_config.tdp["*"].skip = tdp_speech.skip
        self.crp[corpus_key].acoustic_model_config.tdp["*"].exit = tdp_speech.exit

        self.crp[corpus_key].acoustic_model_config.tdp.silence.loop = tdp_silence.loop
        self.crp[corpus_key].acoustic_model_config.tdp.silence.forward = tdp_silence.forward
        self.crp[corpus_key].acoustic_model_config.tdp.silence.skip = tdp_silence.skip
        self.crp[corpus_key].acoustic_model_config.tdp.silence.exit = tdp_silence.exit

        if (
            self.crp[corpus_key].acoustic_model_config.tdp.tying_type == "global-and-nonword"
            and tdp_nonspeech is not None
        ):
            for nw in [0, 1]:
                k = "nonword-%d" % nw
                self.crp[corpus_key].acoustic_model_config.tdp[k].loop = tdp_nonspeech.loop
                self.crp[corpus_key].acoustic_model_config.tdp[k].forward = tdp_nonspeech.forward
                self.crp[corpus_key].acoustic_model_config.tdp[k].skip = tdp_nonspeech.skip
                self.crp[corpus_key].acoustic_model_config.tdp[k].exit = tdp_nonspeech.exit

        return corpus_key

    def _recog(
        self,
        name: str,
        corpus_key: str,
        feature_scorer: rasr.FeatureScorer,
        feature_flow: rasr.FlowNetwork,
        search_job_args: Union[SearchJobArgs, Dict],
        lat_2_ctm_args: Union[Lattice2CtmArgs, Dict],
        scorer_args: Union[ScliteScorerArgs, Dict],
        scorer_hyp_param_name: str,
    ) -> (tk.Job, recog.LatticeToCtmJob, tk.Job):
        """
        the recognition job pipeline: search, lattice to ctm, score
        """
        search_job = self.search_job_class(
            crp=self.crp[corpus_key],
            feature_scorer=feature_scorer,
            feature_flow=feature_flow,
            **search_job_args,
        )
        search_job.add_alias(f"{self.alias_output_prefix}recog_{corpus_key}/{name}")

        lat_2_ctm_job = recog.LatticeToCtmJob(
            crp=self.crp[corpus_key],
            lattice_cache=search_job.out_lattice_bundle,
            **lat_2_ctm_args,
        )
        lat_2_ctm_job.add_alias(f"{self.alias_output_prefix}lattice2ctm_{corpus_key}/{name}")

        scorer_job = self.scorer_job_class(**{scorer_hyp_param_name: lat_2_ctm_job.out_ctm_file}, **scorer_args)
        scorer_job.add_alias(f"{self.alias_output_prefix}scoring_{corpus_key}/{name}")
        tk.register_output(
            f"{self.alias_output_prefix}recog_{corpus_key}/{name}.reports",
            scorer_job.out_report_dir,
        )

        self.jobs[corpus_key]["search"][name] = search_job
        self.jobs[corpus_key]["lat2ctm"][name] = lat_2_ctm_job
        self.jobs[corpus_key]["score"][name] = scorer_job

        return search_job, lat_2_ctm_job, scorer_job

    def _optimize_scales(
        self,
        name: str,
        corpus_key: str,
        lattice_cache: tk.Path,
        initial_pron_scale: float,
        initial_lm_scale: float,
        scorer_args: Union[ScliteScorerArgs, Dict],
        scorer_hyp_param_name: str,
        optimize_args: OptimizeJobArgs,
    ) -> (tk.Variable, tk.Variable):
        """
        optimizes the scales: pronunciation and lm
        """
        opt_job = recog.OptimizeAMandLMScaleJob(
            crp=self.crp[corpus_key],
            lattice_cache=lattice_cache,
            initial_am_scale=initial_pron_scale,  # wrong parameter name in job definition
            initial_lm_scale=initial_lm_scale,
            scorer_cls=self.scorer_job_class,
            scorer_kwargs=scorer_args,
            scorer_hyp_param_name=scorer_hyp_param_name,
            **optimize_args,
        )
        opt_job.add_alias(f"{self.alias_output_prefix}optimize_{corpus_key}/{name}")

        self.jobs[corpus_key]["optlm"][name] = opt_job

        best_pron_scale = opt_job.out_best_am_score
        best_lm_scale = opt_job.out_best_lm_score

        return best_pron_scale, best_lm_scale

    def decode(
        self,
        name: str,
        corpus_key: str,
        *,
        feature_scorer: Optional[rasr.FeatureScorer],
        feature_flow: rasr.FlowNetwork,
        recognition_parameters: List[RecognitionParameters],
        lm_rasr_config: rasr.RasrConfig,
        search_job_args: Union[SearchJobArgs, Dict],
        lat_2_ctm_args: Union[Lattice2CtmArgs, Dict],
        scorer_args: Union[ScliteScorerArgs, Dict],
        optimize_parameters: Union[OptimizeJobArgs, Dict],
        scorer_hyp_param_name: str = "hyp",
        optimize_pron_lm_scales: bool = False,
    ):
        """
        run decoding

        :param name: decoding experiment name
        :param corpus_key: which eval corpus to use
        :param feature_scorer: specific RASR feature scorer
        :param feature_flow: flow network
        :param recognition_parameters: recognition parameters (scales, tdps, ...)
        :param lm_rasr_config: language model RASR config
        :param search_job_args: specific arguments for the search job
        :param lat_2_ctm_args: specific arguments for the lattice to ctm job
        :param scorer_args: specific arguments for the scoring job
        :param optimize_parameters: specific arguments for the optimize lm scale job
        :param scorer_hyp_param_name: key name for the hypothesis file of the scorer
        :param optimize_pron_lm_scales: run the pronunciation and lm scale optimization step. DO NOT RUN THIS STEP ON TEST SET!!!
        """
        for recog_par in recognition_parameters:
            for (
                am_sc,
                lm_sc,
                prior_sc,
                pron_sc,
                tdp_sc,
                tdp_speech,
                tdp_silence,
                tdp_nonspeech,
                altas,
            ) in recog_par:
                scorer_args["ref"] = self.stm_paths[corpus_key]
                self.crp[corpus_key].language_model_config = lm_rasr_config

                derived_corpus_key = self._set_scales_and_tdps(
                    corpus_key=corpus_key,
                    am_scale=am_sc,
                    lm_scale=lm_sc,
                    prior_scale=prior_sc,
                    tdp_scale=tdp_sc,
                    tdp_speech=tdp_speech,
                    tdp_silence=tdp_silence,
                    tdp_nonspeech=tdp_nonspeech,
                    pronunciation_scale=pron_sc,
                    altas=altas,
                )

                feature_scorer.config.scale = am_sc
                feature_scorer.config.priori_scale = prior_sc

                if altas is not None:
                    adv_search_extra_config = rasr.RasrConfig()
                    adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                        altas
                    )
                    search_job_args["extra_config"] = adv_search_extra_config

                if pron_sc is not None:
                    model_combination_config = rasr.RasrConfig()
                    model_combination_config.pronunciation_scale = pron_sc
                    search_job_args["model_combination_config"] = model_combination_config

                search_job, lat_2_ctm_job, scorer_job = self._recog(
                    name=name,
                    corpus_key=derived_corpus_key,
                    feature_scorer=feature_scorer,
                    feature_flow=feature_flow,
                    search_job_args=search_job_args,
                    lat_2_ctm_args=lat_2_ctm_args,
                    scorer_args=scorer_args,
                    scorer_hyp_param_name=scorer_hyp_param_name,
                )
                if optimize_pron_lm_scales:
                    best_pron_scale, best_lm_scale = self._optimize_scales(
                        name=name,
                        corpus_key=derived_corpus_key,
                        lattice_cache=search_job.out_lattice_bundle,
                        initial_pron_scale=pron_sc,
                        initial_lm_scale=lm_sc,
                        scorer_args=scorer_args,
                        scorer_hyp_param_name=scorer_hyp_param_name,
                        optimize_args=optimize_parameters,
                    )

                    optimized_corpus_key = self._set_scales_and_tdps(
                        corpus_key=corpus_key,
                        am_scale=am_sc,
                        lm_scale=best_lm_scale,
                        prior_scale=prior_sc,
                        tdp_scale=tdp_sc,
                        tdp_speech=tdp_speech,
                        tdp_silence=tdp_silence,
                        tdp_nonspeech=tdp_nonspeech,
                        pronunciation_scale=best_pron_scale,
                        altas=altas,
                    )

                    opt_search_job, opt_lat_2_ctm_job, opt_scorer_job = self._recog(
                        name=f"{name}-opt",
                        corpus_key=optimized_corpus_key,
                        feature_scorer=feature_scorer,
                        feature_flow=feature_flow,
                        search_job_args=search_job_args,
                        lat_2_ctm_args=lat_2_ctm_args,
                        scorer_args=scorer_args,
                        scorer_hyp_param_name=scorer_hyp_param_name,
                    )

                self.crp[corpus_key].language_model = None

    def get_results(self):
        pass
