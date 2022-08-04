__all__ = ["BaseDecoder"]

import copy
import itertools
import logging
from typing import Dict, List, Optional, Type, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.recognition as recog

from i6_core.corpus import CorpusToStmJob
from i6_core.meta import CorpusObject

from .config.am_config import Tdp
from .util.decode import (
    BaseRecognitionParameters,
    SearchJobArgs,
    Lattice2CtmArgs,
    ScliteScorerArgs,
    OptimizeJobArgs,
)


class BaseDecoder:
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
        self.rasr_binary_path = rasr_binary_path
        self.rasr_arch = rasr_arch

        self.search_job_class = search_job_class
        self.scorer_job_class = scorer_job_class

        self.crp = {"base": rasr.CommonRasrParameters()}
        rasr.crp_add_default_output(
            self.crp["base"], compress, append, unbuffered, compress_after_run
        )
        self.crp["base"].set_executables(rasr_binary_path, rasr_arch)

        self.eval_corpora = []

        self.alias_output_prefix = alias_output_prefix

    def init_decoder(
        self,
        *,
        acoustic_model_config: rasr.RasrConfig,
        lexicon_config: rasr.RasrConfig,
        language_model_config: rasr.RasrConfig,
        lm_lookahead_config: rasr.RasrConfig,
        acoustic_model_post_config: Optional[rasr.RasrConfig] = None,
        lexicon_post_config: Optional[rasr.RasrConfig] = None,
        language_model_post_config: Optional[rasr.RasrConfig] = None,
        extra_configs: Optional[Dict[str, rasr.RasrConfig]] = None,
        extra_post_configs: Optional[Dict[str, rasr.RasrConfig]] = None,
    ):
        self.crp["base"].acoustic_model_config = acoustic_model_config
        self.crp["base"].lexicon_config = lexicon_config
        self.crp["base"].language_model_config = language_model_config
        self.crp["base"].lm_lookahead_config = lm_lookahead_config

        if acoustic_model_post_config is not None:
            self.crp["base"].acoustic_model_post_config = acoustic_model_post_config
        if lexicon_post_config is not None:
            self.crp["base"].lexicon_post_config_post_config = lexicon_post_config
        if language_model_post_config is not None:
            self.crp["base"].language_model_post_config = language_model_post_config

        for k, v in extra_configs.items():
            self.crp["base"][k] = v
        for k, v in extra_post_configs.items():
            self.crp["base"][k] = v

    def set_crp(self, name: str, crp: rasr.CommonRasrParameters):
        self.crp[name] = copy.deepcopy(crp)

    def init_eval_datasets(self, eval_datasets: Dict[str, CorpusObject]):
        self.eval_corpora.extend([eval_datasets.keys()])
        for corpus_key, corpus_object in eval_datasets.items():
            self.crp[corpus_key] = rasr.CommonRasrParameters(base=self.crp["base"])
            rasr.crp_set_corpus(self.crp[corpus_key], corpus_object)

    @staticmethod
    def _get_scales_string(
        am_scale: float,
        lm_scale: float,
        prior_scale: float,
        tdp_scale: float,
        tdp_speech: Optional[Tdp],
        tdp_silence: Optional[Tdp],
        tdp_nonspeech: Optional[Tdp] = None,
        pronunciation_scale: Optional[float] = None,
    ):
        out_str = ""
        out_str += f"am{am_scale:2f.2f}"
        out_str += f"_lm{lm_scale:2f.2f}"
        out_str += f"_prior{prior_scale:2f.2f}"

        out_str += f"_tdp{tdp_scale}"
        out_str += f"_tdpspeech{tdp_speech}"
        out_str += f"_tdpsilence{tdp_silence}"

        if tdp_nonspeech is not None:
            out_str += f"_tdpnonspeech{tdp_nonspeech}"

        if pronunciation_scale is not None:
            out_str += f"_ps{pronunciation_scale:2f.2f}"

        return out_str

    def _set_scales_and_tdps(
        self,
        corpus_key: str,
        am_scale: float,
        lm_scale: float,
        prior_scale: float,
        tdp_scale: float,
        tdp_speech: Tdp,
        tdp_silence: Tdp,
        tdp_nonspeech: Optional[Tdp],
        pronunciation_scale: Optional[float],
    ) -> str:
        base_corpus_key = corpus_key
        corpus_key += "_"
        corpus_key += self._get_scales_string(
            am_scale=am_scale,
            lm_scale=lm_scale,
            prior_scale=prior_scale,
            tdp_scale=tdp_scale,
            tdp_speech=tdp_speech,
            tdp_silence=tdp_silence,
            tdp_nonspeech=tdp_nonspeech,
            pronunciation_scale=pronunciation_scale,
        )

        self.crp[corpus_key] = rasr.CommonRasrParameters(base=self.crp[base_corpus_key])

        self.crp[corpus_key].acoustic_model_config.scale = am_scale
        self.crp[corpus_key].language_model_config.scale = lm_scale

        self.crp[corpus_key].mixture_set.prior_scale = prior_scale

        self.crp[corpus_key].acoustic_model_config.tdp.scale = tdp_scale

        self.crp[corpus_key].acoustic_model_config.tdp["*"].loop = tdp_speech.loop
        self.crp[corpus_key].acoustic_model_config.tdp["*"].forward = tdp_speech.forward
        self.crp[corpus_key].acoustic_model_config.tdp["*"].skip = tdp_speech.skip
        self.crp[corpus_key].acoustic_model_config.tdp["*"].exit = tdp_speech.exit

        self.crp[corpus_key].acoustic_model_config.tdp.silence.loop = tdp_silence.loop
        self.crp[
            corpus_key
        ].acoustic_model_config.tdp.silence.forward = tdp_silence.forward
        self.crp[corpus_key].acoustic_model_config.tdp.silence.skip = tdp_silence.skip
        self.crp[corpus_key].acoustic_model_config.tdp.silence.exit = tdp_silence.exit

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
        search_job = self.search_job_class(
            crp=self.crp[corpus_key],
            feature_scorer=feature_scorer,
            feature_flow=feature_flow,
            **search_job_args,
        )
        search_job.set_vis_name(
            f"Recog: {self.alias_output_prefix}{name}. Corpus: {corpus_key}"
        )
        search_job.add_alias(f"{self.alias_output_prefix}recog_{corpus_key}/{name}")

        lat_2_ctm_job = recog.LatticeToCtmJob(
            crp=self.crp[corpus_key],
            lattice_cache=search_job.out_lattice_bundle,
            parallelize=False,
            **lat_2_ctm_args,
        )
        lat_2_ctm_job.add_alias(
            f"{self.alias_output_prefix}lattice2ctm_{corpus_key}/{name}"
        )

        scorer_job = self.scorer_job_class(
            **{scorer_hyp_param_name: lat_2_ctm_job.out_ctm_file}, **scorer_args
        )
        scorer_job.add_alias(f"{self.alias_output_prefix}scoring_{corpus_key}/{name}")
        tk.register_output(
            f"{self.alias_output_prefix}recog_{corpus_key}/{name}.reports",
            scorer_job.out_report_dir,
        )

        return search_job, lat_2_ctm_job, scorer_job

    def _optimize_scales(
        self,
        name: str,
        corpus_key: str,
        lattice_cache,
        initial_am_scale,
        initial_lm_scale,
        scorer_kwargs,
        scorer_hyp_param_name: str,
        optimize_args: OptimizeJobArgs,
    ) -> (float, float):
        opt_job = recog.OptimizeAMandLMScaleJob(
            crp=self.crp[corpus_key],
            lattice_cache=lattice_cache,
            initial_am_scale=initial_am_scale,
            initial_lm_scale=initial_lm_scale,
            scorer_cls=self.scorer_job_class,
            scorer_kwargs=scorer_kwargs,
            scorer_hyp_param_name=scorer_hyp_param_name,
            **optimize_args,
        )
        opt_job.add_alias(f"{self.alias_output_prefix}optimize_{corpus_key}/{name}")

        best_am_scale = opt_job.out_best_am_score
        best_lm_scale = opt_job.out_best_lm_score

        return best_am_scale, best_lm_scale

    def decode(
        self,
        name: str,
        feature_scorers: List[Optional[rasr.FeatureScorer]],
        feature_flow: rasr.FlowNetwork,
        recognition_parameters: BaseRecognitionParameters,
        *,
        search_job_args: Union[SearchJobArgs, Dict],
        lat_2_ctm_args: Union[Lattice2CtmArgs, Dict],
        scorer_args: Union[ScliteScorerArgs, Dict],
        optimize_parameters: Union[OptimizeJobArgs, Dict],
        scorer_hyp_param_name: str = "hyp",
        optimize_am_lm_scales: bool = False,
    ):
        for (
            corpus_key,
            scorer,
            am_sc,
            lm_sc,
            prior_sc,
            pron_sc,
            tdp_sc,
            tdp_speech,
            tdp_silence,
            tdp_nonspeech,
        ) in itertools.product(
            self.eval_corpora, feature_scorers, recognition_parameters
        ):
            if scorer_args["ref"] is None:
                logging.warning("Using no cleanup during STM creation.")
                scorer_args["ref"] = CorpusToStmJob(
                    self.crp[corpus_key].corpus_config.file,
                    exclude_non_speech=False,
                    remove_punctuation=False,
                    fix_whitespace=False,
                ).out_stm_path

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
            )

            if pron_sc is not None:
                model_combination_config = rasr.RasrConfig()
                model_combination_config.pronunciation_scale = pron_sc
                search_job_args["model_combination_config"] = model_combination_config

            search_job, lat_2_ctm_job, scorer_job = self._recog(
                name=name,
                corpus_key=derived_corpus_key,
                feature_scorer=scorer,
                feature_flow=feature_flow,
                search_job_args=search_job_args,
                lat_2_ctm_args=lat_2_ctm_args,
                scorer_args=scorer_args,
                scorer_hyp_param_name=scorer_hyp_param_name,
            )
            if optimize_am_lm_scales:
                best_am_scale, best_lm_scale = self._optimize_scales(
                    name=name,
                    corpus_key=derived_corpus_key,
                    lattice_cache=search_job.out_lattice_bundle,
                    initial_am_scale=am_sc,
                    initial_lm_scale=lm_sc,
                    scorer_kwargs={},
                    scorer_hyp_param_name=scorer_hyp_param_name,
                    optimize_args=optimize_parameters,
                )

                optimized_corpus_key = self._set_scales_and_tdps(
                    corpus_key=corpus_key,
                    am_scale=best_am_scale,
                    lm_scale=best_lm_scale,
                    prior_scale=prior_sc,
                    tdp_scale=tdp_sc,
                    tdp_speech=tdp_speech,
                    tdp_silence=tdp_silence,
                    tdp_nonspeech=tdp_nonspeech,
                    pronunciation_scale=pron_sc,
                )

                self._recog(
                    name=f"{name}-opt",
                    corpus_key=optimized_corpus_key,
                    feature_scorer=scorer,
                    feature_flow=feature_flow,
                    search_job_args=search_job_args,
                    lat_2_ctm_args=lat_2_ctm_args,
                    scorer_args=scorer_args,
                    scorer_hyp_param_name=scorer_hyp_param_name,
                )

    def get_results(self):
        pass
