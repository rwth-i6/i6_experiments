__all__ = ["TORCHFactoredHybridDecoder", "TORCHFactoredHybridAligner"]

import copy
import dataclasses
import itertools
import numpy as np
import re

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union
from IPython import embed

from sisyphus import tk
from sisyphus.delayed_ops import Delayed, DelayedBase, DelayedFormat

from i6_experiments.users.raissi.setups.common.decoder import RasrFeatureScorer
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import check_prior_info, \
    get_factored_feature_scorer, RasrShortestPathAlgorithm, DecodingJobs, round2

Path = tk.Path

import i6_core.am as am
import i6_core.features as features
import i6_core.lm as lm
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog


from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorConfig,
    PriorInfo,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
)

from i6_experiments.users.raissi.setups.common.decoder.statistics import ExtractSearchStatisticsJob
from i6_experiments.users.raissi.setups.common.util.tdp import format_tdp_val, format_tdp
from i6_experiments.users.raissi.setups.common.util.argmin import ComputeArgminJob
from i6_experiments.users.raissi.setups.common.data.typings import (
    TDP,
)




#remember you might need to scale to 2-15 the input of the feature flow

def get_onnx_feature_scorer(
    context_type: PhoneticContext,
    feature_scorer_type: RasrFeatureScorer,
    mixtures: tk.Path,
    prior_info: Union[PriorInfo, tk.Variable, DelayedBase],
    posterior_scale: float = 1.0,
):

    assert context_type in [PhoneticContext.monophone, PhoneticContext.joint_diphone]

    if isinstance(prior_info, PriorInfo):
        check_prior_info(context_type=context_type, prior_info=prior_info)

    if context_type.is_joint_diphone():
        prior = prior_info.diphone_prior
    elif context_type.is_monophone():
        prior = prior_info.center_state_prior

    return feature_scorer_type.get_fs_class()(
        scale=posterior_scale,
        prior_mixtures=mixtures,
        priori_scale=prior.scale,
        prior_file=prior.file,
    )


class TORCHFactoredHybridDecoder:
    def __init__(
        self,
        name: str,
        crp: rasr.CommonRasrParameters,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        feature_path: Path,
        model_path: Path,
        graph: Path,
        mixtures: Path,
        eval_args,
        scorer: Optional[Union[recog.ScliteJob, recog.Hub5ScoreJob, recog.Hub5ScoreJob]] = None,
        is_multi_encoder_output: bool = False,
        silence_id: int = 40,
        set_batch_major_for_feature_scorer: bool = True,
        lm_gc_simple_hash=False,
        gpu=False,
    ):

        self.name = name
        self.crp = copy.deepcopy(crp)
        self.context_type = context_type
        self.feature_scorer_type = feature_scorer_type
        self.feature_path = feature_path
        self.model_path = model_path
        self.graph = graph
        self.mixtures = mixtures
        self.is_multi_encoder_output = is_multi_encoder_output
        self.silence_id = silence_id
        self.set_batch_major_for_feature_scorer = set_batch_major_for_feature_scorer
        self.lm_gc_simple_hash = lm_gc_simple_hash

        #ToDo introduce the ONNX mapping
        self.tensor_map = ()

        self.eval_args = eval_args  # ctm file as ref

        self.gpu = gpu
        self.scorer = scorer if scorer is not None else recog.ScliteJob


    def get_base_sample_feature_flow(
        self, audio_format: str, dc_detection: bool = False, **kwargs
    ):
        args = {
            "audio_format": audio_format,
            "dc_detection": dc_detection,
            "input_options": {"block-size": 1},
            "scale_input": 2**-15,
        }
        args.update(kwargs)
        return features.samples_flow(**args)



    def recognize_count_lm(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        calculate_stats=False,
        is_min_duration=False,
        opt_lm_am=True,
        only_lm_opt=True,
        cn_decoding: bool = False,
        keep_value=12,
        use_estimated_tdps=False,
        add_sis_alias_and_output=True,
        rerun_after_opt_lm=False,
        name_override: Union[str, None] = None,
        name_prefix: str = "",
        gpu: Optional[bool] = None,
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        pre_path: str = "decoding",
        rtf_cpu: float = 16,
        rtf_gpu: float = 4,
        lm_config: rasr.RasrConfig = None,
        create_lattice: bool = True,
        separate_lm_image_gc_generation: bool = False,
        search_rqmt_update=None,
        adv_search_extra_config: Optional[rasr.RasrConfig] = None,
        adv_search_extra_post_config: Optional[rasr.RasrConfig] = None,
        cpu_omp_thread=2,
    ) -> DecodingJobs:
        return self.recognize(
            label_info=label_info,
            num_encoder_output=num_encoder_output,
            search_parameters=search_parameters,
            calculate_stats=calculate_stats,
            gpu=gpu,
            cpu_rqmt=cpu_rqmt,
            mem_rqmt=mem_rqmt,
            is_min_duration=is_min_duration,
            opt_lm_am=opt_lm_am,
            only_lm_opt=only_lm_opt,
            cn_decoding=cn_decoding,
            keep_value=keep_value,
            use_estimated_tdps=use_estimated_tdps,
            add_sis_alias_and_output=add_sis_alias_and_output,
            rerun_after_opt_lm=rerun_after_opt_lm,
            name_override=name_override,
            name_prefix=name_prefix,
            is_nn_lm=False,
            lm_config=lm_config,
            pre_path=pre_path,
            crp_update=crp_update,
            rtf_cpu=rtf_cpu,
            rtf_gpu=rtf_gpu,
            create_lattice=create_lattice,
            search_rqmt_update=search_rqmt_update,
            adv_search_extra_config=adv_search_extra_config,
            adv_search_extra_post_config=adv_search_extra_post_config,
            cpu_omp_thread=cpu_omp_thread,
            separate_lm_image_gc_generation=separate_lm_image_gc_generation,
        )

    def recognize(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        calculate_stats=False,
        lm_config: Optional[rasr.RasrConfig] = None,
        is_nn_lm: bool = False,
        only_lm_opt: bool = True,
        opt_lm_am: bool = True,
        cn_decoding: bool = False,
        pre_path: Optional[str] = None,
        rerun_after_opt_lm=False,
        name_override: Union[str, None] = None,
        name_override_without_name: Union[str, None] = None,
        add_sis_alias_and_output=True,
        name_prefix: str = "",
        is_min_duration=False,
        use_estimated_tdps=False,
        keep_value=12,
        gpu: Optional[bool] = None,
        mem_rqmt: Optional[int] = None,
        cpu_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        rtf_cpu: Optional[float] = None,
        rtf_gpu: Optional[float] = None,
        create_lattice: bool = True,
        adv_search_extra_config: Optional[rasr.RasrConfig] = None,
        adv_search_extra_post_config: Optional[rasr.RasrConfig] = None,
        search_rqmt_update=None,
        cpu_omp_thread=2,
        separate_lm_image_gc_generation: bool = False,
    ) -> DecodingJobs:
        if isinstance(search_parameters, SearchParameters):
            assert len(search_parameters.tdp_speech) == 4
            assert len(search_parameters.tdp_silence) == 4
            assert not search_parameters.silence_penalties or len(search_parameters.silence_penalties) == 2
            assert not search_parameters.transition_scales or len(search_parameters.transition_scales) == 2

        name = f"{name_prefix}{self.name}/"
        if name_override_without_name is not None:
            name = name_override_without_name
        elif name_override is not None:
            name += f"{name_override}"
        else:
            if search_parameters.beam_limit < 500_000:
                name += f"Beamlim{search_parameters.beam_limit}"
            name += f"Beam{search_parameters.beam}-Lm{search_parameters.lm_scale}"

        search_crp = copy.deepcopy(self.crp)

        if search_crp.lexicon_config.normalize_pronunciation and search_parameters.pron_scale is not None:
            model_combination_config = rasr.RasrConfig()
            model_combination_config.pronunciation_scale = search_parameters.pron_scale
            pron_scale = search_parameters.pron_scale
            if name_override is None:
                name += f"-Pron{pron_scale}"
        else:
            model_combination_config = None
            pron_scale = 1.0

        if name_override is None:
            if search_parameters.prior_info.left_context_prior is not None:
                name += f"-prL{search_parameters.prior_info.left_context_prior.scale}"
            if search_parameters.prior_info.center_state_prior is not None:
                name += f"-prC{search_parameters.prior_info.center_state_prior.scale}"
            if search_parameters.prior_info.right_context_prior is not None:
                name += f"-prR{search_parameters.prior_info.right_context_prior.scale}"
            if search_parameters.prior_info.diphone_prior is not None:
                name += f"-prJ-C{search_parameters.prior_info.diphone_prior.scale}"
            if search_parameters.we_pruning > 0.5:
                name += f"-wep{search_parameters.we_pruning}"
            if search_parameters.we_pruning_limit < 6000 or search_parameters.we_pruning_limit > 10000:
                name += f"-wepLim{search_parameters.we_pruning_limit}"
            if search_parameters.altas is not None:
                name += f"-ALTAS{search_parameters.altas}"
            if search_parameters.add_all_allophones:
                name += "-allAllos"
            if not create_lattice:
                name += "-noLattice"

        if search_parameters.tdp_scale is not None:
            if name_override is None:
                name += f"-tdpScale-{search_parameters.tdp_scale}"
                name += f"-silTdp-{format_tdp(search_parameters.tdp_silence)}"
                if search_parameters.tdp_nonword is not None:
                    name += f"-nwTdp-{format_tdp(search_parameters.tdp_nonword)}"
                name += f"-spTdp-{format_tdp(search_parameters.tdp_speech)}"

            if self.feature_scorer_type.is_factored():
                if search_parameters.transition_scales is not None:
                    loop_scale, forward_scale = search_parameters.transition_scales
                    if name_override is None:
                        name += f"-loopScale-{loop_scale}"
                        name += f"-fwdScale-{forward_scale}"
                else:
                    loop_scale = forward_scale = 1.0

                if search_parameters.silence_penalties is not None:
                    sil_loop_penalty, sil_fwd_penalty = search_parameters.silence_penalties
                    if name_override is None:
                        name += f"-silLoopP-{sil_loop_penalty}"
                        name += f"-silFwdP-{sil_fwd_penalty}"
                else:
                    sil_fwd_penalty = sil_loop_penalty = 0.0
        else:
            if name_override is None:
                name += "-noTdp"

        state_tying = search_crp.acoustic_model_config.state_tying.type
        if state_tying == RasrStateTying.cart:
            cart_file = search_crp.acoustic_model_config.state_tying.file

        tdp_transition = (
            search_parameters.tdp_speech if search_parameters.tdp_scale is not None else (0.0, 0.0, "infinity", 0.0)
        )
        tdp_silence = (
            search_parameters.tdp_silence if search_parameters.tdp_scale is not None else (0.0, 0.0, "infinity", 0.0)
        )
        tdp_nonword = (
            search_parameters.tdp_nonword if search_parameters.tdp_nonword is not None else (0.0, 0.0, "infinity", 0.0)
        )

        search_crp.acoustic_model_config = am.acoustic_model_config(
            state_tying=state_tying,
            states_per_phone=label_info.n_states_per_phone,
            state_repetitions=1,
            across_word_model=True,
            early_recombination=False,
            tdp_scale=search_parameters.tdp_scale,
            tdp_transition=tdp_transition,
            tdp_silence=tdp_silence,
            tdp_nonword=tdp_nonword,
            nonword_phones=search_parameters.non_word_phonemes,
            tying_type="global-and-nonword",
        )

        search_crp.acoustic_model_config.allophones["add-all"] = search_parameters.add_all_allophones
        search_crp.acoustic_model_config.allophones["add-from-lexicon"] = not search_parameters.add_all_allophones

        if state_tying == RasrStateTying.cart:
            search_crp.acoustic_model_config.state_tying.file = cart_file
        else:
            search_crp.acoustic_model_config["state-tying"][
                "use-boundary-classes"
            ] = label_info.phoneme_state_classes.use_boundary()
            search_crp.acoustic_model_config["state-tying"][
                "use-word-end-classes"
            ] = label_info.phoneme_state_classes.use_word_end()

        #toDo add feature scorer
        feature_scorer = None

        if lm_config is not None:
            search_crp.language_model_config = lm_config
        search_crp.language_model_config.scale = search_parameters.lm_scale

        if crp_update is not None:
            crp_update(search_crp)

        rqms = self.get_requirements(beam=search_parameters.beam, nn_lm=is_nn_lm)
        sp = self.get_search_params(
            search_parameters.beam,
            search_parameters.beam_limit,
            search_parameters.we_pruning,
            search_parameters.we_pruning_limit,
            is_count_based=True,
        )

        if search_crp.language_model_config.type == "cheating-segment":
            adv_search_extra_config = rasr.RasrConfig()
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.disable_unigram_lookahead = True
            la_options = self.get_lookahead_options(clow=0, chigh=10)
            name += "-cheating"
        else:
            la_options = self.get_lookahead_options()
            adv_search_extra_config = (
                copy.deepcopy(adv_search_extra_config) if adv_search_extra_config is not None else rasr.RasrConfig()
            )
            # Set ALTAS
            if search_parameters.altas is not None:
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                    search_parameters.altas
                )
            # Limit the history for neural decoding
            if search_parameters.word_recombination_limit is not None:
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.reduce_context_word_recombination = (
                    True
                )
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.reduce_context_word_recombination_limit = (
                    search_parameters.word_recombination_limit
                )
                name += f"recombLim{search_parameters.word_recombination_limit}"

            if search_parameters.lm_lookahead_scale is not None:
                name += f"-lh{search_parameters.lm_lookahead_scale}"
                # Use 4gram for lookahead. The lookahead LM must not be too good.
                # Half the normal LM scale is a good starting value.
                # To validate the assumption the original LM is a 4gram
                assert search_crp.language_model_config.type.lower() == "arpa"

                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.separate_lookahead_lm = True
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.lm_lookahead_scale = (
                    search_parameters.lm_lookahead_scale
                )
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm.file = (
                    search_crp.language_model_config.file
                )
                # TODO(future): Add LM image instead of file here.
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm.scale = 1.0
                adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm.type = "ARPA"

                if search_parameters.lm_lookahead_history_limit > 1:
                    la_options["history_limit"] = search_parameters.lm_lookahead_history_limit
                    name += f"-lhHisLim{search_parameters.lm_lookahead_history_limit}"

        pre_path = (
            pre_path
            if pre_path is not None and len(pre_path) > 0
            else "decoding-gridsearch"
            if search_parameters.altas is not None
            else "decoding"
        )

        search = recog.AdvancedTreeSearchJob(
            crp=search_crp,
            feature_flow=self.feature_scorer_flow,
            feature_scorer=feature_scorer,
            search_parameters=sp,
            lm_lookahead=True,
            lookahead_options=la_options,
            eval_best_in_lattice=True,
            use_gpu=gpu if gpu is not None else self.gpu,
            rtf=rtf_gpu if rtf_gpu is not None and gpu else rtf_cpu if rtf_cpu is not None else rqms["rtf"],
            mem=rqms["mem"] if mem_rqmt is None else mem_rqmt,
            cpu=2 if cpu_rqmt is None else cpu_rqmt,
            lmgc_scorer=rasr.DiagonalMaximumScorer(self.mixtures) if self.lm_gc_simple_hash else None,
            create_lattice=create_lattice,
            # separate_lm_image_gc_generation=separate_lm_image_gc_generation,
            model_combination_config=model_combination_config,
            model_combination_post_config=None,
            extra_config=adv_search_extra_config,
            extra_post_config=adv_search_extra_post_config,
        )
        self.post_recog_name = name

        if search_rqmt_update is not None:
            search.rqmt.update(search_rqmt_update)
            search.cpu = cpu_omp_thread

        if keep_value is not None:
            search.keep_value(keep_value)

        if add_sis_alias_and_output:
            search.add_alias(f"{pre_path}/{name}")

        if calculate_stats:
            stat = ExtractSearchStatisticsJob(list(search.out_log_file.values()), 5.12)

            if add_sis_alias_and_output:
                pre = f"{pre_path}-" if pre_path != "decoding" and pre_path != "decoding-gridsearch" else ""
                stat.add_alias(f"{pre}statistics/{name}")
                tk.register_output(f"{pre}statistics/rtf/{name}.rtf", stat.overall_rtf)
        else:
            stat = None

        if not create_lattice:
            return DecodingJobs(
                lat2ctm=None,
                scorer=None,
                search=search,
                search_crp=search_crp,
                search_feature_scorer=feature_scorer,
                search_stats=stat,
            )

        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"
        lat2ctm = recog.LatticeToCtmJob(
            crp=search_crp,
            lattice_cache=search.out_lattice_bundle,
            parallelize=True,
            best_path_algo=self.shortest_path_algo.value,
            extra_config=lat2ctm_extra_config,
            fill_empty_segments=True,
        )
        ctm = lat2ctm.out_ctm_file

        if cn_decoding:

            cn_config = rasr.RasrConfig()
            cn_config.flf_lattice_tool.lexicon.normalize_pronunciation = True
            assert search_parameters.pron_scale is not None, "set a pronunciation scale for CN decoding"
            cn_config.flf_lattice_tool.network.scale_pronunciation = search_parameters.pron_scale
            cn_config.flf_lattice_tool.network.dump_ctm.ctm.fill_empty_segments = True
            cn_config.flf_lattice_tool.network.dump_ctm.ctm.non_word_symbol = "[empty]"
            decode = recog.CNDecodingJob(
                crp=search_crp,
                lattice_path=search.out_lattice_bundle,
                lm_scale=search_parameters.lm_scale,
                extra_config=cn_config,
            )
            ctm = decode.out_ctm_file
            name += f"-Pron{search_parameters.pron_scale}"

        scorer = self.scorer(hyp=ctm, **self.eval_args)
        add_prepath = (
            f"{self.shortest_path_algo.value}/"
            if self.shortest_path_algo != RasrShortestPathAlgorithm.bellman_ford
            else ""
        )
        if add_sis_alias_and_output:
            tk.register_output(
                f"{pre_path}/{'cn/' if cn_decoding else ''}{add_prepath}{name}.wer", scorer.out_report_dir
            )

        if opt_lm_am and (search_parameters.altas is None or search_parameters.altas < 3.0):
            assert search_parameters.beam >= 15.0
            if pron_scale is not None:
                if isinstance(pron_scale, DelayedBase) and pron_scale.is_set():
                    pron_scale = pron_scale.get()
            s_kwrgs_opt = copy.copy(self.eval_args)
            s_kwrgs_opt["hyp"] = lat2ctm.out_ctm_file

            opt = recog.OptimizeAMandLMScaleJob(
                crp=search_crp,
                lattice_cache=search.out_lattice_bundle,
                initial_am_scale=pron_scale,
                initial_lm_scale=search_parameters.lm_scale,
                scorer_cls=self.scorer,
                scorer_kwargs=s_kwrgs_opt,
                opt_only_lm_scale=only_lm_opt,
            )

            if add_sis_alias_and_output:
                tk.register_output(
                    f"{pre_path}/{add_prepath}{name}/onlyLmOpt{only_lm_opt}.optlm.txt",
                    opt.out_log_file,
                )

            if rerun_after_opt_lm:
                ps = pron_scale if only_lm_opt else opt.out_best_lm_score
                rounded_lm_scale = Delayed(opt.out_best_lm_score).function(round2)
                rounded_pron_scale = Delayed(ps).function(round2)
                params = search_parameters.with_lm_scale(rounded_lm_scale).with_pron_scale(rounded_pron_scale)
                name_after_rerun = name

                if opt.out_best_lm_score.get() is not None:
                    name_after_rerun = re.sub(r"Lm[0-9]*.[0.9*]", f"Lm{rounded_lm_scale}", name)

                name_prefix_len = len(f"{name_prefix}{self.name}/")
                # in order to have access afterwards to the lm scale mainly
                self.tuned_params = params

                return self.recognize(
                    add_sis_alias_and_output=add_sis_alias_and_output,
                    calculate_stats=calculate_stats,
                    is_min_duration=is_min_duration,
                    is_nn_lm=is_nn_lm,
                    keep_value=keep_value,
                    label_info=label_info,
                    lm_config=lm_config,
                    name_override=f"{name_after_rerun[name_prefix_len:]}-optlm",
                    name_prefix=name_prefix,
                    num_encoder_output=num_encoder_output,
                    only_lm_opt=only_lm_opt,
                    opt_lm_am=False,
                    pre_path=pre_path,
                    rerun_after_opt_lm=rerun_after_opt_lm,
                    search_parameters=params,
                    use_estimated_tdps=use_estimated_tdps,
                )

        return DecodingJobs(
            lat2ctm=lat2ctm,
            scorer=scorer,
            search=search,
            search_crp=search_crp,
            search_feature_scorer=feature_scorer,
            search_stats=stat,
        )

    def recognize_optimize_scales(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        prior_scales: Union[
            List[Tuple[float]],  # center
            List[Tuple[float, float]],  # center, left
            List[Tuple[float, float, float]],  # center, left, right
            np.ndarray,
        ],
        tdp_scales: Union[List[float], np.ndarray],
        tdp_sil: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        tdp_speech: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        pron_scales: Union[List[float], np.ndarray] = None,
        altas_value=14.0,
        altas_beam=14.0,
        keep_value=10,
        gpu: Optional[bool] = None,
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        pre_path: str = "scales",
        cpu_slow: bool = True,
    ) -> SearchParameters:
        assert len(prior_scales) > 0
        assert len(tdp_scales) > 0

        recog_args = dataclasses.replace(search_parameters, altas=altas_value, beam=altas_beam)

        if isinstance(prior_scales, np.ndarray):
            prior_scales = [(s,) for s in prior_scales] if prior_scales.ndim == 1 else [tuple(s) for s in prior_scales]

        prior_scales = [tuple(round(p, 2) for p in priors) for priors in prior_scales]
        prior_scales = [
            (p, 0.0, 0.0)
            if isinstance(p, float)
            else (p[0], 0.0, 0.0)
            if len(p) == 1
            else (p[0], p[1], 0.0)
            if len(p) == 2
            else p
            for p in prior_scales
        ]
        tdp_scales = [round(s, 2) for s in tdp_scales]
        tdp_sil = tdp_sil if tdp_sil is not None else [recog_args.tdp_silence]
        tdp_speech = tdp_speech if tdp_speech is not None else [recog_args.tdp_speech]
        use_pron = self.crp.lexicon_config.normalize_pronunciation and pron_scales is not None

        if use_pron:
            jobs = {
                ((c, l, r), tdp, tdp_sl, tdp_sp, pron): self.recognize_count_lm(
                    add_sis_alias_and_output=False,
                    calculate_stats=False,
                    cpu_rqmt=cpu_rqmt,
                    crp_update=crp_update,
                    gpu=gpu,
                    is_min_duration=False,
                    keep_value=keep_value,
                    label_info=label_info,
                    mem_rqmt=mem_rqmt,
                    name_override=f"{self.name}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{tdp_sl}-tdpSp{tdp_sp}-pron{pron}",
                    num_encoder_output=num_encoder_output,
                    opt_lm_am=False,
                    rerun_after_opt_lm=False,
                    search_parameters=dataclasses.replace(
                        recog_args, tdp_scale=tdp, tdp_silence=tdp_sl, tdp_speech=tdp_sp, pron_scale=pron
                    ).with_prior_scale(left=l, center=c, right=r, diphone=c),
                )
                for ((c, l, r), tdp, tdp_sl, tdp_sp, pron) in itertools.product(
                    prior_scales, tdp_scales, tdp_sil, tdp_speech, pron_scales
                )
            }

        else:
            jobs = {
                ((c, l, r), tdp, tdp_sl, tdp_sp): self.recognize_count_lm(
                    add_sis_alias_and_output=False,
                    calculate_stats=False,
                    cpu_rqmt=cpu_rqmt,
                    crp_update=crp_update,
                    gpu=gpu,
                    is_min_duration=False,
                    keep_value=keep_value,
                    label_info=label_info,
                    mem_rqmt=mem_rqmt,
                    name_override=f"{self.name}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{tdp_sl}-tdpSp{tdp_sp}",
                    num_encoder_output=num_encoder_output,
                    opt_lm_am=False,
                    rerun_after_opt_lm=False,
                    search_parameters=dataclasses.replace(
                        recog_args, tdp_scale=tdp, tdp_silence=tdp_sl, tdp_speech=tdp_sp
                    ).with_prior_scale(left=l, center=c, right=r, diphone=c),
                )
                for ((c, l, r), tdp, tdp_sl, tdp_sp) in itertools.product(prior_scales, tdp_scales, tdp_sil, tdp_speech)
            }
        jobs_num_e = {k: v.scorer.out_num_errors for k, v in jobs.items()}

        if use_pron:
            for ((c, l, r), tdp, tdp_sl, tdp_sp, pron), recog_jobs in jobs.items():
                if cpu_slow:
                    recog_jobs.search.update_rqmt("run", {"cpu_slow": True})

                pre_name = f"{pre_path}/{self.name}/Lm{recog_args.lm_scale}-Pron{pron}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{format_tdp(tdp_sl)}-tdpSp{format_tdp(tdp_sp)}"

                recog_jobs.lat2ctm.set_keep_value(keep_value)
                recog_jobs.search.set_keep_value(keep_value)

                recog_jobs.search.add_alias(pre_name)
                tk.register_output(f"{pre_name}.wer", recog_jobs.scorer.out_report_dir)
        else:
            for ((c, l, r), tdp, tdp_sl, tdp_sp), recog_jobs in jobs.items():
                if cpu_slow:
                    recog_jobs.search.update_rqmt("run", {"cpu_slow": True})

                pre_name = f"{pre_path}/{self.name}/Lm{recog_args.lm_scale}-Pron{recog_args.pron_scale}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{format_tdp(tdp_sl)}-tdpSp{format_tdp(tdp_sp)}"

                recog_jobs.lat2ctm.set_keep_value(keep_value)
                recog_jobs.search.set_keep_value(keep_value)

                recog_jobs.search.add_alias(pre_name)
                tk.register_output(f"{pre_name}.wer", recog_jobs.scorer.out_report_dir)

        best_overall_wer = ComputeArgminJob({k: v.scorer.out_wer for k, v in jobs.items()})
        best_overall_n = ComputeArgminJob(jobs_num_e)
        tk.register_output(
            f"decoding/scales-best/{self.name}/args",
            best_overall_n.out_argmin,
        )
        tk.register_output(
            f"decoding/scales-best/{self.name}/wer",
            best_overall_wer.out_min,
        )

        def push_delayed_tuple(
            argmin: DelayedBase,
        ) -> Tuple[DelayedBase, DelayedBase, DelayedBase, DelayedBase]:
            return tuple(argmin[i] for i in range(4))

        # cannot destructure, need to use indices
        best_priors = best_overall_n.out_argmin[0]
        best_tdp_scale = best_overall_n.out_argmin[1]
        best_tdp_sil = best_overall_n.out_argmin[2]
        best_tdp_sp = best_overall_n.out_argmin[3]
        if use_pron:
            best_pron = best_overall_n.out_argmin[4]

            base_cfg = dataclasses.replace(
                search_parameters,
                tdp_scale=best_tdp_scale,
                tdp_silence=push_delayed_tuple(best_tdp_sil),
                tdp_speech=push_delayed_tuple(best_tdp_sp),
                pron_scale=best_pron,
            )
        else:
            base_cfg = dataclasses.replace(
                search_parameters,
                tdp_scale=best_tdp_scale,
                tdp_silence=push_delayed_tuple(best_tdp_sil),
                tdp_speech=push_delayed_tuple(best_tdp_sp),
            )

        best_center_prior = best_priors[0]
        if self.context_type.is_monophone():
            return base_cfg.with_prior_scale(center=best_center_prior)
        if self.context_type.is_joint_diphone():
            return base_cfg.with_prior_scale(diphone=best_center_prior)

        best_left_prior = best_priors[1]
        if self.context_type.is_diphone():
            return base_cfg.with_prior_scale(center=best_center_prior, left=best_left_prior)

        best_right_prior = best_priors[2]
        return base_cfg.with_prior_scale(
            center=best_center_prior,
            left=best_left_prior,
            right=best_right_prior,
        )

    def recognize_optimize_scales_v2(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        prior_scales: Union[
            List[Tuple[float]],  # center
            List[Tuple[float, float]],  # center, left
            List[Tuple[float, float, float]],  # center, left, right
            np.ndarray,
        ],
        tdp_scales: Union[List[float], np.ndarray],
        tdp_sil: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        tdp_nonword: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        tdp_speech: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        pron_scales: Union[List[float], np.ndarray] = None,
        altas_value=14.0,
        altas_beam=14.0,
        keep_value=10,
        gpu: Optional[bool] = None,
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        pre_path: str = "scales",
        cpu_slow: bool = True,
    ) -> SearchParameters:
        assert len(prior_scales) > 0
        assert len(tdp_scales) > 0

        recog_args = dataclasses.replace(search_parameters, altas=altas_value, beam=altas_beam)

        if isinstance(prior_scales, np.ndarray):
            prior_scales = [(s,) for s in prior_scales] if prior_scales.ndim == 1 else [tuple(s) for s in prior_scales]

        prior_scales = [tuple(round(p, 2) for p in priors) for priors in prior_scales]
        prior_scales = [
            (p, 0.0, 0.0)
            if isinstance(p, float)
            else (p[0], 0.0, 0.0)
            if len(p) == 1
            else (p[0], p[1], 0.0)
            if len(p) == 2
            else p
            for p in prior_scales
        ]
        tdp_scales = [round(s, 2) for s in tdp_scales]
        tdp_sil = tdp_sil if tdp_sil is not None else [recog_args.tdp_silence]
        tdp_nonword = tdp_nonword if tdp_nonword is not None else [recog_args.tdp_nonword]
        tdp_speech = tdp_speech if tdp_speech is not None else [recog_args.tdp_speech]

        use_pron = self.crp.lexicon_config.normalize_pronunciation and pron_scales is not None

        if use_pron:
            jobs = {
                ((c, l, r), tdp, tdp_sl, tdp_nw, tdp_sp, pron): self.recognize_count_lm(
                    add_sis_alias_and_output=False,
                    calculate_stats=False,
                    cpu_rqmt=cpu_rqmt,
                    crp_update=crp_update,
                    gpu=gpu,
                    is_min_duration=False,
                    keep_value=keep_value,
                    label_info=label_info,
                    mem_rqmt=mem_rqmt,
                    name_override=f"{self.name}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{tdp_sl}-tdpNnw{tdp_nw}tdpSp{tdp_sp}-tdpSp{tdp_sp}-pron{pron}",
                    num_encoder_output=num_encoder_output,
                    opt_lm_am=False,
                    rerun_after_opt_lm=False,
                    search_parameters=dataclasses.replace(
                        recog_args,
                        tdp_scale=tdp,
                        tdp_silence=tdp_sl,
                        tdp_nonword=tdp_nw,
                        tdp_speech=tdp_sp,
                        pron_scale=pron,
                    ).with_prior_scale(left=l, center=c, right=r, diphone=c),
                )
                for ((c, l, r), tdp, tdp_sl, tdp_nw, tdp_sp, pron) in itertools.product(
                    prior_scales, tdp_scales, tdp_sil, tdp_nonword, tdp_speech, pron_scales
                )
            }
        else:
            jobs = {
                ((c, l, r), tdp, tdp_sl, tdp_nw, tdp_sp): self.recognize_count_lm(
                    add_sis_alias_and_output=False,
                    calculate_stats=False,
                    cpu_rqmt=cpu_rqmt,
                    crp_update=crp_update,
                    gpu=gpu,
                    is_min_duration=False,
                    keep_value=keep_value,
                    label_info=label_info,
                    mem_rqmt=mem_rqmt,
                    name_override=f"{self.name}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{tdp_sl}-tdpNnw{tdp_nw}-tdpSp{tdp_sp}-",
                    num_encoder_output=num_encoder_output,
                    opt_lm_am=False,
                    rerun_after_opt_lm=False,
                    search_parameters=dataclasses.replace(
                        recog_args, tdp_scale=tdp, tdp_silence=tdp_sl, tdp_nonword=tdp_nw, tdp_speech=tdp_sp
                    ).with_prior_scale(left=l, center=c, right=r, diphone=c),
                )
                for ((c, l, r), tdp, tdp_sl, tdp_nw, tdp_sp) in itertools.product(
                    prior_scales, tdp_scales, tdp_sil, tdp_nonword, tdp_speech
                )
            }
        jobs_num_e = {k: v.scorer.out_num_errors for k, v in jobs.items()}

        if use_pron:
            for ((c, l, r), tdp, tdp_sl, tdp_nw, tdp_sp, pron), recog_jobs in jobs.items():
                if cpu_slow:
                    recog_jobs.search.update_rqmt("run", {"cpu_slow": True})

                pre_name = (
                    f"{pre_path}/{self.name}/Lm{recog_args.lm_scale}-Pron{pron}-pC{c}-pL{l}-pR{r}-tdp{tdp}-"
                    f"tdpSil{format_tdp(tdp_sl)}-tdpNw{format_tdp(tdp_nw)}-tdpSp{format_tdp(tdp_sp)}"
                )

                recog_jobs.lat2ctm.set_keep_value(keep_value)
                recog_jobs.search.set_keep_value(keep_value)

                recog_jobs.search.add_alias(pre_name)
                tk.register_output(f"{pre_name}.wer", recog_jobs.scorer.out_report_dir)
        else:
            for ((c, l, r), tdp, tdp_sl, tdp_nw, tdp_sp), recog_jobs in jobs.items():
                if cpu_slow:
                    recog_jobs.search.update_rqmt("run", {"cpu_slow": True})

                pre_name = (
                    f"{pre_path}/{self.name}/Lm{recog_args.lm_scale}-Pron{recog_args.pron_scale}"
                    f"-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{format_tdp(tdp_sl)}-tdpNw{format_tdp(tdp_nw)}-tdpSp{format_tdp(tdp_sp)}"
                )

                recog_jobs.lat2ctm.set_keep_value(keep_value)
                recog_jobs.search.set_keep_value(keep_value)

                recog_jobs.search.add_alias(pre_name)
                tk.register_output(f"{pre_name}.wer", recog_jobs.scorer.out_report_dir)

        best_overall_wer = ComputeArgminJob({k: v.scorer.out_wer for k, v in jobs.items()})
        best_overall_n = ComputeArgminJob(jobs_num_e)
        tk.register_output(
            f"decoding/scales-best/{self.name}/args",
            best_overall_n.out_argmin,
        )
        tk.register_output(
            f"decoding/scales-best/{self.name}/wer",
            best_overall_wer.out_min,
        )

        def push_delayed_tuple(
            argmin: DelayedBase,
        ) -> Tuple[DelayedBase, DelayedBase, DelayedBase, DelayedBase]:
            return tuple(argmin[i] for i in range(4))

        # cannot destructure, need to use indices
        best_priors = best_overall_n.out_argmin[0]
        best_tdp_scale = best_overall_n.out_argmin[1]
        best_tdp_sil = best_overall_n.out_argmin[2]
        best_tdp_nw = best_overall_n.out_argmin[3]
        best_tdp_sp = best_overall_n.out_argmin[4]
        if use_pron:
            best_pron = best_overall_n.out_argmin[5]

            base_cfg = dataclasses.replace(
                search_parameters,
                tdp_scale=best_tdp_scale,
                tdp_silence=push_delayed_tuple(best_tdp_sil),
                tdp_nonword=push_delayed_tuple(best_tdp_nw),
                tdp_speech=push_delayed_tuple(best_tdp_sp),
                pron_scale=best_pron,
            )
        else:
            base_cfg = dataclasses.replace(
                search_parameters,
                tdp_scale=best_tdp_scale,
                tdp_silence=push_delayed_tuple(best_tdp_sil),
                tdp_nonword=push_delayed_tuple(best_tdp_nw),
                tdp_speech=push_delayed_tuple(best_tdp_sp),
            )

        best_center_prior = best_priors[0]
        if self.context_type.is_monophone():
            return base_cfg.with_prior_scale(center=best_center_prior)
        if self.context_type.is_joint_diphone():
            return base_cfg.with_prior_scale(diphone=best_center_prior)

        best_left_prior = best_priors[1]
        if self.context_type.is_diphone():
            return base_cfg.with_prior_scale(center=best_center_prior, left=best_left_prior)

        best_right_prior = best_priors[2]
        return base_cfg.with_prior_scale(
            center=best_center_prior,
            left=best_left_prior,
            right=best_right_prior,
        )

    def recognize_optimize_transtition_values(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        tdp_sil: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        tdp_speech: Optional[List[Tuple[TDP, TDP, TDP, TDP]]] = None,
        altas_value=14.0,
        altas_beam=14.0,
        keep_value=10,
        gpu: Optional[bool] = None,
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        pre_path: str = "transition-values",
        cpu_slow: bool = True,
    ) -> SearchParameters:

        recog_args = dataclasses.replace(search_parameters, altas=altas_value, beam=altas_beam)

        tdp_sil = tdp_sil if tdp_sil is not None else [recog_args.tdp_silence]
        tdp_speech = tdp_speech if tdp_speech is not None else [recog_args.tdp_speech]
        jobs = {
            (tdp_sl, tdp_sp): self.recognize_count_lm(
                add_sis_alias_and_output=False,
                calculate_stats=False,
                cpu_rqmt=cpu_rqmt,
                crp_update=crp_update,
                gpu=gpu,
                is_min_duration=False,
                keep_value=keep_value,
                label_info=label_info,
                mem_rqmt=mem_rqmt,
                name_override=f"{self.name}-tdpSil{tdp_sl}-tdpSp{tdp_sp}-",
                num_encoder_output=num_encoder_output,
                opt_lm_am=False,
                rerun_after_opt_lm=False,
                search_parameters=dataclasses.replace(recog_args, tdp_silence=tdp_sl, tdp_speech=tdp_sp),
            )
            for (tdp_sl, tdp_sp) in itertools.product(tdp_sil, tdp_speech)
        }
        jobs_num_e = {k: v.scorer.out_num_errors for k, v in jobs.items()}

        for (tdp_sl, tdp_sp), recog_jobs in jobs.items():
            if cpu_slow:
                recog_jobs.search.update_rqmt("run", {"cpu_slow": True})

            pre_name = f"{pre_path}/{self.name}/" f"tdpSil{format_tdp(tdp_sl)}tdpSp{format_tdp(tdp_sp)}"

            recog_jobs.lat2ctm.set_keep_value(keep_value)
            recog_jobs.search.set_keep_value(keep_value)

            recog_jobs.search.add_alias(pre_name)
            tk.register_output(f"{pre_name}.wer", recog_jobs.scorer.out_report_dir)

        best_overall_wer = ComputeArgminJob({k: v.scorer.out_wer for k, v in jobs.items()})
        best_overall_n = ComputeArgminJob(jobs_num_e)
        tk.register_output(
            f"decoding/tdp-best/{self.name}/args",
            best_overall_n.out_argmin,
        )
        tk.register_output(
            f"decoding/tdp-best/{self.name}/wer",
            best_overall_wer.out_min,
        )

        def push_delayed_tuple(
            argmin: DelayedBase,
        ) -> Tuple[DelayedBase, DelayedBase, DelayedBase, DelayedBase]:
            return tuple(argmin[i] for i in range(4))

        # cannot destructure, need to use indices
        best_tdp_sil = best_overall_n.out_argmin[0]
        best_tdp_sp = best_overall_n.out_argmin[1]

        base_cfg = dataclasses.replace(
            search_parameters,
            tdp_silence=push_delayed_tuple(best_tdp_sil),
            tdp_speech=push_delayed_tuple(best_tdp_sp),
        )

        return base_cfg


class TORCHFactoredHybridAligner(TORCHFactoredHybridDecoder):
    def __init__(
        self,
        name: str,
        crp,
        context_type: PhoneticContext,
        feature_scorer_type: RasrFeatureScorer,
        feature_path: Path,
        model_path: Path,
        graph: Path,
        mixtures: Path,
        tf_library: Optional[Union[str, Path]] = None,
        gpu=False,
        is_multi_encoder_output=False,
        silence_id=40,
        set_batch_major_for_feature_scorer: bool = True,
    ):

        super().__init__(
            name=name,
            crp=crp,
            context_type=context_type,
            feature_scorer_type=feature_scorer_type,
            feature_path=feature_path,
            model_path=model_path,
            graph=graph,
            mixtures=mixtures,
            eval_args=None,
            gpu=gpu,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=silence_id,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
        )

    @staticmethod
    def correct_transition_applicator(crp, allow_for_silence_repetitions=False, correct_fsa_strcuture=False):
        # correct for the FSA bug
        crp.acoustic_model_config.tdp.applicator_type = "corrected"
        # The exit penalty is on the lemma level and should not be applied for alignment
        for tdp_type in ["*", "silence", "nonword-0", "nonword-1"]:
            crp.acoustic_model_config.tdp[tdp_type]["exit"] = 0.0
        if correct_fsa_strcuture:
            crp.acoustic_model_config["*"]["fix-allophone-context-at-word-boundaries"] = True
            crp.acoustic_model_config["*"]["transducer-builder-filter-out-invalid-allophones"] = True
            crp.acoustic_model_config["*"]["allow-for-silence-repetitions"] = allow_for_silence_repetitions

        return crp

    def get_alignment_job(
        self,
        label_info: LabelInfo,
        alignment_parameters: AlignmentParameters,
        num_encoder_output: int,
        pre_path: Optional[str] = "alignments",
        correct_fsa_structure: bool = False,
        allow_for_scaled_tdp: bool = False,
        is_min_duration: bool = False,
        use_estimated_tdps: bool = False,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        mem_rqmt: Optional[int] = 8,
        gpu: Optional[bool] = False,
        cpu: Optional[bool] = 2,
    ) -> mm.AlignmentJob:

        assert (
            alignment_parameters.tdp_scale == 1.0 or allow_for_scaled_tdp
        ), "Do not scale the tdp values during alignment"

        if isinstance(alignment_parameters, AlignmentParameters):
            assert len(alignment_parameters.tdp_speech) == 4
            assert len(alignment_parameters.tdp_silence) == 4
            assert not alignment_parameters.silence_penalties or len(alignment_parameters.silence_penalties) == 2
            assert not alignment_parameters.transition_scales or len(alignment_parameters.transition_scales) == 2

        # align_crp = copy.deepcopy(self.crp)
        align_crp = rasr.CommonRasrParameters(self.crp)

        if alignment_parameters.prior_info.left_context_prior is not None:
            self.name += f"-prL{alignment_parameters.prior_info.left_context_prior.scale}"
        if alignment_parameters.prior_info.center_state_prior is not None:
            self.name += f"-prC{alignment_parameters.prior_info.center_state_prior.scale}"
        if alignment_parameters.prior_info.right_context_prior is not None:
            self.name += f"-prR{alignment_parameters.prior_info.right_context_prior.scale}"
        if alignment_parameters.prior_info.diphone_prior is not None:
            self.name += f"-prJ-C{alignment_parameters.prior_info.diphone_prior.scale}"
        if alignment_parameters.add_all_allophones:
            self.name += "-allAllos"
        if alignment_parameters.tdp_scale is not None:
            self.name += f"-tdpScale-{alignment_parameters.tdp_scale}"
            self.name += f"-spTdp-{format_tdp(alignment_parameters.tdp_speech[:3])}"
            self.name += f"-silTdp-{format_tdp(alignment_parameters.tdp_silence[:3])}"
            if self.feature_scorer_type.is_factored():
                if alignment_parameters.transition_scales is not None:
                    loop_scale, forward_scale = alignment_parameters.transition_scales
                    self.name += f"-loopScale-{loop_scale}"
                    self.name += f"-fwdScale-{forward_scale}"
                else:
                    loop_scale = forward_scale = 1.0

                if alignment_parameters.silence_penalties is not None:
                    sil_loop_penalty, sil_fwd_penalty = alignment_parameters.silence_penalties
                    self.name += f"-silLoopP-{sil_loop_penalty}"
                    self.name += f"-silFwdP-{sil_fwd_penalty}"
                else:
                    sil_fwd_penalty = sil_loop_penalty = 0.0
        else:
            self.name += "-noTdp"

        state_tying = align_crp.acoustic_model_config.state_tying.type

        tdp_transition = (
            alignment_parameters.tdp_speech
            if alignment_parameters.tdp_scale is not None
            else (0.0, 0.0, "infinity", 0.0)
        )
        tdp_silence = (
            alignment_parameters.tdp_silence
            if alignment_parameters.tdp_scale is not None
            else (0.0, 0.0, "infinity", 0.0)
        )
        tdp_nonword = (
            alignment_parameters.tdp_nonword
            if alignment_parameters.tdp_nonword is not None
            else (0.0, 0.0, "infinity", 0.0)
        )

        align_crp.acoustic_model_config = am.acoustic_model_config(
            state_tying=state_tying,
            states_per_phone=label_info.n_states_per_phone,
            state_repetitions=1,
            across_word_model=True,
            early_recombination=False,
            tdp_scale=alignment_parameters.tdp_scale,
            tdp_transition=tdp_transition,
            tdp_silence=tdp_silence,
            tdp_nonword=tdp_nonword,
            nonword_phones=alignment_parameters.non_word_phonemes,
            tying_type="global-and-nonword",
        )

        align_crp.acoustic_model_config.allophones["add-all"] = alignment_parameters.add_all_allophones
        align_crp.acoustic_model_config.allophones["add-from-lexicon"] = not alignment_parameters.add_all_allophones
        if alignment_parameters.add_allophones_from_file is not None:
            align_crp.acoustic_model_config.allophones.add_from_file = alignment_parameters.add_allophones_from_file

        align_crp.acoustic_model_config["state-tying"][
            "use-boundary-classes"
        ] = label_info.phoneme_state_classes.use_boundary()
        align_crp.acoustic_model_config["state-tying"][
            "use-word-end-classes"
        ] = label_info.phoneme_state_classes.use_word_end()

        if crp_update is not None:
            crp_update(align_crp)

        #ToDo add the feature scorer

        if alignment_parameters.tdp_scale is not None:
            if (
                alignment_parameters.tdp_speech[-1]
                + alignment_parameters.tdp_silence[-1]
                + alignment_parameters.tdp_nonword[-1]
                > 0.0
            ):
                import warnings

                warnings.warn("you planned to use exit penalty for alignment, we set this to zero")

        align_crp = self.correct_transition_applicator(
            align_crp,
            correct_fsa_strcuture=correct_fsa_structure,
            allow_for_silence_repetitions=alignment_parameters.allow_for_silence_repetitions,
        )

        alignment = mm.AlignmentJob(
            crp=align_crp,
            feature_flow=self.feature_scorer_flow,
            feature_scorer=None,#feature_scorer,
            use_gpu=gpu,
            rtf=10,
        )
        alignment.rqmt["cpu"] = cpu
        alignment.rqmt["mem"] = mem_rqmt

        alignment.add_alias(f"alignments/align_{self.name}")
        tk.register_output(f"{pre_path}/realignment-{self.name}", alignment.out_alignment_bundle)

        return alignment
