__all__ = ["BASEFactoredHybridDecoder", "BASEFactoredHybridAligner"]

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
from sisyphus.delayed_ops import DelayedBase, Delayed

Path = tk.Path

import i6_core.am as am
import i6_core.corpus as corpus_recipes
import i6_core.lm as lm
import i6_core.mm as mm
import i6_core.rasr as rasr
import i6_core.recognition as recog

from i6_core.returnn.flow import (
    make_precomputed_hybrid_tf_feature_flow,
    add_tf_flow_to_base_flow,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
    RasrStateTying,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    default_posterior_scales,
    PriorInfo,
    PosteriorScales,
    SearchParameters,
    AlignmentParameters,
)
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    BASEFactoredHybridDecoder,
    round2,
    RasrFeatureScorer,
    DecodingTensorMap,
    RecognitionJobs,
    check_prior_info,
    get_factored_feature_scorer,
    get_nn_precomputed_feature_scorer,
)
from i6_experiments.users.raissi.setups.common.decoder.factored_hybrid_feature_scorer import (
    FactoredHybridFeatureScorer,
)

from i6_experiments.users.raissi.setups.common.decoder.statistics import ExtractSearchStatisticsJob
from i6_experiments.users.raissi.setups.common.util.tdp import format_tdp_val, format_tdp
from i6_experiments.users.raissi.setups.common.util.argmin import ComputeArgminJob
from i6_experiments.users.raissi.setups.common.data.typings import (
    TDP,
    Float,
)


class SWBFactoredHybridDecoder(BASEFactoredHybridDecoder):
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
        eval_files,
        tensor_map: Optional[Union[dict, DecodingTensorMap]] = None,
        is_multi_encoder_output=False,
        silence_id=3,
        set_batch_major_for_feature_scorer: bool = True,
        tf_library: Optional[Union[str, Path]] = None,
        lm_gc_simple_hash=None,
        gpu=False,
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
            eval_files=eval_files,
            tensor_map=tensor_map,
            is_multi_encoder_output=is_multi_encoder_output,
            silence_id=silence_id,
            set_batch_major_for_feature_scorer=set_batch_major_for_feature_scorer,
            tf_library=tf_library,
            lm_gc_simple_hash=lm_gc_simple_hash,
            gpu=gpu,
        )

    def get_tfrnn_lm_config(
        self,
        name,
        scale,
        min_batch_size=1024,
        opt_batch_size=1024,
        max_batch_size=1024,
        allow_reduced_hist=None,
        async_lm=False,
        single_step_only=False,
    ):
        res = copy.deepcopy(self.tfrnn_lms[name])
        if allow_reduced_hist is not None:
            res.allow_reduced_history = allow_reduced_hist
        res.scale = scale
        res.min_batch_size = min_batch_size
        res.opt_batch_size = opt_batch_size
        res.max_batch_size = max_batch_size
        if async_lm:
            res["async"] = async_lm
        if async_lm or single_step_only:
            res.single_step_only = True

        return res

    def get_nce_lm(self, **kwargs):
        lstm_lm_config = self.get_tfrnn_lm_config(**kwargs)
        lstm_lm_config.output_map.info_1.param_name = "weights"
        lstm_lm_config.output_map.info_1.tensor_name = "output/W/read"
        lstm_lm_config.output_map.info_2.param_name = "bias"
        lstm_lm_config.output_map.info_2.tensor_name = "output/b/read"
        lstm_lm_config.softmax_adapter.type = "blas_nce"

        return lstm_lm_config

    def get_rnn_config(self, scale=1.0):
        lmConfigParams = {
            "name": "kazuki_real_nce",
            "min_batch_size": 1024,
            "opt_batch_size": 1024,
            "max_batch_size": 1024,
            "scale": scale,
            "allow_reduced_hist": True,
        }

        return self.get_nce_lm(**lmConfigParams)

    def get_rnn_full_config(self, scale=13.0):
        lmConfigParams = {
            "name": "kazuki_full",
            "min_batch_size": 4,
            "opt_batch_size": 64,
            "max_batch_size": 128,
            "scale": scale,
            "allow_reduced_hist": True,
        }
        return self.get_tfrnn_lm_config(**lmConfigParams)

    def add_tfrnn_lms(self):
        tfrnn_dir = "/u/beck/setups/swb1/2018-06-08_nnlm_decoding/dependencies/tfrnn"
        # backup:  '/work/asr4/raissi/ms-thesis-setups/dependencies/tfrnn'

        rnn_lm_config = sprint.SprintConfig()
        rnn_lm_config.type = "tfrnn"
        rnn_lm_config.vocab_file = Path("%s/vocabulary" % tfrnn_dir)
        rnn_lm_config.transform_output_negate = True
        rnn_lm_config.vocab_unknown_word = "<unk>"

        rnn_lm_config.loader.type = "meta"
        rnn_lm_config.loader.meta_graph_file = Path("%s/network.019.meta" % tfrnn_dir)
        rnn_lm_config.loader.saved_model_file = sprint.StringWrapper(
            "%s/network.019" % tfrnn_dir, Path("%s/network.019.index" % tfrnn_dir)
        )
        rnn_lm_config.loader.required_libraries = Path(self.native_lstm_path)

        rnn_lm_config.input_map.info_0.param_name = "word"
        rnn_lm_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        rnn_lm_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

        rnn_lm_config.output_map.info_0.param_name = "softmax"
        rnn_lm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        kazuki_fake_nce_lm = copy.deepcopy(rnn_lm_config)
        tfrnn_dir = "/work/asr3/beck/setups/swb1/2018-06-08_nnlm_decoding/dependencies/tfrnn_nce"
        kazuki_fake_nce_lm.vocab_file = Path("%s/vocabmap.freq_sorted.txt" % tfrnn_dir)
        kazuki_fake_nce_lm.loader.meta_graph_file = Path("%s/inference.meta" % tfrnn_dir)
        kazuki_fake_nce_lm.loader.saved_model_file = sprint.StringWrapper(
            "%s/network.018" % tfrnn_dir, Path("%s/network.018.index" % tfrnn_dir)
        )

        kazuki_real_nce_lm = copy.deepcopy(kazuki_fake_nce_lm)
        kazuki_real_nce_lm.output_map.info_0.tensor_name = "sbn/output_batch_major"

        self.tfrnn_lms["kazuki_real_nce"] = kazuki_real_nce_lm

    def add_lstm_full(self):
        tfrnn_dir = "/work/asr4/raissi/ms-thesis-setups/lm-sa-swb/dependencies/lstm-lm-kazuki"

        rnn_lm_config = sprint.SprintConfig()
        rnn_lm_config.type = "tfrnn"
        rnn_lm_config.vocab_file = Path(("/").join([tfrnn_dir, "vocabulary"]))
        rnn_lm_config.transform_output_negate = True
        rnn_lm_config.vocab_unknown_word = "<unk>"

        rnn_lm_config.loader.type = "meta"
        rnn_lm_config.loader.meta_graph_file = Path("%s/network.019.meta" % tfrnn_dir)
        rnn_lm_config.loader.saved_model_file = sprint.StringWrapper(
            "%s/network.019" % tfrnn_dir, Path("%s/network.019.index" % tfrnn_dir)
        )
        rnn_lm_config.loader.required_libraries = Path(self.native_lstm_path)

        rnn_lm_config.input_map.info_0.param_name = "word"
        rnn_lm_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        rnn_lm_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

        rnn_lm_config.output_map.info_0.param_name = "softmax"
        rnn_lm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        self.tfrnn_lms["kazuki_full"] = rnn_lm_config

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
        remove_or_set_concurrency: Union[bool, int] = False,
        search_rqmt_update=None,
        adv_search_extra_config: Optional[rasr.RasrConfig] = None,
        adv_search_extra_post_config: Optional[rasr.RasrConfig] = None,
    ) -> RecognitionJobs:
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
            remove_or_set_concurrency=remove_or_set_concurrency,
            search_rqmt_update=search_rqmt_update,
            adv_search_extra_config=adv_search_extra_config,
            adv_search_extra_post_config=adv_search_extra_post_config,
        )

    def recognize_ls_trafo_lm(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        calculate_stats=False,
        is_min_duration=False,
        opt_lm_am=True,
        only_lm_opt=True,
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
        rtf_gpu: Optional[float] = None,
        rtf_cpu: Optional[float] = None,
        create_lattice: bool = True,
        adv_search_extra_config: Optional[rasr.RasrConfig] = None,
        adv_search_extra_post_config: Optional[rasr.RasrConfig] = None,
    ) -> RecognitionJobs:
        return self.recognize(
            add_sis_alias_and_output=add_sis_alias_and_output,
            calculate_stats=calculate_stats,
            gpu=gpu,
            cpu_rqmt=cpu_rqmt,
            mem_rqmt=mem_rqmt,
            is_min_duration=is_min_duration,
            is_nn_lm=True,
            keep_value=keep_value,
            label_info=label_info,
            lm_config=self.get_ls_eugen_trafo_config(),
            name_override=name_override,
            name_prefix=name_prefix,
            num_encoder_output=num_encoder_output,
            only_lm_opt=only_lm_opt,
            opt_lm_am=opt_lm_am,
            pre_path="decoding-eugen-trafo-lm",
            rerun_after_opt_lm=rerun_after_opt_lm,
            search_parameters=search_parameters,
            use_estimated_tdps=use_estimated_tdps,
            crp_update=crp_update,
            rtf_cpu=rtf_cpu,
            rtf_gpu=rtf_gpu,
            create_lattice=create_lattice,
            adv_search_extra_config=adv_search_extra_config,
            adv_search_extra_post_config=adv_search_extra_post_config,
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
        only_lm_opt=True,
        opt_lm_am=True,
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
        remove_or_set_concurrency: Union[bool, int] = False,
    ) -> RecognitionJobs:
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

        if search_crp.lexicon_config.normalize_pronunciation:
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
            if search_parameters.altas is not None:
                name += f"-ALTAS{search_parameters.altas}"
            if search_parameters.add_all_allophones:
                name += "-allAllos"
            if not create_lattice:
                name += "-noLattice"

        if search_parameters.tdp_scale is not None:
            if name_override is None:
                name += f"-tdpScale-{search_parameters.tdp_scale}"
                name += f"-spTdp-{format_tdp(search_parameters.tdp_speech)}"
                name += f"-silTdp-{format_tdp(search_parameters.tdp_silence)}"
                if (
                    not search_parameters.tdp_speech[2] == "infinity"
                    and not search_parameters.tdp_silence[2] == "infinity"
                ):
                    name += "-withSkip"

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
        tdp_non_word = (
            search_parameters.tdp_non_word
            if search_parameters.tdp_non_word is not None
            else (0.0, 0.0, "infinity", 0.0)
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
            tdp_nonword=tdp_non_word,
            nonword_phones="[LAUGHTER],[NOISE],[VOCALIZEDNOISE]",
            #search_parameters.non_word_phonemes,
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

        # lm config update
        original_lm_config = search_crp.language_model_config
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

        la_options = self.get_lookahead_options()

        adv_search_extra_config = (
            copy.deepcopy(adv_search_extra_config) if adv_search_extra_config is not None else rasr.RasrConfig()
        )
        if search_parameters.altas is not None:
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                search_parameters.altas
            )
        if search_parameters.lm_lookahead_scale is not None:
            name += f"-lh{search_parameters.lm_lookahead_scale}"
            # Use 4gram for lookahead. The lookahead LM must not be too good.
            #
            # Half the normal LM scale is a good starting value.

            # To validate the assumption the original LM is a 4gram
            assert original_lm_config.type.lower() == "arpa"

            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.separate_lookahead_lm = True
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lm_lookahead.lm_lookahead_scale = (
                search_parameters.lm_lookahead_scale
            )
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm.file = (
                original_lm_config.file
            )
            # TODO(future): Add LM image instead of file here.
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm.scale = 1.0
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.lookahead_lm.type = "ARPA"

        if search_parameters.lm_lookahead_history_limit > 1:
            la_options["history_limit"] = search_parameters.lm_lookahead_history_limit
            name += f"-lhHisLim{search_parameters.lm_lookahead_history_limit}"

        if self.feature_scorer_type.is_factored():
            feature_scorer = get_factored_feature_scorer(
                context_type=self.context_type,
                feature_scorer_type=self.feature_scorer_type,
                label_info=label_info,
                feature_scorer_config=self.fs_config,
                mixtures=self.mixtures,
                silence_id=self.silence_id,
                prior_info=search_parameters.prior_info,
                posterior_scales=search_parameters.posterior_scales,
                num_label_contexts=label_info.n_contexts,
                num_states_per_phone=label_info.n_states_per_phone,
                num_encoder_output=num_encoder_output,
                loop_scale=loop_scale,
                forward_scale=forward_scale,
                silence_loop_penalty=sil_loop_penalty,
                silence_forward_penalty=sil_fwd_penalty,
                use_estimated_tdps=use_estimated_tdps,
                state_dependent_tdp_file=search_parameters.state_dependent_tdps,
                is_min_duration=is_min_duration,
                is_multi_encoder_output=self.is_multi_encoder_output,
                is_batch_major=self.set_batch_major_for_feature_scorer,
            )
        elif self.feature_scorer_type.is_nnprecomputed():
            scale = 1.0
            if search_parameters.posterior_scales is not None:
                if context_type.is_joint_diphone():
                    scale = search_parameters.posterior_scales["joint-diphone-scale"]
                elif context_type.is_monophone():
                    scale = search_parameters.posterior_scales["center-state-scale"]

                name += f"-Am{scale}"
            feature_scorer = get_nn_precomputed_feature_scorer(
                posterior_scale=scale,
                context_type=self.context_type,
                feature_scorer_type=self.feature_scorer_type,
                mixtures=self.mixtures,
                prior_info=search_parameters.prior_info,
            )
        else:
            raise NotImplementedError

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
            create_lattice=create_lattice,
            # separate_lm_image_gc_generation=True,
            model_combination_config=model_combination_config,
            model_combination_post_config=None,
            extra_config=adv_search_extra_config,
            extra_post_config=adv_search_extra_post_config,
        )

        if search_rqmt_update is not None:
            search.rqmt.update(search_rqmt_update)
            search.cpu = 1

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
                # tk.register_output(f"{pre}statistics/rtf/{name}.rtf", stat.decoding_rtf)
                # tk.register_output(f"{pre}statistics/rtf/{name}.recognizer.rtf", stat.recognizer_rtf)
                # tk.register_output(f"{pre}statistics/rtf/{name}.stats", stat.ss_statistics)
        else:
            stat = None

        if not create_lattice:
            return RecognitionJobs(
                lat2ctm=None,
                sclite=None,
                search=search,
                search_crp=crp,
                search_feature_scorer=feature_scorer,
                search_stats=stat,
            )

        lat2ctm_extra_config = rasr.RasrConfig()
        lat2ctm_extra_config.flf_lattice_tool.network.to_lemma.links = "best"
        lat2ctm = recog.LatticeToCtmJob(
            crp=search_crp,
            lattice_cache=search.out_lattice_bundle,
            parallelize=True,
            best_path_algo="bellman-ford",
            extra_config=lat2ctm_extra_config,
            fill_empty_segments=True,
        )

        s_kwrgs = copy.deepcopy(self.eval_files)
        s_kwrgs["hyp"] = lat2ctm.out_ctm_file
        scorer = recog.Hub5ScoreJob(**s_kwrgs)

        if add_sis_alias_and_output:
            tk.register_output(f"{pre_path}/{name}.wer", scorer.out_report_dir)

        if opt_lm_am and search_parameters.altas is None:
            assert search_parameters.beam >= 15.0

            opt = recog.OptimizeAMandLMScaleJob(
                crp=search_crp,
                lattice_cache=search.out_lattice_bundle,
                initial_am_scale=pron_scale,
                initial_lm_scale=search_parameters.lm_scale,
                scorer_cls=recog.Hub5ScoreJob,
                scorer_kwargs=s_kwrgs,
                opt_only_lm_scale=only_lm_opt,
            )
            #opt.rqmt = None

            if add_sis_alias_and_output:
                tk.register_output(
                    f"{pre_path}/{name}/onlyLmOpt{only_lm_opt}.optlm.txt",
                    opt.out_log_file,
                )

            if rerun_after_opt_lm:
                rounded_lm_scale = Delayed(opt.out_best_lm_score).function(round2)
                params = search_parameters.with_lm_scale(rounded_lm_scale)
                name_after_rerun = name

                if opt.out_best_lm_score.get() is not None:
                    name_after_rerun = re.sub(r"Lm[0-9]*.[0.9*]", f"Lm{rounded_lm_scale}", name)

                name_prefix_len = len(f"{name_prefix}{self.name}/")

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

        return RecognitionJobs(
            lat2ctm=lat2ctm,
            sclite=scorer,
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
                remove_or_set_concurrency=False,
            )
            for ((c, l, r), tdp, tdp_sl, tdp_sp) in itertools.product(prior_scales, tdp_scales, tdp_sil, tdp_speech)
        }
        jobs_num_e = {k: v.sclite.out_num_errors for k, v in jobs.items()}

        for ((c, l, r), tdp, tdp_sl, tdp_sp), recog_jobs in jobs.items():
            if cpu_slow:
                recog_jobs.search.update_rqmt("run", {"cpu_slow": True})

            pre_name = f"{pre_path}/{self.name}/Lm{recog_args.lm_scale}-Pron{recog_args.pron_scale}-pC{c}-pL{l}-pR{r}-tdp{tdp}-tdpSil{format_tdp(tdp_sl)}-tdpSp{format_tdp(tdp_sp)}"

            recog_jobs.lat2ctm.set_keep_value(keep_value)
            recog_jobs.search.set_keep_value(keep_value)

            recog_jobs.search.add_alias(pre_name)
            tk.register_output(f"{pre_name}.wer", recog_jobs.sclite.out_report_dir)

        best_overall_wer = ComputeArgminJob({k: v.sclite.out_wer for k, v in jobs.items()})
        best_overall_n = ComputeArgminJob(jobs_num_e)
        tk.register_output(
            f"decoding/scales-best/{self.name}/args",
            best_overall_n.out_argmin,
        )
        tk.register_output(
            f"decoding/scales-best/{self.name}/wer",
            best_overall_wer.out_min,
        )

        # cannot destructure, need to use indices
        best_priors = best_overall_n.out_argmin[0]
        best_tdp_scale = best_overall_n.out_argmin[1]
        best_tdp_sil = best_overall_n.out_argmin[2]
        best_tdp_sp = best_overall_n.out_argmin[3]

        def push_delayed_tuple(
            argmin: DelayedBase,
        ) -> Tuple[DelayedBase, DelayedBase, DelayedBase, DelayedBase]:
            return tuple(argmin[i] for i in range(4))

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
