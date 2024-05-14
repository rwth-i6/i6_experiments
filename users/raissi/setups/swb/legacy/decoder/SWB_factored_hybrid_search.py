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
        scorer: Optional[Union[recog.ScliteJob, recog.Hub5ScoreJob, recog.KaldiScorerJob]] = None,
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
            scorer=scorer,
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
            pre_path="decoding-NNLM",
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





