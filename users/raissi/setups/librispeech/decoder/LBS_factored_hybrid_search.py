__all__ = ["LBSFactoredHybridDecoder"]

import dataclasses
import itertools
import numpy as np

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


class LBSFactoredHybridDecoder(BASEFactoredHybridDecoder):
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
        scorer: Optional[Union[recog.ScliteJob, recog.Hub5ScoreJob, recog.Hub5ScoreJob]] = None,
        tensor_map: Optional[Union[dict, DecodingTensorMap]] = None,
        is_multi_encoder_output=False,
        silence_id=40,
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
        self.trafo_lm_config = self.get_eugen_trafo_with_quant_and_compress_config()

    def get_ls_kazuki_lstm_lm_config(
        self,
        scale: Optional[float] = None,
    ) -> rasr.RasrConfig:
        assert self.library_path is not None

        lm_model_dir = Path("/u/mgunz/gunz/dependencies/kazuki_lstmlm_20190627")

        return TfRnnLmRasrConfig(
            vocab_path=lm_model_dir.join_right("vocabulary"),
            meta_graph_path=lm_model_dir.join_right("network.040.meta"),
            returnn_checkpoint=returnn.Checkpoint(lm_model_dir.join_right("network.040.index")),
            scale=scale,
            libraries=self.library_path,
            state_manager="lstm",
        ).get()

    def get_eugen_trafo_with_quant_and_compress_config(
        self,
        min_batch_size: int = 0,
        opt_batch_size: int = 64,
        max_batch_size: int = 64,
        scale: Optional[float] = None,
    ) -> rasr.RasrConfig:
        # assert self.library_path is not None

        trafo_config = rasr.RasrConfig()

        # Taken from /work/asr4/beck/setups/librispeech/2019-09-15_state_compression/work/recognition/advanced_tree_search/AdvancedTreeSearchJob.9vgSulF2hfbc/work/recognition.config

        trafo_config.min_batch_size = min_batch_size
        trafo_config.opt_batch_size = opt_batch_size
        trafo_config.max_batch_size = max_batch_size
        trafo_config.allow_reduced_history = True
        if scale is not None:
            trafo_config.scale = scale
        trafo_config.type = "tfrnn"
        trafo_config.vocab_file = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/vocabulary"
        trafo_config.transform_output_negate = True
        trafo_config.vocab_unknown_word = "<UNK>"

        trafo_config.input_map.info_0.param_name = "word"
        trafo_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        trafo_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

        trafo_config.input_map.info_1.param_name = "state-lengths"
        trafo_config.input_map.info_1.tensor_name = "output/rec/dec_0_self_att_att/state_lengths"

        trafo_config.loader.type = "meta"
        trafo_config.loader.meta_graph_file = (
            "/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/graph.meta"
        )
        model_path = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/epoch.030"
        trafo_config.loader.saved_model_file = rasr.StringWrapper(model_path, f"{model_path}.index")
        trafo_config.loader.required_libraries = self.library_path

        trafo_config.nn_output_compression.bits_per_val = 16
        trafo_config.nn_output_compression.epsilon = 0.001
        trafo_config.nn_output_compression.type = "fixed-quantization"

        trafo_config.output_map.info_0.param_name = "softmax"
        trafo_config.output_map.info_0.tensor_name = "output/rec/decoder/add"

        trafo_config.output_map.info_1.param_name = "weights"
        trafo_config.output_map.info_1.tensor_name = "output/rec/output/W/read"

        trafo_config.output_map.info_2.param_name = "bias"
        trafo_config.output_map.info_2.tensor_name = "output/rec/output/b/read"

        trafo_config.softmax_adapter.type = "quantized-blas-nce-16bit"
        trafo_config.softmax_adapter.weights_bias_epsilon = 0.001

        trafo_config.state_compression.bits_per_val = 16
        trafo_config.state_compression.epsilon = 0.001
        trafo_config.state_compression.type = "fixed-quantization"

        trafo_config.state_manager.cache_prefix = True
        trafo_config.state_manager.min_batch_size = min_batch_size
        trafo_config.state_manager.min_common_prefix_length = 0
        trafo_config.state_manager.type = "transformer-with-common-prefix-16bit"

        trafo_config.state_manager.var_map.item_0.common_prefix_initial_value = (
            "output/rec/dec_0_self_att_att/zeros_1:0"
        )
        trafo_config.state_manager.var_map.item_0.common_prefix_initializer = (
            "output/rec/dec_0_self_att_att/common_prefix/Assign:0"
        )
        trafo_config.state_manager.var_map.item_0.var_name = "output/rec/dec_0_self_att_att/keep_state_var:0"

        trafo_config.state_manager.var_map.item_1.common_prefix_initial_value = (
            "output/rec/dec_1_self_att_att/zeros_1:0"
        )
        trafo_config.state_manager.var_map.item_1.common_prefix_initializer = (
            "output/rec/dec_1_self_att_att/common_prefix/Assign:0"
        )
        trafo_config.state_manager.var_map.item_1.var_name = "output/rec/dec_1_self_att_att/keep_state_var:0"

        trafo_config.state_manager.var_map.item_2.common_prefix_initial_value = (
            "output/rec/dec_2_self_att_att/zeros_1:0"
        )
        trafo_config.state_manager.var_map.item_2.common_prefix_initializer = (
            "output/rec/dec_2_self_att_att/common_prefix/Assign:0"
        )
        trafo_config.state_manager.var_map.item_2.var_name = "output/rec/dec_2_self_att_att/keep_state_var:0"

        trafo_config.state_manager.var_map.item_3.common_prefix_initial_value = (
            "output/rec/dec_3_self_att_att/zeros_1:0"
        )
        trafo_config.state_manager.var_map.item_3.common_prefix_initializer = (
            "output/rec/dec_3_self_att_att/common_prefix/Assign:0"
        )
        trafo_config.state_manager.var_map.item_3.var_name = "output/rec/dec_3_self_att_att/keep_state_var:0"

        trafo_config.state_manager.var_map.item_4.common_prefix_initial_value = (
            "output/rec/dec_4_self_att_att/zeros_1:0"
        )
        trafo_config.state_manager.var_map.item_4.common_prefix_initializer = (
            "output/rec/dec_4_self_att_att/common_prefix/Assign:0"
        )
        trafo_config.state_manager.var_map.item_4.var_name = "output/rec/dec_4_self_att_att/keep_state_var:0"

        trafo_config.state_manager.var_map.item_5.common_prefix_initial_value = (
            "output/rec/dec_5_self_att_att/zeros_1:0"
        )
        trafo_config.state_manager.var_map.item_5.common_prefix_initializer = (
            "output/rec/dec_5_self_att_att/common_prefix/Assign:0"
        )
        trafo_config.state_manager.var_map.item_5.var_name = "output/rec/dec_5_self_att_att/keep_state_var:0"

        return trafo_config

    def get_eugen_trafo_config(
        self,
        min_batch_size: int = 0,
        opt_batch_size: int = 64,
        max_batch_size: int = 64,
        scale: Optional[float] = None,
    ) -> rasr.RasrConfig:
        # assert self.library_path is not None


        trafo_config = rasr.RasrConfig()

        trafo_config.min_batch_size = min_batch_size
        trafo_config.opt_batch_size = opt_batch_size
        trafo_config.max_batch_size = max_batch_size
        trafo_config.allow_reduced_history = True
        if scale is not None:
            trafo_config.scale = scale
        trafo_config.type = "tfrnn"
        trafo_config.vocab_file = tk.Path("/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/vocabulary", cached=True)
        trafo_config.transform_output_negate = True
        trafo_config.vocab_unknown_word = "<UNK>"

        trafo_config.input_map.info_0.param_name = "word"
        trafo_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        trafo_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"

        trafo_config.input_map.info_1.param_name = "state-lengths"
        trafo_config.input_map.info_1.tensor_name = "output/rec/dec_0_self_att_att/state_lengths"

        trafo_config.loader.type = "meta"
        trafo_config.loader.meta_graph_file = (
            "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/trafo-lm_eugen/integrated_fixup_graph_no_cp_no_quant.meta"
        )
        model_path = "/work/asr3/raissi/shared_workspaces/gunz/dependencies/ls-eugen-trafo-lm/epoch.030"
        trafo_config.loader.saved_model_file = rasr.StringWrapper(model_path, f"{model_path}.index")
        trafo_config.loader.required_libraries = self.library_path

        trafo_config.output_map.info_0.param_name = "softmax"
        trafo_config.output_map.info_0.tensor_name = "output/rec/decoder/add"

        trafo_config.output_map.info_1.param_name = "weights"
        trafo_config.output_map.info_1.tensor_name = "output/rec/output/W/read"

        trafo_config.output_map.info_2.param_name = "bias"
        trafo_config.output_map.info_2.tensor_name = "output/rec/output/b/read"


        trafo_config.state_manager.cache_prefix = True
        trafo_config.state_manager.min_batch_size = min_batch_size
        trafo_config.state_manager.min_common_prefix_length = 0
        trafo_config.state_manager.type = "transformer"
        trafo_config.softmax_adapter.type = "blas-nce"

        return trafo_config

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
            lm_config=self.trafo_lm_config,
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
