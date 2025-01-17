__all__ = ["LBSFactoredHybridDecoder"]

import dataclasses
import itertools
import numpy as np

from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union
from IPython import embed

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase, Delayed, DelayedFormat

Path = tk.Path

import i6_core.rasr as rasr
import i6_core.recognition as recog
import i6_core.returnn as returnn


from i6_experiments.users.raissi.setups.common.data.factored_label import (
    LabelInfo,
    PhoneticContext,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    SearchParameters,
)
from i6_experiments.users.raissi.setups.common.decoder.BASE_factored_hybrid_search import (
    BASEFactoredHybridDecoder,
    RasrFeatureScorer,
    DecodingTensorMap,
    DecodingJobs,

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
        eval_args,
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
            eval_args=eval_args,
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
        self.lstm_lm_config = self.get_kazuki_lstm_config()


    def get_kazuki_lstm_config(
        self,
        min_batch_size: int = 4,
        opt_batch_size: int = 8,
        max_batch_size: int = 8,
        scale: Optional[float] = None,
    ) -> rasr.RasrConfig:

        assert self.library_path is not None

        kazuki_lstm_path = tk.Path("/work/asr4/raissi/setups/librispeech/960-ls/dependencies/lstm-lm_kazuki", hash_overwrite="KAZUKI_LSTM_LM")

        lstm_config = rasr.RasrConfig()

        #model and graph info
        lstm_config.loader.type = "meta"
        lstm_config.loader.meta_graph_file = kazuki_lstm_path.join_right("graph.meta")
        lstm_config.loader.saved_model_file = DelayedFormat("/work/asr4/raissi/setups/librispeech/960-ls/dependencies/lstm-lm_kazuki/network.040")
        lstm_config.loader.required_libraries = self.library_path

        lstm_config.type = "tfrnn"
        lstm_config.vocab_file = kazuki_lstm_path.join_right("vocabulary")
        lstm_config.transform_output_negate = True
        lstm_config.vocab_unknown_word = "<UNK>"

        lstm_config.min_batch_size = min_batch_size
        lstm_config.opt_batch_size = opt_batch_size
        lstm_config.max_batch_size = max_batch_size
        #trafo_config.allow_reduced_history = True
        if scale is not None:
            lstm_config.scale = scale

        #Tensor names
        #in
        lstm_config.input_map.info_0.param_name = "word"
        lstm_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        lstm_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"
        #out
        lstm_config.output_map.info_0.param_name = "softmax"
        lstm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        return lstm_config

    def get_nick_lstm_config(
        self,
        min_batch_size: int = 4,
        opt_batch_size: int = 8,
        max_batch_size: int = 8,
        scale: Optional[float] = None,
    ) -> rasr.RasrConfig:

        assert self.library_path is not None

        nick_prepath = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/lstm-lm_nick"

        lstm_config = rasr.RasrConfig()

        #model and graph info
        lstm_config.loader.type = "meta"
        lstm_config.loader.meta_graph_file = ("/").join([nick_prepath, "graph.meta"])
        lstm_config.loader.saved_model_file = DelayedFormat("/u/rossenbach/experiments/domain_lm_aed_2024/work/i6_core/returnn/training/ReturnnTrainingJob.d2q0Od3IdTEu/output/models/epoch.300")
        lstm_config.loader.required_libraries = self.library_path

        lstm_config.type = "tfrnn"
        lstm_config.vocab_file = ("/").join([nick_prepath, "lm.vocab.txt"])
        lstm_config.transform_output_negate = True
        lstm_config.vocab_unknown_word = "<UNK>"

        lstm_config.min_batch_size = min_batch_size
        lstm_config.opt_batch_size = opt_batch_size
        lstm_config.max_batch_size = max_batch_size
        #trafo_config.allow_reduced_history = True
        if scale is not None:
            lstm_config.scale = scale

        #Tensor names
        #in
        lstm_config.input_map.info_0.param_name = "word"
        lstm_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        lstm_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"
        #out
        lstm_config.output_map.info_0.param_name = "softmax"
        lstm_config.output_map.info_0.tensor_name = "output/output_batch_major"

        return lstm_config

    def get_kazuki_trafo_config(
        self,
        min_batch_size: int = 0,
        opt_batch_size: int = 64,
        max_batch_size: int = 64,
        scale: Optional[float] = None,
    ) -> rasr.RasrConfig:

        assert self.library_path is not None
        if "generic-seq2seq-dev" not in self.crp.flf_tool_exe.get_path().split('arch')[0]:
            from i6_experiments.users.raissi.utils.default_tools import U16_RASR_GENERIC_SEQ2SEQ
            self.crp.flf_tool_exe = U16_RASR_GENERIC_SEQ2SEQ.join_right("flf-tool.linux-x86_64-standard")

        dependency_path = Path("/work/asr4/raissi/setups/librispeech/960-ls/dependencies/trafo-lm_kazuki/IS2019", hash_overwrite="LBS_LM_KAZUKI")

        trafo_config = rasr.RasrConfig()

        #model and graph info
        trafo_config.loader.type = "meta"
        trafo_config.loader.meta_graph_file = dependency_path.join_right("inference.graph")
        trafo_config.loader.saved_model_file = returnn.Checkpoint(
            index_path=dependency_path.join_right("network.030.index"))
        trafo_config.loader.required_libraries = self.library_path

        trafo_config.type = "simple-transformer"
        trafo_config.vocab_file = "/work/asr4/raissi/setups/librispeech/960-ls/dependencies/trafo-lm_kazuki/IS2019/vocabulary"
        trafo_config.transform_output_negate = True
        trafo_config.vocab_unknown_word = "<UNK>"

        trafo_config.min_batch_size = min_batch_size
        trafo_config.opt_batch_size = opt_batch_size
        trafo_config.max_batch_size = max_batch_size
        trafo_config.allow_reduced_history = True
        if scale is not None:
            trafo_config.scale = scale

        #Tensor names
        #in
        trafo_config.input_map.info_0.param_name = "word"
        trafo_config.input_map.info_0.tensor_name = "extern_data/placeholders/delayed/delayed"
        trafo_config.input_map.info_0.seq_length_tensor_name = "extern_data/placeholders/delayed/delayed_dim0_size"
        #out
        trafo_config.output_map.info_0.param_name = "softmax"
        trafo_config.output_map.info_0.tensor_name = "output/output_batch_major"

        return trafo_config



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

    def recognize_lstm_lm(
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
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        rtf_gpu: Optional[float] = None,
        rtf_cpu: Optional[float] = None,
        create_lattice: bool = True,
        lm_lookahead_options: Optional = None,
        adv_search_extra_config: Optional[rasr.RasrConfig] = None,
        adv_search_extra_post_config: Optional[rasr.RasrConfig] = None,
    ) -> DecodingJobs:
        if lm_lookahead_options is None:
            lm_lookahead_options = {"clow": 2000, "chigh": 3000}
        return self.recognize(
            add_sis_alias_and_output=add_sis_alias_and_output,
            calculate_stats=calculate_stats,
            cpu_rqmt=cpu_rqmt,
            mem_rqmt=mem_rqmt,
            is_min_duration=is_min_duration,
            is_nn_lm=True,
            keep_value=keep_value,
            label_info=label_info,
            lm_config=self.lstm_lm_config,
            name_override=name_override,
            name_prefix=name_prefix,
            num_encoder_output=num_encoder_output,
            only_lm_opt=only_lm_opt,
            opt_lm_am=opt_lm_am,
            cn_decoding=cn_decoding,
            pre_path="decoding-lstm-lm",
            rerun_after_opt_lm=rerun_after_opt_lm,
            search_parameters=search_parameters,
            use_estimated_tdps=use_estimated_tdps,
            crp_update=crp_update,
            rtf_cpu=rtf_cpu,
            rtf_gpu=rtf_gpu,
            lm_lookahead_options=lm_lookahead_options,
            create_lattice=create_lattice,
            adv_search_extra_config=adv_search_extra_config,
            adv_search_extra_post_config=adv_search_extra_post_config,
        )


    def recognize_trafo_lm(
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
        cpu_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        crp_update: Optional[Callable[[rasr.RasrConfig], Any]] = None,
        rtf_gpu: Optional[float] = None,
        rtf_cpu: Optional[float] = None,
        create_lattice: bool = True,
        lm_lookahead_options: Optional = None,
        adv_search_extra_config: Optional[rasr.RasrConfig] = None,
        adv_search_extra_post_config: Optional[rasr.RasrConfig] = None,
    ) -> DecodingJobs:
        if lm_lookahead_options is None:
            lm_lookahead_options = {"clow": 2000, "chigh": 3000}
        return self.recognize(
            add_sis_alias_and_output=add_sis_alias_and_output,
            calculate_stats=calculate_stats,
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
            cn_decoding=cn_decoding,
            pre_path="decoding-trafo-lm",
            rerun_after_opt_lm=rerun_after_opt_lm,
            search_parameters=search_parameters,
            use_estimated_tdps=use_estimated_tdps,
            crp_update=crp_update,
            rtf_cpu=rtf_cpu,
            rtf_gpu=rtf_gpu,
            lm_lookahead_options=lm_lookahead_options,
            create_lattice=create_lattice,
            adv_search_extra_config=adv_search_extra_config,
            adv_search_extra_post_config=adv_search_extra_post_config,
        )
