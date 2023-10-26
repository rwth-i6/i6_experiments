__all__ = ["DecodingTensorMap", "FHDecoder"]

import copy
import dataclasses
from dataclasses import dataclass
import itertools
import numpy as np
import typing

import i6_core.corpus
import i6_core.recognition as recog
from i6_core import am, mm, rasr, returnn

from sisyphus import tk
from sisyphus.delayed_ops import Delayed, DelayedBase, DelayedJoin

from ...common.decoder.rtf import ExtractSearchStatisticsJob
from ...common.lm_config import TfRnnLmRasrConfig
from ...common.tdp import TDP, format_tdp
from ..factored import LabelInfo, PhoneticContext
from ..rust_scorer import RecompileTfGraphJob
from ..util.argmin import ComputeArgminJob
from .config import PosteriorScales, PriorInfo, SearchParameters
from .scorer import FactoredHybridFeatureScorer


@dataclass(eq=True, frozen=True)
class DecodingTensorMap:
    """Map of tensor names used during decoding."""

    in_classes: str
    """Name of the input tensor carrying the classes."""

    in_encoder_output: str
    """
    Name of input tensor for feeding in the previously obtained encoder output while
    processing the state posteriors.

    This can be different from `out_encoder_output` because the tensor name for feeding
    in the intermediate data for computing the output softmaxes can be different from
    the tensor name where the encoder-output is provided.
    """

    in_delta_encoder_output: str
    """
    Name of input tensor for feeding in the previously obtained delta encoder output.

    See `in_encoder_output` for further explanation and why this can be different
    from `out_delta_encoder_output`.
    """

    in_data: str
    """Name of the input tensor carrying the audio features."""

    in_seq_length: str
    """Tensor name of the tensor where the feature length is fed in (as a dimension)."""

    out_encoder_output: str
    """Name of output tensor carrying the raw encoder output (before any softmax)."""

    out_delta_encoder_output: str
    """Name of the output tensor carrying the raw delta encoder output (before any softmax)."""

    out_left_context: str
    """Tensor name of the softmax for the left context."""

    out_right_context: str
    """Tensor name of the softmax for the right context."""

    out_center_state: str
    """Tensor name of the softmax for the center state."""

    out_delta: str
    """Tensor name of the softmax for the delta output."""

    @classmethod
    def default(cls) -> "DecodingTensorMap":
        return DecodingTensorMap(
            in_classes="extern_data/placeholders/classes/classes",
            in_data="extern_data/placeholders/data/data",
            in_seq_length="extern_data/placeholders/data/data_dim0_size",
            in_delta_encoder_output="delta-ce/output_batch_major",
            in_encoder_output="encoder-output/output_batch_major",
            out_encoder_output="encoder-output/output_batch_major",
            out_delta_encoder_output="deltaEncoder-output/output_batch_major",
            out_left_context="left-output/output_batch_major",
            out_right_context="right-output/output_batch_major",
            out_center_state="center-output/output_batch_major",
            out_delta="delta-ce/output_batch_major",
        )


@dataclass
class RecognitionJobs:
    lat2ctm: typing.Optional[recog.LatticeToCtmJob]
    sclite: typing.Optional[recog.ScliteJob]
    search: recog.AdvancedTreeSearchJob
    search_crp: rasr.RasrConfig
    search_feature_scorer: rasr.FeatureScorer
    search_stats: typing.Optional[ExtractSearchStatisticsJob]


def get_feature_scorer(
    context_type: PhoneticContext,
    label_info: LabelInfo,
    feature_scorer_config,
    mixtures,
    silence_id: int,
    prior_info: typing.Union[PriorInfo, tk.Variable, DelayedBase],
    num_label_contexts: int,
    num_states_per_phone: int,
    num_encoder_output: int,
    loop_scale: float,
    forward_scale: float,
    silence_loop_penalty: float,
    silence_forward_penalty: float,
    posterior_scales: typing.Optional[PosteriorScales] = None,
    use_estimated_tdps=False,
    state_dependent_tdp_file: typing.Optional[typing.Union[str, tk.Path]] = None,
    is_min_duration=False,
    is_multi_encoder_output=False,
    set_is_batch_major=False,
):
    if isinstance(prior_info, PriorInfo):
        assert prior_info.center_state_prior is not None and prior_info.center_state_prior.file is not None
        if prior_info.center_state_prior.scale is None:
            print(f"center state prior scale is unset, bug")
        if context_type.is_diphone():
            assert prior_info.left_context_prior is not None and prior_info.left_context_prior.file is not None
            if prior_info.left_context_prior.scale is None:
                print(f"left context prior scale is unset, bug")
        if context_type.is_triphone():
            assert prior_info.right_context_prior is not None and prior_info.right_context_prior.file is not None
            if prior_info.right_context_prior.scale is None:
                print(f"right context prior scale is unset, bug")

    return FactoredHybridFeatureScorer(
        feature_scorer_config,
        prior_mixtures=mixtures,
        context_type=context_type.value,
        prior_info=prior_info,
        num_states_per_phone=num_states_per_phone,
        num_label_contexts=num_label_contexts,
        silence_id=silence_id,
        num_encoder_output=num_encoder_output,
        posterior_scales=posterior_scales,
        is_multi_encoder_output=is_multi_encoder_output,
        loop_scale=loop_scale,
        forward_scale=forward_scale,
        silence_loop_penalty=silence_loop_penalty,
        silence_forward_penalty=silence_forward_penalty,
        use_estimated_tdps=use_estimated_tdps,
        state_dependent_tdp_file=state_dependent_tdp_file,
        is_min_duration=is_min_duration,
        use_word_end_classes=label_info.phoneme_state_classes.use_word_end(),
        use_boundary_classes=label_info.phoneme_state_classes.use_boundary(),
        set_is_batch_major=set_is_batch_major,
    )


def round2(num: float):
    return round(num, 2)


class FHDecoder:
    def __init__(
        self,
        name: str,
        search_crp,
        context_type: PhoneticContext,
        feature_path: typing.Union[str, tk.Path],
        model_path: typing.Union[str, tk.Path],
        graph: typing.Union[str, tk.Path],
        mixtures,
        eval_files,
        tf_library: typing.Optional[typing.Union[str, tk.Path]] = None,
        gpu=True,
        tensor_map: typing.Optional[typing.Union[dict, DecodingTensorMap]] = None,
        is_multi_encoder_output=False,
        silence_id=40,
        recompile_graph_for_feature_scorer=False,
        in_graph_acoustic_scoring=False,
        corpus_duration: typing.Optional[float] = 5.12,  # dev-other
        set_batch_major_for_feature_scorer=False,
        lm_gc_simple_hash=False,
    ):
        assert not (recompile_graph_for_feature_scorer and in_graph_acoustic_scoring)

        self.name = name
        self.search_crp = copy.deepcopy(search_crp)  # s.crp["dev_magic"]
        self.context_type = context_type  # PhoneticContext.value
        self.model_path = model_path
        self.graph = graph
        self.mixtures = mixtures
        self.is_multi_encoder_output = is_multi_encoder_output
        self.tdp = {}
        self.silence_id = silence_id
        self.corpus_duration = corpus_duration
        self.set_batch_major = set_batch_major_for_feature_scorer
        self.lm_gc_simple_hash = lm_gc_simple_hash

        self.parallel = None

        self.tensor_map = (
            dataclasses.replace(DecodingTensorMap.default(), **tensor_map)
            if isinstance(tensor_map, dict)
            else tensor_map
            if isinstance(tensor_map, DecodingTensorMap)
            else DecodingTensorMap.default()
        )

        self.eval_files = eval_files  # ctm file as ref

        self.bellman_post_config = False
        self.gpu = gpu
        self.library_path = DelayedJoin(tf_library, ":") if isinstance(tf_library, list) else tf_library
        self.recompile_graph_for_feature_scorer = recompile_graph_for_feature_scorer
        self.in_graph_acoustic_scoring = in_graph_acoustic_scoring

        # LM attributes
        self.tfrnn_lms = {}

        # setting other attributes
        self.set_tf_fs_flow(feature_path, model_path, graph)
        self.set_fs_tf_config()

    def get_search_params(
        self,
        beam: float,
        beam_limit: int,
        we_pruning=0.5,
        we_pruning_limit=10000,
        lm_state_pruning=None,
        is_count_based=False,
    ):
        sp = {
            "beam-pruning": beam,
            "beam-pruning-limit": beam_limit,
            "word-end-pruning": we_pruning,
            "word-end-pruning-limit": we_pruning_limit,
        }
        if is_count_based:
            return sp
        if lm_state_pruning is not None:
            sp["lm-state-pruning"] = lm_state_pruning
        return sp

    def get_requirements(self, beam: float, nn_lm=False):
        # under 27 is short queue
        rtf = 15

        if not self.gpu:
            rtf *= 4

        if self.context_type not in [
            PhoneticContext.monophone,
            PhoneticContext.diphone,
        ]:
            rtf += 5

        if beam > 17:
            rtf += 10

        if nn_lm:
            rtf += 20
            mem = 12.0
            if "eval" in self.name:
                rtf *= 2
        else:
            mem = 4

        return {"rtf": rtf, "mem": mem}

    def get_lookahead_options(scale=1.0, hlimit=-1, clow=0, chigh=500):
        lmla_options = {
            "scale": scale,
            "history_limit": hlimit,
            "cache_low": clow,
            "cache_high": chigh,
        }
        return lmla_options

    def set_tf_fs_flow(self, feature_path, model_path, graph):
        tf_feature_flow = rasr.FlowNetwork()
        base_mapping = tf_feature_flow.add_net(feature_path)
        if self.is_multi_encoder_output:
            tf_flow = self.get_tf_flow_delta(model_path, graph)
        else:
            tf_flow = self.get_tf_flow(model_path, graph)

        tf_mapping = tf_feature_flow.add_net(tf_flow)

        tf_feature_flow.interconnect_inputs(feature_path, base_mapping)
        tf_feature_flow.interconnect(
            feature_path,
            base_mapping,
            tf_flow,
            tf_mapping,
            {"features": "input-features"},
        )
        tf_feature_flow.interconnect_outputs(tf_flow, tf_mapping)

        self.featureScorerFlow = tf_feature_flow

    def get_tf_flow(self, model_path, graph):
        tf_flow = rasr.FlowNetwork()
        tf_flow.add_input("input-features")
        tf_flow.add_output("features")
        tf_flow.add_param("id")
        tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tf_flow.link("network:input-features", tf_fwd + ":features")
        tf_flow.link(tf_fwd + ":encoder-output", "network:features")

        tf_flow.config = rasr.RasrConfig()

        tf_flow.config[tf_fwd].input_map.info_0.param_name = "features"
        tf_flow.config[tf_fwd].input_map.info_0.tensor_name = self.tensor_map.in_data
        tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = self.tensor_map.in_seq_length

        tf_flow.config[tf_fwd].output_map.info_0.param_name = "encoder-output"
        tf_flow.config[tf_fwd].output_map.info_0.tensor_name = self.tensor_map.out_encoder_output

        tf_flow.config[tf_fwd].loader.type = "meta"
        tf_flow.config[tf_fwd].loader.meta_graph_file = graph
        tf_flow.config[tf_fwd].loader.saved_model_file = model_path
        tf_flow.config[tf_fwd].loader.required_libraries = self.library_path

        return tf_flow

    def get_tf_flow_delta(self, model_path, graph):
        tf_flow = rasr.FlowNetwork()
        tf_flow.add_input("input-features")
        tf_flow.add_output("features")
        tf_flow.add_param("id")
        tf_fwd = tf_flow.add_node("tensorflow-forward", "tf-fwd", {"id": "$(id)"})
        tf_flow.link("network:input-features", tf_fwd + ":features")

        concat = tf_flow.add_node(
            "generic-vector-f32-concat",
            "concatenation",
            {"check-same-length": True, "timestamp-port": "feature-1"},
        )

        tf_flow.link(tf_fwd + ":encoder-output", "%s:%s" % (concat, "feature-1"))
        tf_flow.link(tf_fwd + ":deltaEncoder-output", "%s:%s" % (concat, "feature-2"))

        tf_flow.link(concat, "network:features")

        tf_flow.config = rasr.RasrConfig()

        tf_flow.config[tf_fwd].input_map.info_0.param_name = "features"
        tf_flow.config[tf_fwd].input_map.info_0.tensor_name = self.tensor_map.in_data
        tf_flow.config[tf_fwd].input_map.info_0.seq_length_tensor_name = self.tensor_map.in_seq_length

        tf_flow.config[tf_fwd].output_map.info_0.param_name = "encoder-output"
        tf_flow.config[tf_fwd].output_map.info_0.tensor_name = self.tensor_map.out_encoder_output

        tf_flow.config[tf_fwd].output_map.info_1.param_name = "deltaEncoder-output"
        tf_flow.config[tf_fwd].output_map.info_1.tensor_name = self.tensor_map.out_delta_encoder_output

        tf_flow.config[tf_fwd].loader.type = "meta"
        tf_flow.config[tf_fwd].loader.meta_graph_file = graph
        tf_flow.config[tf_fwd].loader.saved_model_file = model_path
        tf_flow.config[tf_fwd].loader.required_libraries = self.library_path

        return tf_flow

    def set_fs_tf_config(self):
        fs_tf_config = rasr.RasrConfig()
        fs_tf_config.loader = self.featureScorerFlow.config["tf-fwd"]["loader"]

        # dirty hack for the Rust feature scorer
        if self.recompile_graph_for_feature_scorer:
            fs_tf_config.loader.meta_graph_file = RecompileTfGraphJob(
                meta_graph_file=fs_tf_config.loader.meta_graph_file
            ).out_graph
        elif self.in_graph_acoustic_scoring:
            raise NotImplementedError("in-graph scoring is being reworked")

            fs_tf_config.loader.meta_graph_file = AddInGraphScorerJob(
                graph=fs_tf_config.loader.meta_graph_file,
                n_contexts=42,
                tensor_map=self.tensor_map,
            ).out_graph

        del fs_tf_config.input_map

        # input is the same for each model, since the label embeddings are calculated from the dense label identity
        fs_tf_config.input_map.info_0.param_name = "encoder-output"
        fs_tf_config.input_map.info_0.tensor_name = self.tensor_map.in_encoder_output

        # monophone does not have any context
        if self.context_type != PhoneticContext.monophone:
            fs_tf_config.input_map.info_1.param_name = "dense-classes"
            fs_tf_config.input_map.info_1.tensor_name = self.tensor_map.in_classes

        if self.context_type in [
            PhoneticContext.monophone,
            PhoneticContext.mono_state_transition,
        ]:
            fs_tf_config.output_map.info_0.param_name = "center-state-posteriors"
            fs_tf_config.output_map.info_0.tensor_name = self.tensor_map.out_center_state
            if self.context_type == PhoneticContext.mono_state_transition:
                # add the delta outputs
                fs_tf_config.output_map.info_1.param_name = "delta-posteriors"
                fs_tf_config.output_map.info_1.tensor_name = self.tensor_map.out_delta

        if self.context_type in [
            PhoneticContext.diphone,
            PhoneticContext.diphone_state_transition,
        ]:
            fs_tf_config.output_map.info_0.param_name = "center-state-posteriors"
            fs_tf_config.output_map.info_0.tensor_name = self.tensor_map.out_center_state
            fs_tf_config.output_map.info_1.param_name = "left-context-posteriors"
            fs_tf_config.output_map.info_1.tensor_name = self.tensor_map.out_left_context
            if self.context_type == PhoneticContext.diphone_state_transition:
                fs_tf_config.output_map.info_2.param_name = "delta-posteriors"
                fs_tf_config.output_map.info_2.tensor_name = self.tensor_map.out_delta

        if self.context_type == PhoneticContext.triphone_symmetric:
            fs_tf_config.output_map.info_0.param_name = "center-state-posteriors"
            fs_tf_config.output_map.info_0.tensor_name = self.tensor_map.out_center_state
            fs_tf_config.output_map.info_1.param_name = "left-context-posteriors"
            fs_tf_config.output_map.info_1.tensor_name = self.tensor_map.out_left_context
            fs_tf_config.output_map.info_2.param_name = "right-context-posteriors"
            fs_tf_config.output_map.info_2.tensor_name = self.tensor_map.out_right_context

        if self.context_type in [
            PhoneticContext.triphone_forward,
            PhoneticContext.tri_state_transition,
        ]:
            # outputs
            fs_tf_config.output_map.info_0.param_name = "right-context-posteriors"
            fs_tf_config.output_map.info_0.tensor_name = self.tensor_map.out_right_context
            fs_tf_config.output_map.info_1.param_name = "center-state-posteriors"
            fs_tf_config.output_map.info_1.tensor_name = self.tensor_map.out_center_state
            fs_tf_config.output_map.info_2.param_name = "left-context-posteriors"
            fs_tf_config.output_map.info_2.tensor_name = self.tensor_map.out_left_context

            if self.context_type == PhoneticContext.tri_state_transition:
                fs_tf_config.output_map.info_3.param_name = "delta-posteriors"
                fs_tf_config.output_map.info_3.tensor_name = self.tensor_map.out_delta

        elif self.context_type == PhoneticContext.triphone_backward:
            # outputs
            fs_tf_config.output_map.info_0.param_name = "left-context-posteriors"
            fs_tf_config.output_map.info_0.tensor_name = self.tensor_map.out_left_context
            fs_tf_config.output_map.info_1.param_name = "right-context-posteriors"
            fs_tf_config.output_map.info_1.tensor_name = self.tensor_map.out_right_context
            fs_tf_config.output_map.info_2.param_name = "center-state-posteriors"
            fs_tf_config.output_map.info_2.tensor_name = self.tensor_map.out_center_state

        if self.is_multi_encoder_output:
            if self.context_type in [
                PhoneticContext.monophone,
                PhoneticContext.mono_state_transition,
            ]:
                fs_tf_config.input_map.info_1.param_name = "deltaEncoder-output"
                fs_tf_config.input_map.info_1.tensor_name = self.tensor_map.in_delta_encoder_output
            else:
                fs_tf_config.input_map.info_2.param_name = "deltaEncoder-output"
                fs_tf_config.input_map.info_2.tensor_name = self.tensor_map.in_delta_encoder_output

        self.featureScorerConfig = fs_tf_config

    def getFeatureFlow(self, feature_path, tf_flow):
        tf_feature_flow = rasr.FlowNetwork()
        base_mapping = tf_feature_flow.add_net(feature_path)
        tf_mapping = tf_feature_flow.add_net(tf_flow)
        tf_feature_flow.interconnect_inputs(feature_path, base_mapping)
        tf_feature_flow.interconnect(
            feature_path,
            base_mapping,
            tf_flow,
            tf_mapping,
            {"features": "input-features"},
        )
        tf_feature_flow.interconnect_outputs(tf_flow, tf_mapping)

        return tf_feature_flow

    def get_ls_kazuki_lstm_lm_config(
        self,
        scale: typing.Optional[float] = None,
    ) -> rasr.RasrConfig:
        assert self.library_path is not None

        lm_model_dir = tk.Path("/u/mgunz/gunz/dependencies/kazuki_lstmlm_20190627")

        return TfRnnLmRasrConfig(
            vocab_path=lm_model_dir.join_right("vocabulary"),
            meta_graph_path=lm_model_dir.join_right("network.040.meta"),
            returnn_checkpoint=returnn.Checkpoint(lm_model_dir.join_right("network.040.index")),
            scale=scale,
            libraries=self.library_path,
            state_manager="lstm",
        ).get()

    def get_ls_eugen_trafo_config(
        self,
        min_batch_size: int = 0,
        opt_batch_size: int = 64,
        max_batch_size: int = 64,
        scale: typing.Optional[float] = None,
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
        name_override: typing.Union[str, None] = None,
        name_prefix: str = "",
        gpu: typing.Optional[bool] = None,
        cpu_rqmt: typing.Optional[int] = None,
        mem_rqmt: typing.Optional[int] = None,
        crp_update: typing.Optional[typing.Callable[[rasr.RasrConfig], typing.Any]] = None,
        pre_path: str = "decoding",
        rtf_cpu: float = 16,
        rtf_gpu: float = 4,
        lm_config: rasr.RasrConfig = None,
        create_lattice: bool = True,
        remove_or_set_concurrency: typing.Union[bool, int] = False,
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
        )

    def recognize_optimize_scales(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        prior_scales: typing.Union[
            typing.List[typing.Tuple[float]],  # center
            typing.List[typing.Tuple[float, float]],  # center, left
            typing.List[typing.Tuple[float, float, float]],  # center, left, right
            np.ndarray,
        ],
        tdp_scales: typing.Union[typing.List[float], np.ndarray],
        tdp_sil: typing.Optional[typing.List[typing.Tuple[TDP, TDP, TDP, TDP]]] = None,
        tdp_speech: typing.Optional[typing.List[typing.Tuple[TDP, TDP, TDP, TDP]]] = None,
        altas_value=14.0,
        altas_beam=14.0,
        keep_value=10,
        gpu: typing.Optional[bool] = None,
        cpu_rqmt: typing.Optional[int] = None,
        mem_rqmt: typing.Optional[int] = None,
        crp_update: typing.Optional[typing.Callable[[rasr.RasrConfig], typing.Any]] = None,
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
                ).with_prior_scale(left=l, center=c, right=r),
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
            tk.register_output(f"{pre_name}.err", recog_jobs.sclite.out_num_errors)
            tk.register_output(f"{pre_name}.wer", recog_jobs.sclite.out_wer)

        best_overall_wer = ComputeArgminJob({k: v.sclite.out_wer for k, v in jobs.items()})
        best_overall_n = ComputeArgminJob(jobs_num_e)
        tk.register_output(
            f"scales-best/{self.name}/args",
            best_overall_n.out_argmin,
        )
        tk.register_output(
            f"scales-best/{self.name}/num_err",
            best_overall_n.out_min,
        )
        tk.register_output(
            f"scales-best/{self.name}/wer",
            best_overall_wer.out_min,
        )

        # cannot destructure, need to use indices
        best_priors = best_overall_n.out_argmin[0]
        best_tdp_scale = best_overall_n.out_argmin[1]
        best_tdp_sil = best_overall_n.out_argmin[2]
        best_tdp_sp = best_overall_n.out_argmin[3]

        def push_delayed_tuple(
            argmin: DelayedBase,
        ) -> typing.Tuple[DelayedBase, DelayedBase, DelayedBase, DelayedBase]:
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

        best_left_prior = best_priors[1]
        if self.context_type.is_diphone():
            return base_cfg.with_prior_scale(center=best_center_prior, left=best_left_prior)

        best_right_prior = best_priors[2]
        return base_cfg.with_prior_scale(
            center=best_center_prior,
            left=best_left_prior,
            right=best_right_prior,
        )

    def recognize_ls_lstm_lm(
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
        name_override: typing.Union[str, None] = None,
        name_prefix: str = "",
        gpu: typing.Optional[bool] = None,
        cpu_rqmt: typing.Optional[int] = None,
        mem_rqmt: typing.Optional[int] = None,
        crp_update: typing.Optional[typing.Callable[[rasr.RasrConfig], typing.Any]] = None,
    ) -> RecognitionJobs:
        return None  # buggy

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
            lm_config=self.get_ls_kazuki_lstm_lm_config(),
            name_override=name_override,
            name_prefix=name_prefix,
            num_encoder_output=num_encoder_output,
            only_lm_opt=only_lm_opt,
            opt_lm_am=opt_lm_am,
            pre_path="decoding-kazuki-lstm-20190627-lm",
            rerun_after_opt_lm=rerun_after_opt_lm,
            search_parameters=search_parameters,
            use_estimated_tdps=use_estimated_tdps,
            crp_update=crp_update,
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
        name_override: typing.Union[str, None] = None,
        name_prefix: str = "",
        gpu: typing.Optional[bool] = None,
        cpu_rqmt: typing.Optional[int] = None,
        mem_rqmt: typing.Optional[int] = None,
        crp_update: typing.Optional[typing.Callable[[rasr.RasrConfig], typing.Any]] = None,
        rtf_gpu: typing.Optional[float] = None,
        rtf_cpu: typing.Optional[float] = None,
        create_lattice: bool = True,
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
        )

    def recognize(
        self,
        *,
        label_info: LabelInfo,
        num_encoder_output: int,
        search_parameters: SearchParameters,
        add_sis_alias_and_output=True,
        calculate_stats=False,
        gpu: typing.Optional[bool] = None,
        is_min_duration=False,
        is_nn_lm: bool = False,
        keep_value=12,
        lm_config: typing.Optional[rasr.RasrConfig] = None,
        name_override: typing.Union[str, None] = None,
        name_override_without_name: typing.Union[str, None] = None,
        name_prefix: str = "",
        only_lm_opt=True,
        opt_lm_am=True,
        pre_path: typing.Optional[str] = None,
        rerun_after_opt_lm=False,
        use_estimated_tdps=False,
        mem_rqmt: typing.Optional[int] = None,
        cpu_rqmt: typing.Optional[int] = None,
        crp_update: typing.Optional[typing.Callable[[rasr.RasrConfig], typing.Any]] = None,
        rtf_cpu: typing.Optional[float] = None,
        rtf_gpu: typing.Optional[float] = None,
        create_lattice: bool = True,
        remove_or_set_concurrency: typing.Union[bool, int] = False,
    ) -> RecognitionJobs:
        if (
            isinstance(search_parameters, SearchParameters)
            and isinstance(search_parameters.tdp_speech, tuple)
            and isinstance(search_parameters.tdp_silence, tuple)
        ):
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
            name += f"Beam{search_parameters.beam}-Lm{search_parameters.lm_scale}"

        search_crp = copy.deepcopy(self.search_crp)

        if search_crp.lexicon_config.normalize_pronunciation:
            model_combination_config = rasr.RasrConfig()
            model_combination_config.pronunciation_scale = search_parameters.pron_scale
            pron_scale = search_parameters.pron_scale
        else:
            model_combination_config = None
            pron_scale = 1.0

        # additional search parameters
        if name_override is None:
            name += f"-Pron{pron_scale}"

        if name_override is None:
            if search_parameters.prior_info.left_context_prior is not None:
                name += f"-prL{search_parameters.prior_info.left_context_prior.scale}"
            if search_parameters.prior_info.center_state_prior is not None:
                name += f"-prC{search_parameters.prior_info.center_state_prior.scale}"
            if search_parameters.prior_info.right_context_prior is not None:
                name += f"-prR{search_parameters.prior_info.right_context_prior.scale}"

        loop_scale = forward_scale = 1.0
        sil_fwd_penalty = sil_loop_penalty = 0.0

        if search_parameters.tdp_scale is not None:
            if name_override is None:
                name += f"-tdpScale-{search_parameters.tdp_scale}"
                name += f"-spTdp-{format_tdp(search_parameters.tdp_speech)}"
                name += f"-silTdp-{format_tdp(search_parameters.tdp_silence)}"
                name += f"-tdpNWex-{20.0}"

            if search_parameters.transition_scales is not None:
                loop_scale, forward_scale = search_parameters.transition_scales

                if name_override is None:
                    name += f"-loopScale-{loop_scale}"
                    name += f"-fwdScale-{forward_scale}"

            if search_parameters.silence_penalties is not None:
                sil_loop_penalty, sil_fwd_penalty = search_parameters.silence_penalties

                if name_override is None:
                    name += f"-silLoopP-{sil_loop_penalty}"
                    name += f"-silFwdP-{sil_fwd_penalty}"

            if (
                name_override is None
                and search_parameters.tdp_speech[2] == "infinity"
                and search_parameters.tdp_silence[2] == "infinity"
            ):
                name += "-noSkip"
        else:
            if name_override is None:
                name += "-noTdp"

        if name_override is None:
            if search_parameters.we_pruning > 0.5:
                name += f"-wep{search_parameters.we_pruning}"
            if search_parameters.altas is not None:
                name += f"-ALTAS{search_parameters.altas}"
            if search_parameters.add_all_allophones:
                name += "-allAllos"
            if not create_lattice:
                name += "-noLattice"

        state_tying = search_crp.acoustic_model_config.state_tying.type

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
            nonword_phones=search_parameters.non_word_phonemes,
            tying_type="global-and-nonword",
        )

        search_crp.acoustic_model_config.allophones.add_all = search_parameters.add_all_allophones
        search_crp.acoustic_model_config.allophones.add_from_lexicon = not search_parameters.add_all_allophones

        search_crp.acoustic_model_config.state_tying.use_boundary_classes = (
            label_info.phoneme_state_classes.use_boundary()
        )
        search_crp.acoustic_model_config.state_tying.use_word_end_classes = (
            label_info.phoneme_state_classes.use_word_end()
        )

        # lm config update
        if lm_config is not None:
            search_crp.language_model_config = lm_config
        search_crp.language_model_config.scale = search_parameters.lm_scale

        rqms = self.get_requirements(beam=search_parameters.beam, nn_lm=is_nn_lm)
        sp = self.get_search_params(
            search_parameters.beam,
            search_parameters.beam_limit,
            search_parameters.we_pruning,
            search_parameters.we_pruning_limit,
            is_count_based=True,
        )

        if search_parameters.altas is not None:
            adv_search_extra_config = rasr.RasrConfig()
            adv_search_extra_config.flf_lattice_tool.network.recognizer.recognizer.acoustic_lookahead_temporal_approximation_scale = (
                search_parameters.altas
            )
        else:
            adv_search_extra_config = None

        feature_scorer = get_feature_scorer(
            context_type=self.context_type,
            label_info=label_info,
            feature_scorer_config=self.featureScorerConfig,
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
            set_is_batch_major=self.set_batch_major,
        )
        self.feature_scorer = feature_scorer

        pre_path = (
            pre_path
            if pre_path is not None and len(pre_path) > 0
            else "decoding-gridsearch"
            if search_parameters.altas is not None
            else "decoding"
        )

        if crp_update is not None:
            crp_update(search_crp)

        use_gpu = gpu if gpu is not None else self.gpu

        ts_args = {}
        if self.parallel is not None:
            ts_args["parallel"] = self.parallel

        flow = self.featureScorerFlow

        if (isinstance(remove_or_set_concurrency, bool) and remove_or_set_concurrency == True) or (
            isinstance(remove_or_set_concurrency, int) and remove_or_set_concurrency < search_crp.concurrent
        ):
            concurrent = max(int(remove_or_set_concurrency), 1)
            search_crp.concurrent = concurrent

            flow = copy.deepcopy(flow)
            flow.flags["cache_mode"] = "bundle"

            search_crp.segment_path = i6_core.corpus.SegmentCorpusJob(
                search_crp.corpus_config.file, concurrent
            ).out_segment_path

        search = recog.AdvancedTreeSearchJob(
            crp=search_crp,
            feature_flow=self.featureScorerFlow,
            feature_scorer=feature_scorer,
            search_parameters=sp,
            lm_lookahead=True,
            eval_best_in_lattice=True,
            use_gpu=use_gpu,
            rtf=rtf_gpu if rtf_gpu is not None and use_gpu else rtf_cpu if rtf_cpu is not None else rqms["rtf"],
            mem=rqms["mem"] if mem_rqmt is None else mem_rqmt,
            cpu=2 if cpu_rqmt is None else cpu_rqmt,
            model_combination_config=model_combination_config,
            model_combination_post_config=None,
            extra_config=adv_search_extra_config,
            extra_post_config=None,
            lmgc_scorer=rasr.DiagonalMaximumScorer(self.mixtures) if self.lm_gc_simple_hash else None,
            create_lattice=create_lattice,
            separate_lm_image_gc_generation=True,
            **ts_args,
        )

        if add_sis_alias_and_output:
            search.add_alias(f"{pre_path}/{name}")

        if calculate_stats:
            stat = ExtractSearchStatisticsJob(list(search.out_log_file.values()), self.corpus_duration)

            if add_sis_alias_and_output:
                pre = f"{pre_path}-" if pre_path != "decoding" and pre_path != "decoding-gridsearch" else ""
                stat.add_alias(f"{pre}statistics/{name}")
                tk.register_output(f"{pre}statistics/rtf/{name}.rtf", stat.decoding_rtf)
        else:
            stat = None

        if keep_value is not None:
            search.keep_value(keep_value)

        if not create_lattice:
            return RecognitionJobs(
                lat2ctm=None,
                sclite=None,
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
            best_path_algo="bellman-ford",
            extra_config=lat2ctm_extra_config,
            fill_empty_segments=True,
        )

        s_kwrgs = copy.copy(self.eval_files)
        s_kwrgs["hyp"] = lat2ctm.out_ctm_file
        scorer = recog.ScliteJob(**s_kwrgs)

        if add_sis_alias_and_output:
            tk.register_output(f"{pre_path}/{name}.wer", scorer.out_report_dir)

        if opt_lm_am and search_parameters.altas is None:
            assert search_parameters.beam >= 15.0

            opt = recog.OptimizeAMandLMScaleJob(
                crp=search_crp,
                lattice_cache=search.out_lattice_bundle,
                initial_am_scale=pron_scale,
                initial_lm_scale=search_parameters.lm_scale,
                scorer_cls=recog.ScliteJob,
                scorer_kwargs=s_kwrgs,
                opt_only_lm_scale=only_lm_opt,
            )

            if add_sis_alias_and_output:
                tk.register_output(
                    f"{pre_path}/{name}/onlyLmOpt{only_lm_opt}.optlm.txt",
                    opt.out_log_file,
                )

            if rerun_after_opt_lm:
                rounded = Delayed(opt.out_best_lm_score).function(round2)
                params = search_parameters.with_lm_scale(rounded)

                name_prefix_len = len(f"{name_prefix}{self.name}/")

                return self.recognize(
                    add_sis_alias_and_output=add_sis_alias_and_output,
                    calculate_stats=calculate_stats,
                    is_min_duration=is_min_duration,
                    is_nn_lm=is_nn_lm,
                    keep_value=keep_value,
                    label_info=label_info,
                    lm_config=lm_config,
                    name_override=f"{name[name_prefix_len:]}-optlm",
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

    def align(
        self,
        name: str,
        crp: typing.Optional[rasr.CommonRasrParameters] = None,
        feature_scorer: typing.Optional[rasr.FeatureScorer] = None,
        feature_flow: typing.Optional[rasr.FlowNetwork] = None,
        default_tdp=True,
        set_do_not_normalize_lemma_sequence_scores: bool = True,
        set_no_tying_dense: bool = True,
        rtf=4,
    ):
        align_crp = copy.deepcopy(crp) if crp is not None else self.search_crp

        if default_tdp:
            v = (3.0, 0.0, "infinity", 0.0)
            sv = (0.0, 3.0, "infinity", 3.0)
            keys = ["loop", "forward", "skip", "exit"]
            for i, k in enumerate(keys):
                align_crp.acoustic_model_config.tdp["*"][k] = v[i]
                align_crp.acoustic_model_config.tdp["silence"][k] = sv[i]

        if set_no_tying_dense:
            # make sure it is correct for the fh feature scorer scorer
            align_crp.acoustic_model_config.state_tying.type = "no-tying-dense"

        # make sure the FSA is not buggy
        align_crp.acoustic_model_config["*"]["fix-allophone-context-at-word-boundaries"] = True
        align_crp.acoustic_model_config["*"]["transducer-builder-filter-out-invalid-allophones"] = True
        align_crp.acoustic_model_config["*"]["allow-for-silence-repetitions"] = False
        align_crp.acoustic_model_config["*"]["fix-tdp-leaving-epsilon-arc"] = True
        if set_do_not_normalize_lemma_sequence_scores:
            align_crp.acoustic_model_config["*"]["normalize-lemma-sequence-scores"] = False

        alignment = mm.AlignmentJob(
            crp=align_crp,
            feature_flow=feature_flow if feature_flow is not None else self.featureScorerFlow,
            feature_scorer=feature_scorer if feature_scorer is not None else self.feature_scorer,
            use_gpu=self.gpu,
            rtf=rtf,
        )
        alignment.update_rqmt("run", {"mem": 4})

        alignment.add_alias(f"alignments/{name}")
        tk.register_output(f"alignments/{name}", alignment.out_alignment_bundle)

        return alignment
