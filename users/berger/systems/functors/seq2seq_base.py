from abc import ABC
import copy

from i6_core import lexicon, rasr, returnn
from i6_experiments.users.berger.recipe import rasr as custom_rasr
from i6_experiments.users.berger.util import lru_cache_with_signature
from sisyphus import tk

from .rasr_base import RasrFunctor
from ..dataclasses import FeatureType


class Seq2SeqFunctor(RasrFunctor, ABC):
    def requires_label_file(self, label_unit: str) -> bool:
        return label_unit != "hmm"

    @lru_cache_with_signature
    def _get_label_file(self, crp: rasr.CommonRasrParameters) -> tk.Path:
        state_tying_file = lexicon.DumpStateTyingJob(crp).out_state_tying
        return custom_rasr.GenerateLabelFileFromStateTyingJobV2(
            state_tying_file,
        ).out_label_file

    def _make_model_loader_config(self, tf_graph: tk.Path, tf_checkpoint: returnn.Checkpoint) -> rasr.RasrConfig:
        loader_config = rasr.RasrConfig()
        loader_config.type = "meta"
        loader_config.meta_graph_file = tf_graph
        loader_config.saved_model_file = tf_checkpoint
        loader_config.required_libraries = self._get_native_lstm_op()
        return loader_config

    def _get_tf_feature_flow_for_label_scorer(
        self,
        label_scorer: custom_rasr.LabelScorer,
        base_feature_flow: rasr.FlowNetwork,
        tf_graph: tk.Path,
        checkpoint: returnn.Checkpoint,
        feature_type: FeatureType = FeatureType.SAMPLES
    ) -> rasr.FlowNetwork:
        if custom_rasr.LabelScorer.need_tf_flow(label_scorer.scorer_type):
            feature_flow = self._make_tf_feature_flow(base_feature_flow, tf_graph, checkpoint)
        else:
            feature_flow = copy.deepcopy(base_feature_flow)
            feature_flow.config = feature_flow.config or rasr.RasrConfig()
            feature_flow.config.main_port_name = "samples" if feature_type == FeatureType.SAMPLES else "features"
            label_scorer.set_input_config()
            label_scorer.set_loader_config(self._make_model_loader_config(tf_graph, checkpoint))

        return feature_flow

    def _get_onnx_feature_flow_for_label_scorer(
        self,
        label_scorer: custom_rasr.LabelScorer,
        base_feature_flow: rasr.FlowNetwork,
        onnx_model: tk.Path,
    ) -> rasr.FlowNetwork:
        assert custom_rasr.LabelScorer.need_tf_flow(label_scorer.scorer_type)
        return self._make_onnx_feature_flow(base_feature_flow, onnx_model)
