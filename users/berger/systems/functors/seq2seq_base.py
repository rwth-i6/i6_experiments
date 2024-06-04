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

    def _make_tf_model_loader_config(self, tf_graph: tk.Path, tf_checkpoint: returnn.Checkpoint) -> rasr.RasrConfig:
        loader_config = rasr.RasrConfig()
        loader_config.type = "meta"
        loader_config.meta_graph_file = tf_graph
        loader_config.saved_model_file = tf_checkpoint
        loader_config.required_libraries = self._get_native_lstm_op()
        return loader_config

    def _make_onnx_enc_dec_config_for_label_scorer(
        self,
        label_scorer: custom_rasr.LabelScorer,
        enc_onnx_model: tk.Path,
        dec_onnx_model: tk.Path,
        enc_features_name: str = "sources",
        enc_features_size: str = "sources:size1",
        enc_output_name: str = "source_encodings",
        dec_features_name: str = "source_encodings",
        dec_history_name: str = "targets",
        dec_output_name: str = "log_probs",
    ) -> None:
        encoder_io_map = rasr.RasrConfig()
        encoder_io_map.features = enc_features_name
        encoder_io_map.features_size = enc_features_size
        encoder_io_map.encoder_output = enc_output_name

        encoder_session = rasr.RasrConfig()
        encoder_session.file = enc_onnx_model
        encoder_session.inter_op_num_threads = 2
        encoder_session.intra_op_num_threads = 2

        decoder_io_map = rasr.RasrConfig()
        decoder_io_map.encoder_output = dec_features_name
        decoder_io_map.feedback = dec_history_name
        decoder_io_map.output = dec_output_name

        decoder_session = rasr.RasrConfig()
        decoder_session.file = dec_onnx_model
        decoder_session.inter_op_num_threads = 2
        decoder_session.intra_op_num_threads = 2

        label_scorer.config.encoder_io_map = encoder_io_map
        label_scorer.config.encoder_session = encoder_session
        label_scorer.config.decoder_io_map = decoder_io_map
        label_scorer.config.decoder_session = decoder_session

    def _get_tf_feature_flow_for_label_scorer(
        self,
        label_scorer: custom_rasr.LabelScorer,
        base_feature_flow: rasr.FlowNetwork,
        tf_graph: tk.Path,
        checkpoint: returnn.Checkpoint,
        feature_type: FeatureType = FeatureType.SAMPLES,
        output_layer_name: str = "output",
        **_,
    ) -> rasr.FlowNetwork:
        if label_scorer.scorer_type == "precomputed-log-posterior":
            feature_flow = self._make_precomputed_tf_feature_flow(
                base_flow=base_feature_flow,
                tf_graph=tf_graph,
                tf_checkpoint=checkpoint,
                output_layer_name=output_layer_name,
            )
        elif label_scorer.scorer_type in ["tf-attention", "tf-rnn-transducer", "tf-ffnn-transducer", "tf-segmental"]:
            feature_flow = copy.deepcopy(base_feature_flow)
            feature_flow.config = feature_flow.config or rasr.RasrConfig()
            feature_flow.config.main_port_name = "samples" if feature_type == FeatureType.SAMPLES else "features"
            label_scorer.set_input_config()
            label_scorer.set_loader_config(self._make_tf_model_loader_config(tf_graph, checkpoint))
        else:
            raise NotImplementedError

        return feature_flow

    def _get_onnx_feature_flow_for_label_scorer(
        self,
        label_scorer: custom_rasr.LabelScorer,
        base_feature_flow: rasr.FlowNetwork,
        feature_type: FeatureType = FeatureType.SAMPLES,
        **kwargs,
    ) -> rasr.FlowNetwork:
        if label_scorer.scorer_type == "precomputed-log-posterior":
            feature_flow = self._make_precomputed_onnx_feature_flow(base_feature_flow, **kwargs)
        elif label_scorer.scorer_type in ["onnx-ffnn-transducer"]:
            feature_flow = copy.deepcopy(base_feature_flow)
            feature_flow.config = feature_flow.config or rasr.RasrConfig()
            feature_flow.config.main_port_name = "samples" if feature_type == FeatureType.SAMPLES else "features"
            self._make_onnx_enc_dec_config_for_label_scorer(label_scorer, **kwargs)
        else:
            raise NotImplementedError

        return feature_flow
