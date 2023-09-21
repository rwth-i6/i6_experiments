import itertools
from typing import Dict, List, Optional, Tuple, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn
from i6_core.returnn.flow import add_tf_flow_to_base_flow
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.users.rossenbach.returnn.onnx import ExportPyTorchModelToOnnxJob

Path = tk.setup_path(__package__)


def make_precomputed_hybrid_onnx_feature_flow(
    onnx_model: tk.Path,
    io_map: Dict[str, str],
    onnx_fwd_input_name: str = "onnx-fwd-input",
    cpu: int = 1,
) -> rasr.FlowNetwork:
    """
    Create the feature flow for a simple ONNX network that predicts frame-wise outputs, to be used
    in combination with the `nn-precomputed-hybrid` feature-scorer setting in RASR.
    Very similar to `make_precomputed_hybrid_tf_feature_flow()`.

    The resulting flow is a trivial:

        <link from="<onnx_fwd_input_name>" to="onnx-fwd:input"/>
        <node name="onnx-fwd" id="$(id)" filter="onnx-forward"/>
        <link from="onnx-fwd:log-posteriors" to="network:features"/>

    With the config settings:

        [flf-lattice-tool.network.recognizer.feature-extraction.onnx-fwd.io-map]
        features      = data
        features-size = data_len
        output        = classes

        [flf-lattice-tool.network.recognizer.feature-extraction.onnx-fwd.session]
        file                 = <onnx_file>
        inter-op-num-threads = 2
        intra-op-num-threads = 2


    :param onnx_model: usually the output of a ExportPyTorchModelToOnnxJob
    :param io_map:
    :param onnx_fwd_input_name: naming for the onnx network input, usually no need to be changed
    :param cpu: number of CPUs to use
    :return: onnx-forward node flow with output link and related config
    """

    # onnx flow (model scoring done in onnx flow node)
    onnx_flow = rasr.FlowNetwork()
    onnx_flow.add_input(onnx_fwd_input_name)
    onnx_flow.add_output("features")
    onnx_flow.add_param("id")

    onnx_fwd = onnx_flow.add_node("onnx-forward", "onnx-fwd", {"id": "$(id)"})
    onnx_flow.link(f"network:{onnx_fwd_input_name}", onnx_fwd + ":input")
    onnx_flow.link(onnx_fwd + ":log-posteriors", "network:features")

    onnx_flow.config = rasr.RasrConfig()
    for k, v in io_map.items():
        onnx_flow.config[onnx_fwd].io_map[k] = v

    onnx_flow.config[onnx_fwd].session.file = onnx_model
    onnx_flow.config[onnx_fwd].session.inter_op_num_threads = cpu
    onnx_flow.config[onnx_fwd].session.intra_op_num_threads = cpu

    return onnx_flow


class OnnxPrecomputedHybridSystem(HybridSystem):
    """
    System class for hybrid systems that train PyTorch models and export them to onnx for recognition. The NN
    precomputed hybrid feature scorer is used.
    """

    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, returnn.PtCheckpoint],
        acoustic_mixture_path: tk.Path,  # TODO maybe Optional if prior file provided -> automatically construct dummy file
        prior_scales: List[float],
        pronunciation_scales: List[float],
        lm_scales: List[float],
        optimize_am_lm_scale: bool,
        recognition_corpus_key: str,
        feature_flow_key: str,
        search_parameters: Dict,
        lattice_to_ctm_kwargs: Dict,
        parallelize_conversion: bool,
        rtf: int,
        mem: int,
        epochs: Optional[List[int]] = None,
        quantize_dynamic: bool = False,
        needs_features_size = True,
        **kwargs,
    ):
        """
        Run recognition with onnx export and precomputed hybrid feature scorer.
        """
        with tk.block(f"{name}_recognition"):
            recog_func = self.recog_and_optimize if optimize_am_lm_scale else self.recog

            feature_flow = self.feature_flows[recognition_corpus_key]
            if isinstance(feature_flow, Dict):
                feature_flow = feature_flow[feature_flow_key]
            assert isinstance(
                feature_flow, rasr.FlowNetwork
            ), f"type incorrect: {recognition_corpus_key} {type(feature_flow)}"

            epochs = epochs if epochs is not None else list(checkpoints.keys())

            for pron, lm, prior, epoch in itertools.product(pronunciation_scales, lm_scales, prior_scales, epochs):
                assert epoch in checkpoints.keys()
                assert acoustic_mixture_path is not None

                onnx_model = ExportPyTorchModelToOnnxJob(
                    pytorch_checkpoint=checkpoints[epoch],
                    returnn_config=returnn_config,
                    returnn_root=self.returnn_root,
                    quantize_dynamic=quantize_dynamic,
                ).out_onnx_model

                io_map = {
                    "features": "data",
                    "output": "classes"
                }
                if needs_features_size:
                    io_map["features-size"] = "data_len"
                onnx_flow = make_precomputed_hybrid_onnx_feature_flow(
                    onnx_model=onnx_model, io_map=io_map, cpu=kwargs.get("cpu", 1),
                )
                flow = add_tf_flow_to_base_flow(feature_flow, onnx_flow, tf_fwd_input_name="onnx-fwd-input")

                scorer = rasr.PrecomputedHybridFeatureScorer(
                    prior_mixtures=acoustic_mixture_path,
                    priori_scale=prior,
                )

                self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}"] = scorer
                self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-onnx-{epoch:03d}"] = flow

                recog_name = f"e{epoch:03d}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}"
                recog_func(
                    name=f"{name}-{recognition_corpus_key}-{recog_name}",
                    prefix=f"nn_recog/{name}/",
                    corpus=recognition_corpus_key,
                    flow=flow,
                    feature_scorer=scorer,
                    pronunciation_scale=pron,
                    lm_scale=lm,
                    search_parameters=search_parameters,
                    lattice_to_ctm_kwargs=lattice_to_ctm_kwargs,
                    parallelize_conversion=parallelize_conversion,
                    rtf=rtf,
                    mem=mem,
                    **kwargs,
                )
