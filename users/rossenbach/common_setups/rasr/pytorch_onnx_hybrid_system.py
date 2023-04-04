import itertools
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.returnn.flow import (
    make_precomputed_hybrid_tf_feature_flow,
    add_tf_flow_to_base_flow,
)


# -------------------- Init --------------------

Path = tk.setup_path(__package__)

from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

from i6_experiments.users.rossenbach.returnn.onnx import ExportPyTorchModelToOnnxJob


class OnnxFeatureScorer(rasr.FeatureScorer):
    def __init__(
        self,
        mixtures,
        model,
        io_map,
        *args,
        scale=1.0,
        priori_scale=0.7,
        prior_file=None,
        intra_op_threads=1,
        inter_op_threads=1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.config.feature_scorer_type = "onnx-feature-scorer"
        self.config.file = mixtures
        self.config.scale = scale
        self.config.priori_scale = priori_scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        self.config.normalize_mixture_weights = False

        self.config.session.file = model
        self.config.session.intra_op_num_threads = intra_op_threads
        self.config.session.inter_op_num_threads = inter_op_threads

        for k, v in io_map.items():
            self.config.io_map[k] = v


class PyTorchOnnxHybridSystem(HybridSystem):
    
    
    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, returnn.Checkpoint],
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
        **kwargs,
    ):
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
                    returnn_root=self.returnn_root
                ).out_onnx_model

                scorer = OnnxFeatureScorer(
                    mixtures=acoustic_mixture_path,
                    model=onnx_model,
                    priori_scale=prior,
                    io_map={
                        "features": "data",
                        "features-size": "data_len",
                        "output": "classes"
                    }
                )

                self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}"] = scorer
                self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-tf-{epoch:03d}"] = feature_flow

                recog_name = f"e{epoch:03d}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}"
                recog_func(
                    name=f"{name}-{recognition_corpus_key}-{recog_name}",
                    prefix=f"nn_recog/{name}/",
                    corpus=recognition_corpus_key,
                    flow=feature_flow,
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