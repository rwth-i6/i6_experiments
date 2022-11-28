__all__ = ["HybridDecoder"]

from typing import Dict, List, Optional, Type, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.recognition as recog
import i6_core.returnn as returnn

from i6_core.returnn.flow import (
    make_precomputed_hybrid_tf_feature_flow,
    add_tf_flow_to_base_flow,
)

from .base_decoder import BaseDecoder
from .util.decode import (
    RecognitionParameters,
    SearchJobArgs,
    Lattice2CtmArgs,
    ScliteScorerArgs,
    OptimizeJobArgs,
    PriorPath,
)


class HybridDecoder(BaseDecoder):
    def __init__(
        self,
        rasr_binary_path: tk.Path,
        rasr_arch: "str" = "linux-x86_64-standard",
        compress: bool = False,
        append: bool = False,
        unbuffered: bool = False,
        compress_after_run: bool = True,
        search_job_class: Type[tk.Job] = recog.AdvancedTreeSearchJob,
        scorer_job_class: Type[tk.Job] = recog.ScliteJob,
        alias_output_prefix: str = "",
        returnn_root: Optional[tk.Path] = None,
        returnn_python_home: Optional[tk.Path] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        blas_lib: Optional[tk.Path] = None,
        search_numpy_blas: bool = True,
        required_native_ops: Optional[List[str]] = None,
    ):
        """
        Class to perform Hybrid Decoding with a RETURNN model

        param rasr_binary_path: path to the rasr binary folder
        param rasr_arch: RASR compile architecture suffix
        """
        super().__init__(
            rasr_binary_path=rasr_binary_path,
            rasr_arch=rasr_arch,
            compress=compress,
            append=append,
            unbuffered=unbuffered,
            compress_after_run=compress_after_run,
            search_job_class=search_job_class,
            scorer_job_class=scorer_job_class,
            alias_output_prefix=alias_output_prefix,
        )

        self.returnn_root = returnn_root
        self.returnn_python_home = returnn_python_home
        self.returnn_python_exe = returnn_python_exe

        self.blas_lib = blas_lib
        self.search_numpy_blas = search_numpy_blas
        self.required_native_ops = required_native_ops
        self.native_ops = []

    def _compile_necessary_native_ops(self):
        for op in self.required_native_ops:
            native_ops_job = returnn.CompileNativeOpJob(
                native_op=op,
                returnn_python_exe=self.returnn_python_exe,
                returnn_root=self.returnn_root,
                search_numpy_blas=self.search_numpy_blas,
                blas_lib=self.blas_lib,
            )
            self.native_ops.append(native_ops_job.out_op)

    def _compile_tf_graphs(self, returnn_config: returnn.ReturnnConfig):
        graph_job = returnn.CompileTFGraphJob(
            returnn_config=returnn_config,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
        )
        return graph_job.out_graph

    def _get_lm_image_and_global_cache(self):
        pass

    def _get_prior_file(self):
        pass

    def _build_recog_loop(self):
        pass

    def recognition(
        self,
        name: str,
        returnn_config: Union[returnn.ReturnnConfig, tk.Path],
        checkpoints: Dict[int, Union[returnn.Checkpoint, tk.Path]],
        prior_paths: Dict[str, PriorPath],
        feature_flow: rasr.FlowNetwork,
        recognition_parameters: RecognitionParameters,
        lm_rasr_configs: Dict[str, rasr.RasrConfig],
        *,
        search_job_args: Union[SearchJobArgs, Dict],
        lat_2_ctm_args: Union[Lattice2CtmArgs, Dict],
        scorer_args: Union[ScliteScorerArgs, Dict],
        optimize_parameters: Union[OptimizeJobArgs, Dict],
        epochs: Optional[List[int]] = None,
        scorer_hyp_param_name: str = "hyp",
        optimize_am_lm_scales: bool = False,
        forward_output_layer: str = "output",
        tf_fwd_input_name: str = "tf-fwd-input",
    ):
        self._compile_necessary_native_ops()
        am_meta_graph = self._compile_tf_graphs(returnn_config)

        if epochs is None:
            epochs = [checkpoints.keys()]

        len_scorers = len(prior_paths.keys())

        for idx, ckpt in checkpoints.items():
            if idx not in epochs:
                continue
            for lm_name, lm_conf in lm_rasr_configs:
                for s_name, p in prior_paths:
                    tf_flow = make_precomputed_hybrid_tf_feature_flow(
                        tf_graph=am_meta_graph,
                        tf_checkpoint=ckpt,
                        output_layer_name=forward_output_layer,
                        native_ops=self.native_ops,
                        tf_fwd_input_name=tf_fwd_input_name,
                    )
                    feature_tf_flow = add_tf_flow_to_base_flow(
                        base_flow=feature_flow,
                        tf_flow=tf_flow,
                        tf_fwd_input_name=tf_fwd_input_name,
                    )

                    feature_scorer = rasr.PrecomputedHybridFeatureScorer(
                        prior_mixtures=p.acoustic_mixture_path,
                        scale=1.0,
                        priori_scale=p.priori_scale,
                        prior_file=p.prior_xml_path,
                    )

                    exp_name = f"{name}_ep{idx:3f.}_lm{lm_name}"
                    if len_scorers >= 1:
                        exp_name += f"_{s_name}"

                    self.decode(
                        name=exp_name,
                        feature_scorer=feature_scorer,
                        feature_flow=feature_tf_flow,
                        recognition_parameters=recognition_parameters,
                        search_job_args=search_job_args,
                        lat_2_ctm_args=lat_2_ctm_args,
                        scorer_args=scorer_args,
                        optimize_parameters=optimize_parameters,
                        scorer_hyp_param_name=scorer_hyp_param_name,
                        optimize_am_lm_scales=optimize_am_lm_scales,
                    )
