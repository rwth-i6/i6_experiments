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
from .config.am_config import AmRasrConfig
from .config.lex_config import LexiconRasrConfig
from .config.lm_config import (
    ArpaLmRasrConfig,
    TfRnnLmRasrConfig,
    SimpleTfNeuralLmRasrConfig,
    CombineLmRasrConfig,
)
from .util.decode import (
    RecognitionParameters,
    SearchJobArgs,
    Lattice2CtmArgs,
    ScliteScorerArgs,
    OptimizeJobArgs,
    PriorPath,
)

LmConfig = Union[
    ArpaLmRasrConfig,
    TfRnnLmRasrConfig,
    SimpleTfNeuralLmRasrConfig,
    CombineLmRasrConfig,
]


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
        self.required_native_ops = (
            required_native_ops if required_native_ops is not None else []
        )
        self.native_ops = []

    def init_decoder(
        self,
        *,
        acoustic_model_config: AmRasrConfig,
        lexicon_config: LexiconRasrConfig,
        extra_configs: Optional[Dict[str, rasr.RasrConfig]] = None,
        crp_name: str = "base",
    ):
        self.init_base_crp(
            acoustic_model_config=acoustic_model_config.get(),
            lexicon_config=lexicon_config.get(),
            extra_configs=extra_configs if extra_configs is not None else None,
            crp_name=crp_name,
        )

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

    def recognition(
        self,
        name: str,
        *,
        returnn_config: Union[returnn.ReturnnConfig, tk.Path],
        checkpoints: Dict[int, Union[returnn.Checkpoint, tk.Path]],
        recognition_parameters: RecognitionParameters,
        lm_configs: Dict[str, LmConfig],
        prior_paths: Dict[str, PriorPath],
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

        for idx, ckpt in checkpoints.items():
            if idx in epochs:
                continue
            for lm_name, lm_conf in lm_configs.items():
                for s_name, p in prior_paths.items():
                    feature_scorer = rasr.PrecomputedHybridFeatureScorer(
                        prior_mixtures=p.acoustic_mixture_path,
                        prior_file=p.prior_xml_path,
                    )

                    tf_flow = make_precomputed_hybrid_tf_feature_flow(
                        tf_graph=am_meta_graph,
                        tf_checkpoint=ckpt,
                        output_layer_name=forward_output_layer,
                        native_ops=self.native_ops,
                        tf_fwd_input_name=tf_fwd_input_name,
                    )

                    for eval_c in self.eval_corpora:
                        feature_tf_flow = add_tf_flow_to_base_flow(
                            base_flow=self.feature_flows[eval_c],
                            tf_flow=tf_flow,
                            tf_fwd_input_name=tf_fwd_input_name,
                        )

                        exp_name = f"{name}_ep{idx:03}_lm-{lm_name}_{s_name}"

                        self.decode(
                            name=exp_name,
                            corpus_key=eval_c,
                            feature_scorer=feature_scorer,
                            feature_flow=feature_tf_flow,
                            recognition_parameters=recognition_parameters,
                            lm_rasr_config=lm_conf.get(),
                            search_job_args=search_job_args,
                            lat_2_ctm_args=lat_2_ctm_args,
                            scorer_args=scorer_args,
                            optimize_parameters=optimize_parameters,
                            scorer_hyp_param_name=scorer_hyp_param_name,
                            optimize_am_lm_scales=optimize_am_lm_scales,
                        )
