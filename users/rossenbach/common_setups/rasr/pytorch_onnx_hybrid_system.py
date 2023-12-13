import copy
import itertools
from typing import Dict, List, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from i6_core.returnn.compile import TorchOnnxExportJob

from i6_experiments.common.setups.rasr.util.nn import HybridArgs

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
        super().__init__()

        self.config.feature_scorer_type = "onnx-feature-scorer"
        self.config.file = mixtures
        self.config.scale = scale
        self.config.priori_scale = priori_scale
        if prior_file is not None:
            self.config.prior_file = prior_file
        self.config.normalize_mixture_weights = False

        self.config.session.file = model

        self.post_config.session.intra_op_num_threads = intra_op_threads
        self.post_config.session.inter_op_num_threads = inter_op_threads

        for k, v in io_map.items():
            self.config.io_map[k] = v


class PyTorchOnnxHybridSystem(HybridSystem):
    
    
    def nn_recognition(
        self,
        name: str,
        returnn_config: ReturnnConfig,
        checkpoints: Dict[int, PtCheckpoint],
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

                onnx_model_job = ExportPyTorchModelToOnnxJob(
                    pytorch_checkpoint=checkpoints[epoch],
                    returnn_config=returnn_config,
                    returnn_root=self.returnn_root,
                    quantize_dynamic=quantize_dynamic,
                )
                onnx_model_job.add_alias(f"nn_recog/{name}/onnx_export_e{epoch:03d}")
                onnx_model = onnx_model_job.out_onnx_model

                io_map = {
                    "features": "data",
                    "output": "classes"
                }
                if needs_features_size:
                    io_map["features-size"] = "data_len"

                scorer = OnnxFeatureScorer(
                    mixtures=acoustic_mixture_path,
                    model=onnx_model,
                    priori_scale=prior,
                    io_map=io_map,
                    inter_op_threads=kwargs.get("cpu", 1),
                    intra_op_threads=kwargs.get("cpu", 1)
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


    def nn_recog(
        self,
        train_name: str,
        train_corpus_key: str,
        returnn_config: Path,
        checkpoints: Dict[int, returnn.PtCheckpoint],
        step_args: HybridArgs,
        train_job: Optional[Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob]] = None,
    ):
        for recog_name, recog_args in step_args.recognition_args.items():
            recog_args = copy.deepcopy(recog_args)
            whitelist = recog_args.pop("training_whitelist", None)
            if whitelist:
                if train_name not in whitelist:
                    continue

            for dev_c in self.dev_corpora:
                self.nn_recognition(
                    name=f"{train_corpus_key}-{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[train_corpus_key].acoustic_mixtures,
                    recognition_corpus_key=dev_c,
                    **recog_args,
                )

            for tst_c in self.test_corpora:
                r_args = copy.deepcopy(recog_args)
                if step_args.test_recognition_args is None or recog_name not in step_args.test_recognition_args.keys():
                    break
                r_args.update(step_args.test_recognition_args[recog_name])
                r_args["optimize_am_lm_scale"] = False
                self.nn_recognition(
                    name=f"{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[train_corpus_key].acoustic_mixtures,
                    recognition_corpus_key=tst_c,
                    **r_args,
                )


class PyTorchOnnxHybridSystemV2(HybridSystem):
    """Use different export job"""

    def nn_recognition(
            self,
            name: str,
            returnn_config: ReturnnConfig,
            checkpoints: Dict[int, PtCheckpoint],
            acoustic_mixture_path: tk.Path,
            # TODO maybe Optional if prior file provided -> automatically construct dummy file
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
            needs_features_size=True,
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

                onnx_model_job = TorchOnnxExportJob(
                    returnn_config=returnn_config,
                    checkpoint=checkpoints[epoch],
                    returnn_root=self.returnn_root,
                )
                onnx_model_job.add_alias(f"nn_recog/{name}/onnx_export_e{epoch:03d}")
                onnx_model = onnx_model_job.out_onnx_model

                io_map = {
                    "features": "data",
                    "output": "classes"
                }
                if needs_features_size:
                    io_map["features-size"] = "data_len"

                scorer = OnnxFeatureScorer(
                    mixtures=acoustic_mixture_path,
                    model=onnx_model,
                    priori_scale=prior,
                    io_map=io_map,
                    inter_op_threads=kwargs.get("cpu", 1),
                    intra_op_threads=kwargs.get("cpu", 1)
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

    def nn_recog(
            self,
            train_name: str,
            train_corpus_key: str,
            returnn_config: Path,
            checkpoints: Dict[int, returnn.PtCheckpoint],
            step_args: HybridArgs,
            train_job: Optional[Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob]] = None,
    ):
        for recog_name, recog_args in step_args.recognition_args.items():
            recog_args = copy.deepcopy(recog_args)
            whitelist = recog_args.pop("training_whitelist", None)
            if whitelist:
                if train_name not in whitelist:
                    continue

            for dev_c in self.dev_corpora:
                self.nn_recognition(
                    name=f"{train_corpus_key}-{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[train_corpus_key].acoustic_mixtures,
                    recognition_corpus_key=dev_c,
                    **recog_args,
                )

            for tst_c in self.test_corpora:
                r_args = copy.deepcopy(recog_args)
                if step_args.test_recognition_args is None or recog_name not in step_args.test_recognition_args.keys():
                    break
                r_args.update(step_args.test_recognition_args[recog_name])
                r_args["optimize_am_lm_scale"] = False
                self.nn_recognition(
                    name=f"{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[train_corpus_key].acoustic_mixtures,
                    recognition_corpus_key=tst_c,
                    **r_args,
                )
                
    def returnn_rasr_training(
        self,
        name,
        returnn_config,
        nn_train_args,
        train_corpus_key,
        cv_corpus_key,
        feature_flow_key: str = "gt",
    ):
        train_job = super().returnn_rasr_training(
            name=name,
            returnn_config=returnn_config,
            nn_train_args=nn_train_args,
            train_corpus_key=train_corpus_key,
            cv_corpus_key=cv_corpus_key,
            feature_flow_key=feature_flow_key
        )

        # train_job.rqmt["gpu"] = "gtx_1080:1"
        return train_job