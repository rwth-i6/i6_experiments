import copy
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

from i6_core.mm import CreateDummyMixturesJob
from i6_core.rasr import OnnxFeatureScorer

from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

from i6_experiments.users.hilmes.tools.onnx import ExportPyTorchModelToOnnxJob


class PyTorchOnnxHybridSystem(HybridSystem):
    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, Union[returnn.Checkpoint, returnn.PtCheckpoint]],
        train_job: Union[
            returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob
        ],  # TODO maybe Optional if prior file provided -> automatically construct dummy file
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
        needs_features_size=True,
        acoustic_mixture_path: Optional[tk.Path] = None,
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

                onnx_job = ExportPyTorchModelToOnnxJob(
                    pytorch_checkpoint=checkpoints[epoch],
                    returnn_config=returnn_config,
                    returnn_root=self.returnn_root,
                    quantize_dynamic=quantize_dynamic,
                )
                onnx_job.add_alias("export_onnx/" + name + "/epoch_" + str(epoch))
                onnx_model = onnx_job.out_onnx_model

                io_map = {"features": "data", "output": "classes"}
                if needs_features_size:
                    io_map["features-size"] = "data_len"

                from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.torch_baselines.pytorch_networks.prior.forward import (
                    ReturnnForwardComputePriorJob,
                )

                acoustic_mixture_path = CreateDummyMixturesJob(
                    num_mixtures=returnn_config.config["extern_data"]["classes"]["dim"],
                    num_features=returnn_config.config["extern_data"]["data"]["dim"],
                ).out_mixtures
                lmgc_scorer = rasr.GMMFeatureScorer(acoustic_mixture_path)
                prior_config = copy.deepcopy(returnn_config)
                assert len(self.train_cv_pairing) == 1, "multiple train corpora not supported yet"
                train_data = self.train_input_data[self.train_cv_pairing[0][0]]
                prior_config.config["train"] = (
                    copy.deepcopy(train_data)
                    if isinstance(train_data, Dict)
                    else copy.deepcopy(train_data.get_data_dict())
                )
                # prior_config.config["train"]["datasets"]["align"]["partition_epoch"] = 3
                prior_config.config["train"]["datasets"]["align"]["seq_ordering"] = "random"
                prior_config.config["forward_batch_size"] = 10000
                if "chunking" in prior_config.config.keys():
                    del prior_config.config["chunking"]
                from i6_core.tools.git import CloneGitRepositoryJob

                returnn_root = CloneGitRepositoryJob(
                    "https://github.com/rwth-i6/returnn",
                    commit="925e0023c52db071ecddabb8f7c2d5a88be5e0ec",
                ).out_repository
                # prior_config.config["max_seqs"] = 5
                nn_prior_job = ReturnnForwardComputePriorJob(
                    model_checkpoint=checkpoints[epoch],
                    returnn_config=prior_config,
                    returnn_python_exe=self.returnn_python_exe,
                    returnn_root=returnn_root,
                    log_verbosity=train_job.returnn_config.post_config["log_verbosity"],
                )
                nn_prior_job.rqmt["gpu_mem"] = 22
                nn_prior_job.add_alias("extract_nn_prior/" + name + "/epoch_" + str(epoch))
                prior_file = nn_prior_job.out_prior_xml_file

                scorer = OnnxFeatureScorer(
                    mixtures=acoustic_mixture_path,
                    model=onnx_model,
                    priori_scale=prior,
                    io_map=io_map,
                    inter_op_threads=kwargs.get("cpu", 1),
                    intra_op_threads=kwargs.get("cpu", 1),
                    prior_file=prior_file,
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
                    lmgc_alias=f"lmgc/{name}/{recognition_corpus_key}-{recog_name}",
                    lmgc_scorer=lmgc_scorer,
                    **kwargs,
                )
