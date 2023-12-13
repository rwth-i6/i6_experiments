import itertools
import copy
from typing import Dict, List, Optional, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn
from i6_core.mm import CreateDummyMixturesJob
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.torch_baselines.pytorch_networks.prior.forward import \
    ReturnnForwardComputePriorJob
from i6_core.returnn import ReturnnForwardJobV2
from i6_core.returnn import GetBestPtCheckpointJob, TorchOnnxExportJob
from i6_core.returnn.flow import make_precomputed_hybrid_onnx_feature_flow, add_fwd_flow_to_base_flow
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem

Path = tk.setup_path(__package__)


class OnnxPrecomputedHybridSystem(HybridSystem):
    """
    System class for hybrid systems that train PyTorch models and export them to onnx for recognition. The NN
    precomputed hybrid feature scorer is used.
    """

    def calcluate_nn_prior(self, returnn_config, epoch, epoch_num, name, checkpoint):
        prior_config = copy.deepcopy(returnn_config)
        assert len(self.train_cv_pairing) == 1, "multiple train corpora not supported"
        train_data = self.train_input_data[self.train_cv_pairing[0][0]]
        prior_config.config["train"] = copy.deepcopy(train_data) if isinstance(train_data,
                                                                               Dict) else copy.deepcopy(
            train_data.get_data_dict())
        prior_config.config["train"]["datasets"]["align"]["seq_ordering"] = "random"
        prior_config.config["forward_batch_size"] = 10000
        if epoch == "best":
            prior_config.config["load_epoch"] = epoch_num
        if "chunking" in prior_config.config.keys():
            del prior_config.config["chunking"]

        nn_prior_job = ReturnnForwardJobV2(
            model_checkpoint=checkpoint,
            returnn_config=prior_config,
            log_verbosity=5,
            mem_rqmt=4,
            time_rqmt=1,
            device="gpu",
            cpu_rqmt=4,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            output_files=["prior.txt"]
        )
        # nn_prior_job = ReturnnForwardComputePriorJob(
        #     model_checkpoint=checkpoint,
        #     returnn_config=prior_config,
        #     returnn_python_exe=self.returnn_python_exe,
        #     returnn_root=self.returnn_root,
        #     log_verbosity=5,
        # )
        nn_prior_job.add_alias("extract_nn_prior/" + name + "/epoch_" + str(epoch))
        prior_file = nn_prior_job.out_files["prior.txt"]
        return prior_file, prior_config

    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, returnn.PtCheckpoint],
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
        nn_prior: bool,
        epochs: Optional[List[Union[int, str]]] = None,
        train_job: Optional[Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob]] = None,
        needs_features_size: bool = True,
        acoustic_mixture_path: Optional[tk.Path] = None,
        best_checkpoint_key: str = "dev_loss_CE",
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
                if epoch == "best":
                    assert train_job is not None, "train_job needed to get best epoch checkpoint"
                    best_checkpoint_job = GetBestPtCheckpointJob(
                        train_job.out_model_dir, train_job.out_learning_rates, key=best_checkpoint_key, index=0
                    )
                    checkpoint = best_checkpoint_job.out_checkpoint
                    epoch_str = epoch
                    epoch_num = best_checkpoint_job.out_epoch
                else:
                    assert epoch in checkpoints.keys()
                    checkpoint = checkpoints[epoch]
                    epoch_str = f"{epoch:03d}"
                    epoch_num = None

                onnx_job = TorchOnnxExportJob(
                    returnn_config=returnn_config,
                    checkpoint=checkpoint,
                    returnn_root=self.returnn_root,
                    returnn_python_exe=self.returnn_python_exe,
                )
                onnx_job.add_alias(f"export_onnx/{name}/epoch_{epoch_str}")
                onnx_model = onnx_job.out_onnx_model

                io_map = {"features": "data", "output": "classes"}
                if needs_features_size:
                    io_map["features-size"] = "data_len"
                onnx_flow = make_precomputed_hybrid_onnx_feature_flow(
                    onnx_model=onnx_model,
                    io_map=io_map,
                    cpu=kwargs.get("cpu", 1),
                )
                flow = add_fwd_flow_to_base_flow(feature_flow, onnx_flow, fwd_input_name="onnx-fwd-input")

                if nn_prior:
                    prior_file, prior_config = self.calcluate_nn_prior(
                        returnn_config=returnn_config,
                        epoch=epoch,
                        epoch_num=epoch_num,
                        name=name,
                        checkpoint=checkpoint,
                    )
                    # This can't be acoustic_mixture_path because python hands in the object itself, not a copy thus
                    # one would override the old mixture_path (if there is any) for all other exps
                    tmp_acoustic_mixture_path = CreateDummyMixturesJob(
                        num_mixtures=returnn_config.config['extern_data']['classes']['dim'],
                        num_features=returnn_config.config['extern_data']['data']['dim']).out_mixtures
                    lmgc_scorer = rasr.GMMFeatureScorer(tmp_acoustic_mixture_path)
                    scorer = rasr.PrecomputedHybridFeatureScorer(
                        prior_mixtures=tmp_acoustic_mixture_path,
                        priori_scale=prior_file,
                    )
                else:
                    assert acoustic_mixture_path is not None, "need mixtures if no nn prior is computed"
                    scorer = rasr.PrecomputedHybridFeatureScorer(
                        prior_mixtures=acoustic_mixture_path,
                        priori_scale=prior,
                    )
                    lmgc_scorer = rasr.GMMFeatureScorer(acoustic_mixture_path)

                self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}"] = scorer
                self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-onnx-{epoch_str}"] = flow

                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}"
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
                    lmgc_scorer=lmgc_scorer,
                    **kwargs,
                )
