import itertools
import copy
from typing import Dict, List, Optional, Union

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn
from i6_core.mm import CreateDummyMixturesJob
from i6_core.returnn import ReturnnForwardJobV2
from i6_core.returnn import GetBestPtCheckpointJob
from i6_core.returnn.flow import make_precomputed_hybrid_onnx_feature_flow, add_fwd_flow_to_base_flow
from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.common.setups.rasr.util.nn import HybridArgs, ReturnnRasrDataInput
from i6_experiments.users.hilmes.tools.onnx import ExportPyTorchModelToOnnxJob
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.torch_baselines.pytorch_networks.prior.forward import ReturnnForwardComputePriorJob



Path = tk.setup_path(__package__)


class OnnxPrecomputedHybridSystem(HybridSystem):
    """
    System class for hybrid systems that train PyTorch models and export them to onnx for recognition. The NN
    precomputed hybrid feature scorer is used.
    """
    def run_nn_step(self, step_name: str, step_args: HybridArgs):
        for pairing in self.train_cv_pairing:
            trn_c = pairing[0]
            cv_c = pairing[1]
            name_list = [pairing[2]] if len(pairing) >= 3 else list(step_args.returnn_training_configs.keys())
            dvtr_c_list = [pairing[3]] if len(pairing) >= 4 else self.devtrain_corpora
            dvtr_c_list = [None] if len(dvtr_c_list) == 0 else dvtr_c_list

            for name, dvtr_c in itertools.product(name_list, dvtr_c_list):
                if isinstance(self.train_input_data[trn_c], ReturnnRasrDataInput):
                    returnn_train_job = self.returnn_rasr_training(
                        name=name,
                        returnn_config=step_args.returnn_training_configs[name],
                        nn_train_args=step_args.training_args,
                        train_corpus_key=trn_c,
                        cv_corpus_key=cv_c,
                    )
                else:
                    returnn_train_job = self.returnn_training(
                        name=name,
                        returnn_config=step_args.returnn_training_configs[name],
                        nn_train_args=step_args.training_args,
                        train_corpus_key=trn_c,
                        cv_corpus_key=cv_c,
                        devtrain_corpus_key=dvtr_c,
                    )

                returnn_recog_config = step_args.returnn_recognition_configs.get(
                    name, step_args.returnn_training_configs[name]
                )

                self.nn_recog(
                    train_name=name,
                    train_corpus_key=trn_c,
                    returnn_config=returnn_recog_config,
                    checkpoints=returnn_train_job.out_checkpoints,
                    step_args=step_args,
                    train_job=returnn_train_job
                )


    def nn_recog(
        self,
        train_name: str,
        train_corpus_key: str,
        returnn_config: Path,
        checkpoints: Dict[int, returnn.Checkpoint],
        step_args: HybridArgs,
        train_job: Optional[Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob]]
    ):
        for recog_name, recog_args in step_args.recognition_args.items():
            for dev_c in self.dev_corpora:
                self.nn_recognition(
                    name=f"{train_corpus_key}-{train_name}-{recog_name}",
                    returnn_config=returnn_config,
                    checkpoints=checkpoints,
                    acoustic_mixture_path=self.train_input_data[train_corpus_key].acoustic_mixtures,
                    recognition_corpus_key=dev_c,
                    train_job=train_job,
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
                    train_job=train_job,
                    **r_args,
                )

    def calcluate_nn_prior(self, returnn_config, epoch, epoch_num, name, checkpoint):
        prior_config = copy.deepcopy(returnn_config)
        assert len(self.train_cv_pairing) == 1, "multiple train corpora not supported"
        train_data = self.train_input_data[self.train_cv_pairing[0][0]]
        prior_config.config["train"] = copy.deepcopy(train_data) if isinstance(train_data,
                                                                               Dict) else copy.deepcopy(
            train_data.get_data_dict())
        #prior_config.config["train"]["datasets"]["align"]["partition_epoch"] = 100
        prior_config.config["forward_data"] = "train"
        prior_config.config["train"]["datasets"]["align"]["seq_ordering"] = "random"
        if epoch == "best":
            prior_config.config["load_epoch"] = epoch_num
        from i6_core.tools.git import CloneGitRepositoryJob
        # returnn_root = CloneGitRepositoryJob(
        #     "https://github.com/rwth-i6/returnn",
        #     commit="925e0023c52db071ecddabb8f7c2d5a88be5e0ec",
        # ).out_repository
        nn_prior_job = ReturnnForwardJobV2(
            model_checkpoint=checkpoint,
            returnn_config=prior_config,
            log_verbosity=5,
            mem_rqmt=4,
            time_rqmt=4,
            device="gpu",
            cpu_rqmt=4,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=self.returnn_root,
            output_files=["prior.txt", "prior.xml", "prior.png"],
        )
        nn_prior_job.add_alias("extract_nn_prior/" + name + "/epoch_" + str(epoch))
        prior_file = nn_prior_job.out_files["prior.xml"]
        return prior_file, prior_config

    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, returnn.Checkpoint],
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
                if returnn_config.config["behavior_version"] == 15:
                    returnn_config.config["behavior_version"] = 16
                onnx_job = ExportPyTorchModelToOnnxJob(
                    returnn_config=returnn_config,
                    pytorch_checkpoint=checkpoint,
                    returnn_root=self.returnn_root,
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
                flow = add_fwd_flow_to_base_flow(feature_flow, onnx_flow)

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
                        prior_file=prior_file,
                        priori_scale=prior
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
