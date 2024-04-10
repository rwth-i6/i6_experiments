import itertools
from typing import Dict, List, Optional, Union
import copy

from sisyphus import tk

import i6_core.rasr as rasr
import i6_core.returnn as returnn
from i6_core.returnn import GetBestPtCheckpointJob, TorchOnnxExportJob, ReturnnForwardJobV2, AverageTorchCheckpointsJob
from i6_core.returnn.flow import make_precomputed_hybrid_onnx_feature_flow, add_fwd_flow_to_base_flow
from i6_core.mm import CreateDummyMixturesJob
from i6_experiments.users.hilmes.common.setups.rasr.hybrid_system import HybridSystem
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType
from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob

Path = tk.setup_path(__package__)


class OnnxPrecomputedHybridSystem(HybridSystem):
    """
    System class for hybrid systems that train PyTorch models and export them to onnx for recognition. The NN
    precomputed hybrid feature scorer is used.
    """
    def calcluate_nn_prior(self, returnn_config, epoch, epoch_num, name, checkpoint, train_job):
        prior_config = copy.deepcopy(returnn_config)
        assert len(self.train_cv_pairing) == 1, "multiple train corpora not supported yet"
        train_data = self.train_input_data[self.train_cv_pairing[0][0]]
        prior_config.config["train"] = copy.deepcopy(train_data) if isinstance(train_data,
                                                                               Dict) else copy.deepcopy(
            train_data.get_data_dict())
        # prior_config.config["train"]["datasets"]["align"]["partition_epoch"] = 3
        if "align" not in prior_config.config["train"]["datasets"]:
            prior_config.config["train"]["datasets"]["ogg"]["seq_ordering"] = "random"
        else:
            prior_config.config["train"]["datasets"]["align"]["seq_ordering"] = "random"
        prior_config.config["forward_batch_size"] = 10000
        #if epoch == "best":
        #    prior_config.config["load_epoch"] = epoch_num
        if "chunking" in prior_config.config.keys():
            del prior_config.config["chunking"]
        if "torch_amp" in prior_config.config.keys():
            del prior_config.config['torch_amp']
        from i6_core.tools.git import CloneGitRepositoryJob
        # if "align" not in prior_config.config["train"]["datasets"]:
        returnn_root = CloneGitRepositoryJob(
            "https://github.com/rwth-i6/returnn",
            commit="d4ab1d8fcbe3baa11f6d8e2cf8e443bc0e9e9fa2",
        ).out_repository.copy()
        returnn_root.hash_overwrite = "RETURNN_PRIOR_COMMIT"
        big_gpu_names = ["whisper_large"] # not including v2large and v2medium for legacy reasons
        if any(x in name for x in big_gpu_names):
            prior_config.config["max_seqs"] = 3
        elif "whisper" in name:
            prior_config.config["max_seqs"] = 1
        nn_prior_job = ReturnnForwardJobV2(
            model_checkpoint=checkpoint,
            returnn_config=prior_config,
            log_verbosity=5,
            mem_rqmt=16 if any(x in name for x in ["whisper", "hubert", "parakeet"]) else 4,
            time_rqmt=2 if not "whisper" in name else 4,
            device="gpu",
            cpu_rqmt=2,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=returnn_root,
            output_files=["prior.txt", "prior.xml"]
        )
        if any(x in name for x in ["whisper_v2_large", "whisper_v2_medium"] + big_gpu_names):
            nn_prior_job.rqmt["gpu_mem"] = 24
        nn_prior_job.add_alias("extract_nn_prior/" + name + "/epoch_" + str(epoch))
        prior_file = nn_prior_job.out_files["prior.xml"]
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
                    best_checkpoint_job.add_alias(f"get_best/{name}")
                    checkpoint = best_checkpoint_job.out_checkpoint
                    epoch_str = epoch
                    #epoch_num = best_checkpoint_job.out_epoch
                    epoch_num = None
                elif epoch == "avrg":
                    assert train_job is not None, "train_job needed to average checkpoints"
                    chkpts = []
                    for x in [0, 1, 2, 3]:
                        best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates,
                                                          key=best_checkpoint_key, index=x)
                        best_job.add_alias(f"get_best/{name}_{x}")
                        chkpts.append(best_job.out_checkpoint)
                        avrg_job = AverageTorchCheckpointsJob(
                            checkpoints=chkpts,
                            returnn_python_exe=self.returnn_python_exe,
                            returnn_root=self.returnn_root
                        )
                        avrg_job.add_alias(f"avrg_chkpt/{name}")
                        checkpoint = avrg_job.out_checkpoint
                        epoch_str = "avrg"
                        epoch_num = None
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
                if any(x in name for x in ["parakeet"]):
                    onnx_job.rqmt["mem"] = 64
                # elif any(x in name for x in ["distill_hubert"]):
                #     onnx_job.rqmt["mem"] = 64
                elif any(x in name for x in ["hubert"]):
                    onnx_job.rqmt["mem"] = 48
                elif any(x in name for x in ["whisper"]):
                    onnx_job.rqmt["mem"] = 48

                onnx_job.add_alias(f"export_onnx/{name}/epoch_{epoch_str}")
                onnx_job.set_keep_value(5)
                onnx_model = onnx_job.out_onnx_model
                io_map = {"features": "data", "output": "log_probs"}
                if needs_features_size:
                    io_map["features-size"] = "data:size1"

                if nn_prior or acoustic_mixture_path is None:
                    prior_file, prior_config = self.calcluate_nn_prior(
                        returnn_config=returnn_config,
                        epoch=epoch,
                        epoch_num=epoch_num,
                        name=name,
                        checkpoint=checkpoint,
                        train_job=train_job
                    )
                    # This can't be acoustic_mixture_path because python hands in the object itself, not a copy thus
                    # one would override the old mixture_path (if there is any) for all other exps
                    tmp_acoustic_mixture_path = CreateDummyMixturesJob(
                        num_mixtures=returnn_config.config['extern_data']['classes']['dim'],
                        num_features=returnn_config.config['extern_data']['data']['dim']).out_mixtures
                    lmgc_scorer = rasr.GMMFeatureScorer(acoustic_mixture_path)
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
                kwargs = copy.deepcopy(kwargs)
                data_num_ls = kwargs.pop("quantize", None)
                quant_modes = [CalibrationMethod.MinMax, CalibrationMethod.Entropy, CalibrationMethod.Percentile]
                average_modes = [True, False]
                sym_modes = [True, False]
                activation_types = [QuantType.QInt8]
                weight_types = [QuantType.QInt8]
                quant_ops_ls = [None, ["Conv"], ["Linear"], ["Conv", "Linear"]]
                quant_formats = [QuantFormat.QDQ, QuantFormat.QOperator]
                if data_num_ls is not None and ((prior == 0.7 and lm == 10.0) or "speed" in name):
                    for data_num, quant_mode, average, sym, activation_type, weight_type, quant_ops, quant_format in itertools.product(quant_modes, average_modes, sym_modes, activation_types, weight_types, quant_ops_ls, quant_formats):
                        if average and (not quant_mode == CalibrationMethod.MinMax or "speed" in name):
                            continue
                        if not quant_mode == CalibrationMethod.MinMax and "speed" in name:
                            continue
                        if quant_mode == CalibrationMethod.MinMax:
                            mode_str = "quant_min_max"
                        elif quant_mode == CalibrationMethod.Entropy:
                            mode_str = "quant_entropy"
                        else:
                            mode_str = "quant_percentile"
                        if average:
                            mode_str += "_avg"
                        if sym:
                            mode_str += "_sym"
                        quant_job = ModelQuantizeStaticJob(
                            model=onnx_model,
                            dataset=prior_config.config["train"]["datasets"]["feat"],
                            num_seqs=data_num,
                            num_parallel_seqs=10,
                            calibrate_method=quant_mode,
                            moving_average=average,
                            symmetric=sym,
                            activation_type=activation_type,
                            weight_type=weight_type,
                            ops_to_quant=quant_ops,
                            quant_format=quant_format,
                        )
                        quant_job.add_alias(
                            "quantize_static/" + name + "/" + mode_str + "/epoch" + epoch_str + "_" + str(data_num))
                        quant_model = quant_job.out_model

                        onnx_flow = make_precomputed_hybrid_onnx_feature_flow(
                            onnx_model=quant_model,
                            io_map=io_map,
                            cpu=kwargs.get("cpu", 1),
                        )
                        flow = add_fwd_flow_to_base_flow(feature_flow, onnx_flow, fwd_input_name="onnx-fwd-input")
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
                            lmgc_alias=f"lmgc/{name}/{recognition_corpus_key}-{recog_name}",
                            **kwargs,
                        )
                if "quant" in name:
                    continue
                onnx_flow = make_precomputed_hybrid_onnx_feature_flow(
                    onnx_model=onnx_model,
                    io_map=io_map,
                    cpu=kwargs.get("cpu", 1),
                )
                flow = add_fwd_flow_to_base_flow(feature_flow, onnx_flow)
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
                    lmgc_alias=f"lmgc/{name}/{recognition_corpus_key}-{recog_name}",
                    **kwargs,
                )
