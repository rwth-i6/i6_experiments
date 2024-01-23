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
from i6_core.returnn.training import GetBestPtCheckpointJob


# -------------------- Init --------------------

Path = tk.setup_path(__package__)
from i6_core.mm import CreateDummyMixturesJob

from i6_experiments.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.torch_baselines.pytorch_networks.prior.forward import \
    ReturnnForwardComputePriorJob
from i6_experiments.users.hilmes.tools.onnx import ExportPyTorchModelToOnnxJob


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

        self.post_config.session.intra_op_num_threads = intra_op_threads
        self.post_config.session.inter_op_num_threads = inter_op_threads

        for k, v in io_map.items():
            self.config.io_map[k] = v


class PyTorchOnnxHybridSystem(HybridSystem):
    
    
    def nn_recognition(
        self,
        name: str,
        returnn_config: returnn.ReturnnConfig,
        checkpoints: Dict[int, Union[returnn.Checkpoint, returnn.PtCheckpoint]],
        train_job: Union[returnn.ReturnnTrainingJob, returnn.ReturnnRasrTrainingJob],  # TODO maybe Optional if prior file provided -> automatically construct dummy file
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
                #assert acoustic_mixture_path is not None
                if epoch == "best":
                    assert train_job is not None, "train_job needed to get best epoch checkpoint"
                    best_checkpoint_job = GetBestPtCheckpointJob(
                        train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_CE", index=0
                    )
                    checkpoint = best_checkpoint_job.out_checkpoint
                    epoch_str = epoch
                    epoch_num = best_checkpoint_job.out_epoch
                else:
                    assert epoch in checkpoints.keys()
                    checkpoint = checkpoints[epoch]
                    epoch_str = f"{epoch:03d}"

                io_map = {
                    "features": "data",
                    "output": "classes"
                }
                if needs_features_size:
                    io_map["features-size"] = "data_len"
                acoustic_mixture_path = CreateDummyMixturesJob(
                    num_mixtures=returnn_config.config['extern_data']['classes']['dim'],
                    num_features=returnn_config.config['extern_data']['data']['dim']).out_mixtures
                lmgc_scorer = rasr.GMMFeatureScorer(acoustic_mixture_path)
                prior_config = copy.deepcopy(returnn_config)
                assert len(self.train_cv_pairing) == 1, "multiple train corpora not supported yet"
                train_data = self.train_input_data[self.train_cv_pairing[0][0]]
                prior_config.config["train"] = copy.deepcopy(train_data) if isinstance(train_data,
                    Dict) else copy.deepcopy(train_data.get_data_dict())
                # prior_config.config["train"]["datasets"]["align"]["partition_epoch"] = 3
                prior_config.config["train"]["datasets"]["align"]["seq_ordering"] = "random"
                prior_config.config["forward_batch_size"] = 10000
                if epoch == "best":
                    prior_config.config["load_epoch"] = epoch_num
                if "chunking" in prior_config.config.keys():
                    del prior_config.config["chunking"]
                from i6_core.tools.git import CloneGitRepositoryJob
                returnn_root = CloneGitRepositoryJob(
                    "https://github.com/rwth-i6/returnn",
                    commit="925e0023c52db071ecddabb8f7c2d5a88be5e0ec",
                ).out_repository
                if "whisper_large" in name:
                    prior_config.config["max_seqs"] = 3
                elif "whisper" in name:
                    prior_config.config["max_seqs"] = 1
                nn_prior_job = ReturnnForwardComputePriorJob(
                    model_checkpoint=checkpoint,
                    returnn_config=prior_config,
                    returnn_python_exe=self.returnn_python_exe,
                    returnn_root=returnn_root,
                    log_verbosity=train_job.returnn_config.post_config["log_verbosity"],
                )
                if "whisper" in name:
                    nn_prior_job.rqmt["mem"] = 16
                if any(x in name for x in ["whisper_large", "whisper_v2_large", "whisper_v2_medium"]):
                    nn_prior_job.rqmt["gpu_mem"] = 24
                nn_prior_job.add_alias("extract_nn_prior/" + name + "/epoch_" + str(epoch))
                prior_file = nn_prior_job.out_prior_xml_file
                onnx_job = ExportPyTorchModelToOnnxJob(
                    pytorch_checkpoint=checkpoint,
                    returnn_config=returnn_config,
                    returnn_root=self.returnn_root,
                    quantize_dynamic=quantize_dynamic,
                )
                onnx_job.add_alias("export_onnx/" + name + "/epoch_" + epoch_str)
                onnx_model = onnx_job.out_onnx_model

                kwargs = copy.deepcopy(kwargs)
                data_num_ls = kwargs.pop("quantize", None)
                if data_num_ls is not None and ((prior == 0.7 and lm == 10.0) or "speed" in name):
                    for data_num in data_num_ls:
                        if data_num > 250 and "blstm" in name:
                            continue
                        from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob
                        from onnxruntime.quantization import CalibrationMethod
                        for quant_mode in [CalibrationMethod.MinMax, CalibrationMethod.Entropy, CalibrationMethod.Percentile]:
                            for average in [True, False]:
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
                                quant_job = ModelQuantizeStaticJob(
                                    model=onnx_model,
                                    dataset=prior_config.config["train"]["datasets"]["feat"],
                                    num_seqs=data_num,
                                    num_parallel_seqs=10,
                                    calibrate_method=quant_mode,
                                    moving_average=average
                                )
                                quant_job.add_alias("quantize_static/"+ name + "/" + mode_str + "/epoch" + epoch_str + "_" + str(data_num))
                                quant_model = quant_job.out_model
                                scorer = OnnxFeatureScorer(
                                    mixtures=acoustic_mixture_path,
                                    model=quant_model,
                                    priori_scale=prior,
                                    io_map=io_map,
                                    inter_op_threads=kwargs.get("cpu", 1),
                                    intra_op_threads=kwargs.get("cpu", 1),
                                    prior_file=prior_file
                                )

                                self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}-{mode_str}-{data_num}"] = scorer
                                self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-onnx-{epoch_str}-{mode_str}-{data_num}"] = feature_flow

                                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{mode_str}-{data_num}"
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
                                mode_str += "_sym"
                                quant_job = ModelQuantizeStaticJob(
                                    model=onnx_model,
                                    dataset=prior_config.config["train"]["datasets"]["feat"],
                                    num_seqs=data_num,
                                    num_parallel_seqs=10,
                                    calibrate_method=quant_mode,
                                    moving_average=average,
                                    symmetric=True
                                )
                                quant_job.add_alias(
                                    "quantize_static/" + name + "/" + mode_str + "/epoch" + epoch_str + "_" + str(
                                        data_num))
                                quant_model = quant_job.out_model
                                scorer = OnnxFeatureScorer(
                                    mixtures=acoustic_mixture_path,
                                    model=quant_model,
                                    priori_scale=prior,
                                    io_map=io_map,
                                    inter_op_threads=kwargs.get("cpu", 1),
                                    intra_op_threads=kwargs.get("cpu", 1),
                                    prior_file=prior_file
                                )

                                self.feature_scorers[recognition_corpus_key][
                                    f"pre-nn-{name}-{prior:02.2f}-{mode_str}-{data_num}"] = scorer
                                self.feature_flows[recognition_corpus_key][
                                    f"{feature_flow_key}-onnx-{epoch_str}-{mode_str}-{data_num}"] = feature_flow

                                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{mode_str}-{data_num}"
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
                            #for smooth in [-0.1, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0]:
                            for smooth in [-0.1, 0.05, 0.1, 0.2, 0.25, 0.3]:
                                if not quant_mode == CalibrationMethod.MinMax or "speed" in name:
                                    continue
                                mode_str = f"quant_min_max_smooth_{str(smooth)}"
                                quant_job = ModelQuantizeStaticJob(
                                    model=onnx_model,
                                    dataset=prior_config.config["train"]["datasets"]["feat"],
                                    num_seqs=data_num,
                                    num_parallel_seqs=10,
                                    calibrate_method=quant_mode,
                                    smoothing_factor=smooth
                                )
                                quant_job.add_alias(
                                    "quantize_static/" + name + "/" + mode_str + "/epoch" + epoch_str + "_" + str(
                                        data_num))
                                quant_model = quant_job.out_model
                                scorer = OnnxFeatureScorer(
                                    mixtures=acoustic_mixture_path,
                                    model=quant_model,
                                    priori_scale=prior,
                                    io_map=io_map,
                                    inter_op_threads=kwargs.get("cpu", 1),
                                    intra_op_threads=kwargs.get("cpu", 1),
                                    prior_file=prior_file
                                )

                                self.feature_scorers[recognition_corpus_key][
                                    f"pre-nn-{name}-{prior:02.2f}-{mode_str}-{data_num}"] = scorer
                                self.feature_flows[recognition_corpus_key][
                                    f"{feature_flow_key}-onnx-{epoch_str}-{mode_str}-{data_num}"] = feature_flow

                                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{mode_str}-{data_num}"
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
                    if name == "train.train-torch_jj_config2-quant":
                        for data_num in [10000, 25000, 50000, 75000, 100000]:
                            from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob
                            for model in ["new", "entropy"]:
                                if model == "new":
                                    mode_str = "quant_min_max"
                                elif model == "entropy":
                                    mode_str = "quant_entropy"
                                else:
                                    raise NotImplementedError
                                scorer = OnnxFeatureScorer(
                                    mixtures=acoustic_mixture_path,
                                    model=tk.Path(f"/work/asr4/hilmes/debug/quantization/model_quant_{data_num}_{model}.onnx"),
                                    priori_scale=prior,
                                    io_map=io_map,
                                    inter_op_threads=kwargs.get("cpu", 1),
                                    intra_op_threads=kwargs.get("cpu", 1),
                                    prior_file=prior_file
                                )

                                self.feature_scorers[recognition_corpus_key][
                                    f"pre-nn-{name}-{prior:02.2f}-{mode_str}-{data_num}"] = scorer
                                self.feature_flows[recognition_corpus_key][
                                    f"{feature_flow_key}-onnx-{epoch_str}-{mode_str}-{data_num}"] = feature_flow

                                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{mode_str}-{data_num}"
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
                if name == "train.train-torch_jj_config2-quant":
                    for data_num in [10000, 25000, 50000, 75000, 100000]:
                        from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob
                        for model in ["new", "entropy"]:
                            if model == "new":
                                mode_str = "quant_min_max"
                            elif model == "entropy":
                                mode_str = "quant_entropy"
                            else:
                                raise NotImplementedError
                            scorer = OnnxFeatureScorer(
                                mixtures=acoustic_mixture_path,
                                model=tk.Path(
                                    f"/work/asr4/hilmes/debug/quantization/model_quant_{data_num}_{model}.onnx"),
                                priori_scale=prior,
                                io_map=io_map,
                                inter_op_threads=kwargs.get("cpu", 1),
                                intra_op_threads=kwargs.get("cpu", 1),
                                prior_file=prior_file
                            )

                            self.feature_scorers[recognition_corpus_key][
                                f"pre-nn-{name}-{prior:02.2f}-{mode_str}-{data_num}"] = scorer
                            self.feature_flows[recognition_corpus_key][
                                f"{feature_flow_key}-onnx-{epoch_str}-{mode_str}-{data_num}"] = feature_flow

                            recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{mode_str}-{data_num}"
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
                if "quant" in name:
                    continue
                scorer = OnnxFeatureScorer(
                    mixtures=acoustic_mixture_path,
                    model=onnx_model,
                    priori_scale=prior,
                    io_map=io_map,
                    inter_op_threads=kwargs.get("cpu", 1),
                    intra_op_threads=kwargs.get("cpu", 1),
                    prior_file=prior_file
                )

                self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}"] = scorer
                self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-onnx-{epoch_str}"] = feature_flow

                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}"
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

                if name == "train.train-torch_jj_config2-dev" and prior == 0.7 and lm == 10.0:
                    returnn_root = CloneGitRepositoryJob(
                        "https://github.com/rwth-i6/returnn",
                        commit="31c1bbe3b9c90a9234122c762c175fc89cc9d8de",
                    ).out_repository

                    from i6_core.returnn import TorchOnnxExportJob
                    config = copy.deepcopy(returnn_config)
                    config.config["model_outputs"] = {"output": {"dim": 9001}}
                    new_export_job = TorchOnnxExportJob(
                        returnn_config=config,
                        checkpoint=checkpoint,
                        returnn_python_exe=self.returnn_python_exe,
                        returnn_root=returnn_root
                    )
                    new_export_job.add_alias("export_onnx/" + name + "/epoch_" + epoch_str + "_new")
                    new_onnx_model = new_export_job.out_onnx_model
                    from i6_core.rasr.feature_scorer import PrecomputedHybridFeatureScorer
                    from i6_core.returnn.flow import make_precomputed_hybrid_onnx_feature_flow, add_fwd_flow_to_base_flow
                    new_map = copy.deepcopy(io_map)
                    new_map["features-size"] = "data:size1"
                    new_map["output"] = "output"
                    onnx_flow = make_precomputed_hybrid_onnx_feature_flow(
                        onnx_model=new_onnx_model,
                        io_map=new_map,
                        cpu=kwargs.get("cpu", 1),
                    )
                    flow = add_fwd_flow_to_base_flow(feature_flow, onnx_flow, fwd_input_name="fwd-input")
                    scorer = rasr.PrecomputedHybridFeatureScorer(
                        prior_mixtures=acoustic_mixture_path,
                        priori_scale=prior,
                        prior_file=prior_file,
                    )
                    self.feature_scorers[recognition_corpus_key][f"pre-nn-{name}-{prior:02.2f}-test"] = scorer
                    self.feature_flows[recognition_corpus_key][f"{feature_flow_key}-onnx-{epoch:03d}-test"] = flow

                    recog_name = f"e{epoch:03d}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-test"
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