import copy
import itertools
from typing import Dict, List, Optional, Tuple, Union, Any

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.rasr as rasr
import i6_core.returnn as returnn

from i6_core.returnn.flow import (
    make_precomputed_hybrid_tf_feature_flow,
    add_tf_flow_to_base_flow,
)
from i6_core.returnn.training import GetBestPtCheckpointJob, AverageTorchCheckpointsJob
from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob

from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType

# -------------------- Init --------------------

Path = tk.setup_path(__package__)
from i6_core.mm import CreateDummyMixturesJob

from i6_experiments.users.hilmes.common.setups.rasr.hybrid_system import HybridSystem
from i6_experiments.users.hilmes.experiments.tedlium2.asr_2023.hybrid.torch_baselines.pytorch_networks.prior.forward import \
    ReturnnForwardComputePriorJob
from i6_experiments.users.hilmes.tools.onnx import ExportPyTorchModelToOnnxJob


def get_quant_str(
        quant_mode: CalibrationMethod,
        average: bool, sym: bool,
        activation_type: QuantType,
        weight_type: QuantType,
        quant_format: QuantFormat,
        quant_ops: Optional[List[str]],
        percentile: Optional[float],
        num_bins: Optional[int],
        random_seed: int,
        filter_opts: Optional[Dict[str, Any]],
):
    if quant_mode == CalibrationMethod.MinMax:
        mode_str_tmp = "quant_min_max"
    elif quant_mode == CalibrationMethod.Entropy:
        mode_str_tmp = "quant_entropy"
    else:
        mode_str_tmp = "quant_percentile"
    if percentile:
        mode_str_tmp += f"_{percentile}"
    if num_bins:
        mode_str_tmp += f"_{num_bins}"
    if average:
        mode_str_tmp += "_avg"
    if sym:
        mode_str_tmp += "_sym"
    if activation_type == QuantType.QInt8:
        mode_str_tmp += "/act_int8"
    else:
        mode_str_tmp += "/act_qint8"
    if weight_type == QuantType.QInt8:
        mode_str_tmp += "_w_int8"
    else:
        mode_str_tmp += "_w_qint8"
    if quant_format == QuantFormat.QDQ:
        mode_str_tmp += "_qdq"
    elif quant_format == QuantFormat.QOperator:
        mode_str_tmp += "_qop"
    else:
        raise NotImplementedError
    if quant_ops:
        mode_str_tmp = mode_str_tmp + "_" + "_".join(quant_ops)
    if not random_seed == 0:
        mode_str_tmp = mode_str_tmp + "_" + f"iter{random_seed}"
    if filter_opts is not None:
        if "max_seq_len" in filter_opts:
            mode_str_tmp += f"_max_calib_len_{filter_opts['max_seq_len']}"
        if "min_seq_len" in filter_opts:
            mode_str_tmp += f"_min_calib_len_{filter_opts['min_seq_len']}"
        if "partition" in filter_opts:
            mode_str_tmp += f"_partition_{sum([x * y for x,y in filter_opts['partition']])}"
        if "budget" in filter_opts:
            mode_str_tmp += f"_budget_{filter_opts['budget'][0]}_{filter_opts['budget'][1]}"
        if "single_tag" in filter_opts:
            mode_str_tmp += f"_single_tag"
        if "unique_tags" in filter_opts:
            mode_str_tmp += f"_unique_tags"

    return mode_str_tmp


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
        if epoch == "best":
            prior_config.config["load_epoch"] = epoch_num
        if "chunking" in prior_config.config.keys():
            del prior_config.config["chunking"]
        from i6_core.tools.git import CloneGitRepositoryJob
        if any(x in name for x in ["distill_whisper"]):
            returnn_root = CloneGitRepositoryJob(
                "https://github.com/rwth-i6/returnn",
                commit="d4ab1d8fcbe3baa11f6d8e2cf8e443bc0e9e9fa2",
            ).out_repository.copy()
            returnn_root.hash_overwrite = "RETURNN_PRIOR_COMMIT"
        else:
            returnn_root = CloneGitRepositoryJob(
                "https://github.com/rwth-i6/returnn",
                commit="925e0023c52db071ecddabb8f7c2d5a88be5e0ec",
            ).out_repository
        big_gpu_names = ["whisper_large"]  # not including v2large and v2medium for legacy reasons
        if any(x in name for x in big_gpu_names):
            prior_config.config["max_seqs"] = 3
        elif "whisper" in name:
            prior_config.config["max_seqs"] = 1
        elif "hubert" in name:
            prior_config.config["max_seqs"] = 1
        nn_prior_job = ReturnnForwardComputePriorJob(
            model_checkpoint=checkpoint,
            returnn_config=prior_config,
            returnn_python_exe=self.returnn_python_exe,
            returnn_root=returnn_root,
            log_verbosity=train_job.returnn_config.post_config["log_verbosity"],
            time_rqmt=2
        )
        if any(x in name for x in ["whisper", "hubert"]):
            nn_prior_job.rqmt["mem"] = 16
        if any(x in name for x in ["whisper_v2_large"] + big_gpu_names):
            nn_prior_job.rqmt["gpu_mem"] = 24
        nn_prior_job.add_alias("extract_nn_prior/" + name + "/epoch_" + str(epoch))
        prior_file = nn_prior_job.out_prior_xml_file
        return prior_file, prior_config
    
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
        needs_features_size: bool = True,
        acoustic_mixture_path: Optional[tk.Path] = None,
        best_checkpoint_key: str = "dev_loss_CE",
        **kwargs,
    ):
        with (tk.block(f"{name}_recognition")):
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
                    best_checkpoint_job.add_alias(f"get_best/{name}")
                    checkpoint = best_checkpoint_job.out_checkpoint
                    epoch_str = epoch
                    epoch_num = best_checkpoint_job.out_epoch
                elif epoch == "avrg":
                    assert train_job is not None, "train_job needed to average checkpoints"
                    from i6_core.tools.git import CloneGitRepositoryJob
                    chkpts = []
                    for x in [0, 1, 2, 3]:
                        best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates,
                                                          key=best_checkpoint_key, index=x)
                        best_job.add_alias(f"get_best/{name}_{x}")
                        chkpts.append(best_job.out_checkpoint)
                        avrg_job = AverageTorchCheckpointsJob(
                            checkpoints=chkpts,
                            returnn_python_exe=self.returnn_python_exe,
                            returnn_root=CloneGitRepositoryJob(
                            "https://github.com/rwth-i6/returnn",
                                commit="3bd8b438e4bd6d409a1796b6a4ba35175bad34ea",
                            ).out_repository
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
                prior_file, prior_config = self.calcluate_nn_prior(
                    returnn_config=returnn_config,
                    epoch=epoch,
                    epoch_num=epoch_num,
                    name=name,
                    checkpoint=checkpoint,
                    train_job=train_job
                )
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

                onnx_job = ExportPyTorchModelToOnnxJob(
                    pytorch_checkpoint=checkpoint,
                    returnn_config=returnn_config,
                    returnn_root=self.returnn_root,
                    quantize_dynamic=quantize_dynamic,
                )
                if any(x in name for x in ["whisper", "hubert"]):
                    onnx_job.rqmt["mem"] = 48
                onnx_job.add_alias("export_onnx/" + name + "/epoch_" + epoch_str)
                onnx_job.set_keep_value(5)
                onnx_model = onnx_job.out_onnx_model

                tmp_kwargs = copy.deepcopy(kwargs)
                data_num_ls = tmp_kwargs.pop("quantize", None)
                if data_num_ls is not None:
                    quant_modes: List[CalibrationMethod] = tmp_kwargs.pop("quant_modes", [CalibrationMethod.MinMax, CalibrationMethod.Entropy, CalibrationMethod.Percentile])
                    avg_modes: List[bool] = tmp_kwargs.pop("quant_avg_modes", [True, False])
                    sym_modes: List[bool] = tmp_kwargs.pop("quant_sym_modes", [True, False])
                    activation_type_ls: List[QuantType] = tmp_kwargs.pop("quant_activation_types", [QuantType.QInt8])
                    weight_type_ls: List[QuantType] = tmp_kwargs.pop("quant_weight_types", [QuantType.QInt8])
                    quant_format_ls: List[QuantFormat] = tmp_kwargs.pop("quant_format", [QuantFormat.QDQ])
                    quant_ops_ls: List[Union[Optional, List[str]]] = tmp_kwargs.pop("quant_ops", [None])
                    num_parallel_seqs: int = tmp_kwargs.pop("quant_parallel_seqs", 10)
                    percentile_ls: List[float] = tmp_kwargs.pop("quant_percentiles", [99.999])
                    num_bin_ls: List[int] = tmp_kwargs.pop("quant_num_bin_ls", [2048])
                    random_seed_draws: Optional[int] = tmp_kwargs.pop("random_seed_draws", None)
                    final_skip_ls: Optional[Tuple[List[int], List[int]]] = tmp_kwargs.pop("final_skip_ls", None)
                    smooth_ls = tmp_kwargs.pop("smooth_ls", [])
                    quant_filter_opts = tmp_kwargs.pop("quant_filter_opts", [None])
                    for data_num in data_num_ls:
                        for quant_mode, average, sym, activation_type, weight_type, quant_format, quant_ops, percentile, num_bins, filter_opts in itertools.product(quant_modes, avg_modes, sym_modes, activation_type_ls, weight_type_ls, quant_format_ls, quant_ops_ls, percentile_ls, num_bin_ls, quant_filter_opts):
                            if average and (not quant_mode == CalibrationMethod.MinMax or "speed" in name):
                                continue
                            if not quant_mode == CalibrationMethod.MinMax and "speed" in name:
                                continue
                            if activation_type == QuantType.QInt8 and weight_type == QuantType.QUInt8:
                                continue
                            if not quant_mode == CalibrationMethod.Percentile:
                                percentile = None
                                num_bins = None
                            if random_seed_draws is None:
                                random_seed_draws = 1
                            for random_seed in range(random_seed_draws):
                                mode_str = get_quant_str(
                                    quant_mode=quant_mode,
                                    average=average,
                                    sym=sym,
                                    activation_type=activation_type,
                                    weight_type=weight_type,
                                    quant_format=quant_format,
                                    quant_ops=quant_ops,
                                    percentile=percentile,
                                    num_bins=num_bins,
                                    random_seed=random_seed,
                                    filter_opts=filter_opts,
                                )
                                quant_job = ModelQuantizeStaticJob(
                                    model=onnx_model,
                                    dataset=prior_config.config["train"]["datasets"]["feat"],
                                    num_seqs=data_num,
                                    num_parallel_seqs=num_parallel_seqs,
                                    calibrate_method=quant_mode,
                                    moving_average=average,
                                    symmetric=sym,
                                    weight_type=weight_type,
                                    activation_type=activation_type,
                                    quant_format=quant_format,
                                    ops_to_quant=quant_ops,
                                    num_bins=num_bins,
                                    percentile=percentile,
                                    random_seed=random_seed,
                                    filter_opts=filter_opts,
                                )
                                if "whisper" in name:
                                    quant_job.rqmt['mem'] += 48
                                    quant_job.rqmt['time'] += 1
                                quant_job.add_alias("quantize_static/" + name + "/" + mode_str + "/epoch" + epoch_str + "_" + str(data_num))
                                quant_job.set_keep_value(5)
                                quant_model = quant_job.out_model
                                if data_num is None:
                                    self.jobs[recognition_corpus_key][f"quantize_static/" + name + "/" + mode_str + "/epoch" + epoch_str + "_" + str(data_num)] = quant_job
                                scorer = OnnxFeatureScorer(
                                    mixtures=acoustic_mixture_path,
                                    model=quant_model,
                                    priori_scale=prior,
                                    io_map=io_map,
                                    inter_op_threads=tmp_kwargs.get("cpu", 1),
                                    intra_op_threads=tmp_kwargs.get("cpu", 1),
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
                                    **tmp_kwargs,
                                )
                            if final_skip_ls:
                                for final_skip_step in final_skip_ls[0]:
                                    for final_skip_count in final_skip_ls[1]:
                                        tmp_mode = mode_str + f"_skip_{final_skip_step}_{final_skip_count}"
                                        tmp_data = copy.deepcopy(prior_config.config["train"]["datasets"]["feat"])
                                        quant_job = ModelQuantizeStaticJob(
                                            model=onnx_model,
                                            dataset=tmp_data,
                                            num_seqs=data_num,
                                            num_parallel_seqs=num_parallel_seqs,
                                            calibrate_method=quant_mode,
                                            moving_average=average,
                                            symmetric=sym,
                                            random_seed=5,
                                            final_skip=(final_skip_step, final_skip_count),
                                        )
                                        quant_job.add_alias(
                                            "quantize_static/" + name + "/" + tmp_mode + "/epoch" + epoch_str + "_" + str(
                                                data_num))
                                        quant_model = quant_job.out_model
                                        scorer = OnnxFeatureScorer(
                                            mixtures=acoustic_mixture_path,
                                            model=quant_model,
                                            priori_scale=prior,
                                            io_map=io_map,
                                            inter_op_threads=tmp_kwargs.get("cpu", 1),
                                            intra_op_threads=tmp_kwargs.get("cpu", 1),
                                            prior_file=prior_file
                                        )

                                        self.feature_scorers[recognition_corpus_key][
                                            f"pre-nn-{name}-{prior:02.2f}-{tmp_mode}-{data_num}"] = scorer
                                        self.feature_flows[recognition_corpus_key][
                                            f"{feature_flow_key}-onnx-{epoch_str}-{tmp_mode}-{data_num}"] = feature_flow

                                        recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{tmp_mode}-{data_num}"
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
                                            **tmp_kwargs,
                                        )
                            for smooth in smooth_ls:
                                tmp_mode = mode_str + "_" + str(smooth)
                                quant_job = ModelQuantizeStaticJob(
                                        model=onnx_model,
                                        dataset=prior_config.config["train"]["datasets"]["feat"],
                                        num_seqs=data_num,
                                        num_parallel_seqs=num_parallel_seqs,
                                        calibrate_method=quant_mode,
                                        smoothing_factor=smooth
                                    )
                                quant_job.add_alias(
                                    "quantize_static/" + name + "/" + tmp_mode + "/epoch" + epoch_str + "_" + str(
                                        data_num))
                                quant_model = quant_job.out_model
                                scorer = OnnxFeatureScorer(
                                    mixtures=acoustic_mixture_path,
                                    model=quant_model,
                                    priori_scale=prior,
                                    io_map=io_map,
                                    inter_op_threads=tmp_kwargs.get("cpu", 1),
                                    intra_op_threads=tmp_kwargs.get("cpu", 1),
                                    prior_file=prior_file
                                )

                                self.feature_scorers[recognition_corpus_key][
                                    f"pre-nn-{name}-{prior:02.2f}-{tmp_mode}-{data_num}"] = scorer
                                self.feature_flows[recognition_corpus_key][
                                    f"{feature_flow_key}-onnx-{epoch_str}-{tmp_mode}-{data_num}"] = feature_flow

                                recog_name = f"e{epoch_str}-prior{prior:02.2f}-ps{pron:02.2f}-lm{lm:02.2f}-{tmp_mode}-{data_num}"
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
                                    **tmp_kwargs,
                                )

                if "quant" in name and not "rtf" in name:
                    continue
                scorer = OnnxFeatureScorer(
                    mixtures=acoustic_mixture_path,
                    model=onnx_model,
                    priori_scale=prior,
                    io_map=io_map,
                    inter_op_threads=tmp_kwargs.get("cpu", 1),
                    intra_op_threads=tmp_kwargs.get("cpu", 1),
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
                    **tmp_kwargs,
                )
