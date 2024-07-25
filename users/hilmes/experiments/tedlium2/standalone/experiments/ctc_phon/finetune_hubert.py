from dataclasses import asdict
import numpy as np
from typing import cast
import copy
import itertools

from sisyphus import tk
from onnxruntime.quantization.quantize import QuantType, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationMethod

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, get_forward_config, search_single
from ...config import get_onnx_export_config
from ...report import generate_report
from .tune_eval import tune_and_evaluate_helper, eval_model, build_hubert_report
from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob
from i6_core.returnn.compile import TorchOnnxExportJob
from functools import partial

def get_quant_str(num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format):
    if quant_mode == CalibrationMethod.MinMax:
        mode_str = "/quant/min_max"
    elif quant_mode == CalibrationMethod.Entropy:
        mode_str = "quant/entropy"
    else:
        mode_str = "quant/percentile"
    mode_str += f"/{num_seqs}"
    for x in [activation_type, weight_type]:
        if x == QuantType.QInt8:
            mode_str += "_QInt8"
        elif x == QuantType.QUInt8:
            mode_str += "_QUint8"
    if average:
        mode_str += "_avg"
    if sym:
        mode_str += "_sym"
    if quant_ops is not None:
        mode_str += "_" + "_".join(quant_ops)
    else:
        mode_str += "_full"
    if quant_format == QuantFormat.QDQ:
        mode_str += "_QDQ"
    else:
        mode_str += "QOperator"
    return mode_str

def eow_phon_ted_tune_hubert():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/tune_hubert"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.hubert_tune_0711.hubert_tune_v1_cfg import (
        ModelConfig,
    )
    # TODO
    # "large-ls960-ft" "large-ll60k" "xlarge-ll60k" "base-ll60k"
    hubert_report = {}
    for model in ["base-ls960", "large-ls960-ft", "large-ll60k", "xlarge-ll60k", "xlarge-ls960-ft",]:
        #               AMP / no AMP
        # Base-ls960    6.3 / 6.3
            model_config = ModelConfig(
                label_target_size=vocab_size_without_blank,
                final_dropout=0.2,
                model_name=model,
                finetune_layer=True,
                keep_layers=None
            )
            network_module = "ctc.hubert_tune_0711.hubert_tune_v1"
            keep_epochs = [10, 20, 30, 40, 50, 100, 150, 200, 250]
            train_config_24gbgpu = {
                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                + list(np.linspace(5e-4, 5e-5, 110))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 120 * 16000, # if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 30 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 3, # if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 12,
                "cleanup_old_models": {
                    "keep_last_n": 4,
                    "keep_best_n": 4,
                    "keep": keep_epochs,
                }
            }
            if model in ["xlarge-ll60k", "xlarge-ls960-ft"]:
                train_config_24gbgpu["max_seqs"] = 1
            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
            }

            training_name = prefix_name + "/" + network_module + f"_{model}"
            train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args,
                    train_data=train_data,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    specific_epoch=keep_epochs,
                    prior_scales=[0.3, 0.5, 0.7, 0.9],
                )
            generate_report(results=results, exp_name=training_name)
            hubert_report[training_name] = results
            del results
            train_config_24gbgpu_amp = {
                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                + list(np.linspace(5e-4, 5e-5, 110))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 120 * 16000 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 60 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 3 if not model in ["xlarge-ll60k", "xlarge-ls960-ft"] else 6,
                "cleanup_old_models": {
                    "keep_last_n": 4,
                    "keep_best_n": 4,
                    "keep": keep_epochs,
                },
                "torch_amp_options": {"dtype": "bfloat16"},
            }
            if model in ["xlarge-ll60k", "xlarge-ls960-ft"]:
                train_config_24gbgpu_amp["max_seqs"] = 1
            train_args_amp = {
                "config": train_config_24gbgpu_amp,
                "network_module": network_module,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,

            }

            training_name = prefix_name + "/" + network_module + f"_{model}_amp"
            train_job = training(training_name, train_data, train_args_amp, num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args_amp,
                    train_data=train_data,
                    decoder_config=default_decoder_config,
                    dev_dataset_tuples=dev_dataset_tuples,
                    specific_epoch=keep_epochs,
                    prior_scales=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                    lm_scales=[1.4, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
                )
            generate_report(results=results, exp_name=training_name)
            hubert_report[training_name] = results
            del results

            if not model == "base-ls960":
                continue
            results = {}
            epochs = [250]
            num_seqs_ls = [100]
            quant_modes = [CalibrationMethod.MinMax]
            activation_types = [QuantType.QInt8]
            weight_types = [QuantType.QInt8]
            average_modes = [True, False]
            sym_modes = [False, True]
            quant_ops_ls = [["Conv", "MatMul"]]
            quant_formats = [QuantFormat.QDQ]
            quant_decoder_module = "ctc.decoder.flashlight_quant_onnx_ctc"
            from ...pytorch_networks.ctc.decoder.flashlight_quant_onnx_ctc import DecoderConfig

            quant_decoder_config = DecoderConfig(
                lexicon=get_text_lexicon(),
                returnn_vocab=label_datastream.vocab,
                beam_size=1024,
                beam_size_token=12,  # makes it much faster
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,
            )
            for num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format, epoch in (
                    itertools.product(
                        num_seqs_ls, quant_modes, activation_types, weight_types, average_modes,
                        sym_modes, quant_ops_ls, quant_formats, epochs)):
                quant_str = get_quant_str(num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops,
                                          quant_format)

                returnn_export_config = get_onnx_export_config(
                    network_module=network_module,
                    config={},
                    net_args=train_args["net_args"],
                )
                onnx_job = TorchOnnxExportJob(
                    returnn_config=returnn_export_config,
                    checkpoint=train_job.out_checkpoints[epoch],
                    returnn_root=MINI_RETURNN_ROOT,
                    returnn_python_exe=RETURNN_EXE,
                )
                onnx_job.set_keep_value(5)
                onnx_job.add_alias(training_name + f"/onnx_export_{epoch}")
                for lm_weight in [2.2]: # TODO set proper scales
                    for prior_scale in [0.7]:
                        decoder_config = copy.deepcopy(quant_decoder_config)
                        decoder_config.lm_weight = lm_weight
                        decoder_config.prior_scale = prior_scale
                        decoder_args = {
                            "quantized_model": onnx_job.out_onnx_model,
                            "config": asdict(decoder_config)
                        }
                        returnn_search_config = get_forward_config(
                            network_module=network_module,
                            config={},
                            net_args=train_args_amp['net_args'],
                            decoder_args=decoder_args,
                            decoder=quant_decoder_module,
                            debug=train_args_amp['debug'],
                        )
                        wers = {}
                        search_jobs = []
                        search_prefix = training_name + "/onnx" + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                        for key, (dev_dataset, dev_dataset_reference) in dev_dataset_tuples.items():
                            search_name = search_prefix + "/%s" % key
                            wers[search_name], search_job = search_single(
                                search_name,
                                returnn_search_config,
                                train_job.out_checkpoints[epoch],  # dummy placeholder, decoder replaces it by onnx checkpoint
                                dev_dataset,
                                dev_dataset_reference,
                                RETURNN_EXE,
                                MINI_RETURNN_ROOT,
                                mem_rqmt=30 if "hubert_tune" in search_name else 10,
                                use_gpu=False,
                            )
                            search_jobs.append(search_job)
                            results.update(wers)
                        for random_seed in [0, 1]:
                            quant_name = training_name + quant_str + f"_seed_{random_seed}"
                            quant_data = copy.deepcopy(train_data.train.as_returnn_opts())
                            quant_data['datasets']['zip_dataset']['partition_epoch'] = 1
                            quant_data['datasets']['zip_dataset']['seq_ordering'] = "random"
                            quant_data['datasets']['zip_dataset']['fixed_random_seed'] = random_seed
                            quant_job = ModelQuantizeStaticJob(
                                dataset=quant_data,
                                model=onnx_job.out_onnx_model,
                                num_seqs=num_seqs,
                                calibrate_method=quant_mode,
                                activation_type=activation_type,
                                weight_type=weight_type,
                                moving_average=average,
                                symmetric=sym,
                                ops_to_quant=quant_ops,
                                quant_format=quant_format,
                                num_parallel_seqs=None,
                            )
                            quant_job.set_keep_value(5)
                            quant_job.add_alias(quant_name + f"/quantization_{epoch}")
                            decoder_args = {
                                "quantized_model": quant_job.out_model,
                                "config": asdict(decoder_config)
                            }
                            returnn_search_config = get_forward_config(
                                network_module=network_module,
                                config={},
                                net_args=train_args_amp['net_args'],
                                decoder_args=decoder_args,
                                decoder=quant_decoder_module,
                                debug=train_args_amp['debug'],
                            )
                            wers = {}
                            search_jobs = []
                            search_prefix = quant_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                            for key, (dev_dataset, dev_dataset_reference) in dev_dataset_tuples.items():
                                search_name = search_prefix + "/%s" % key
                                wers[search_name], search_job = search_single(
                                    search_name,
                                    returnn_search_config,
                                    train_job.out_checkpoints[epoch],  # dummy placeholder, decoder replaces it by onnx checkpoint
                                    dev_dataset,
                                    dev_dataset_reference,
                                    RETURNN_EXE,
                                    MINI_RETURNN_ROOT,
                                    mem_rqmt=30 if "hubert_tune" in search_name else 10,
                                    use_gpu=False,
                                )
                                search_jobs.append(search_job)
                                results.update(wers)
            generate_report(results=results, exp_name=training_name + "_quantized")
            del results
    tk.register_report("reports/finetune_hubert_report", partial(build_hubert_report, hubert_report))
