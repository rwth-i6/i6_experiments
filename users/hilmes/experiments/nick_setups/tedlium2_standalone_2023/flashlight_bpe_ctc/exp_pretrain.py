import itertools

from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional, Dict
from onnxruntime.quantization.quantize import QuantType, QuantFormat
from onnxruntime.quantization.calibrate import CalibrationMethod

from i6_core.report.report import _Report_Type
from i6_core.returnn import GetBestPtCheckpointJob, TorchOnnxExportJob

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.vocabulary import LabelDatastream

from .data import build_bpe_training_datasets, TrainingDatasetSettings, get_text_lexicon
from ..data import build_test_dataset, TrainingDatasets
from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT

from ..pipeline import training, search, compute_prior

from .config import get_training_config, get_search_config, get_prior_config


def flash_bpe_ctc_report_format(report: _Report_Type) -> str:
    extra_ls = []
    out = [(" ".join(recog.split("/")[-3:]), str(report[recog])) for recog in report if not any(extra in recog for extra in extra_ls)]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = [out[0]]
    for extra in extra_ls:
        out2 = [(" ".join(recog.split("/")[-3:]), str(report[recog])) for recog in report if extra in recog]
        out2 = sorted(out2, key=lambda x: float(x[1]))
        if len(out2) > 0:
            out.append((extra, ""))
            out.extend(out2)
            best_ls.append(out2[0])
    best_ls = sorted(best_ls, key=lambda x: float(x[1]))
    out.append(("Best Results", ""))
    out.extend(best_ls)
    return "\n".join([f"{pair[0]}:  {str(pair[1])}" for pair in out])


def get_quant_str(num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format):
    if quant_mode == CalibrationMethod.MinMax:
        mode_str = "quant_min_max"
    elif quant_mode == CalibrationMethod.Entropy:
        mode_str = "quant_entropy"
    else:
        mode_str = "quant_percentile"
    mode_str += f"_{num_seqs}"
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


def pretrained_experiments():
    prefix_name = "experiments/rescale/tedliumv2/flashlight_bpe_ctc/"

    BPE_SIZE = 1000

    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=5, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )

    train_settings_retrain = copy.deepcopy(train_settings)
    train_settings_retrain.epoch_wise_filters = []

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_bpe_training_datasets(
        bpe_size=BPE_SIZE,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    # build testing datasets
    test_dataset_tuples = {}
    # for testset in ["dev", "test"]:
    for testset in ["dev"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
        )
    from i6_experiments.common.baselines.tedlium2.lm.ngram_config import run_tedlium2_ngram_lm

    lms_system = run_tedlium2_ngram_lm(add_unknown_phoneme_and_mapping=False)
    lm = lms_system.interpolated_lms["dev-pruned"]["4gram"]
    arpa_ted_lm = lm.ngram_lm

    # ---------------------------------------------------------------------------------------------------------------- #

    def run_exp(
        ft_name,
        datasets: TrainingDatasets,
        train_args,
        search_args=None,
        with_prior=False,
        num_epochs=250,
        decoder="ctc.decoder.flashlight_bpe_ctc",
        eval_epochs: Optional[List] = None,
        quantize_args: Optional[Dict[str, str]] = None
    ):
        training_name = "/".join(ft_name.split("/")[:-1])
        search_args = search_args if search_args is not None else {}

        returnn_config = get_training_config(training_datasets=datasets, keep_epochs=eval_epochs, **train_args)
        train_job = training(training_name, returnn_config, RETURNN_EXE, MINI_RETURNN_ROOT, num_epochs=num_epochs)

        if eval_epochs is None or "onnx" in ft_name:
            eval_epochs = [num_epochs]
        search_job_ls = []
        report = {}
        returnn_search_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder)
        for epoch in eval_epochs:
            if with_prior:
                prior_args = copy.deepcopy(train_args)
                if "max_seqs" in prior_args["config"]:
                    prior_args["config"]["max_seqs"] = 15
                returnn_config = get_prior_config(training_datasets=datasets, **prior_args)
                prior_file = compute_prior(
                    ft_name,
                    returnn_config,
                    checkpoint=train_job.out_checkpoints[epoch],
                    returnn_exe=RETURNN_EXE,
                    returnn_root=MINI_RETURNN_ROOT,
                    epoch=str(epoch)  # just for alias generation
                )
                tk.register_output(training_name + f"/prior/{epoch}.txt", prior_file)
                search_args["prior_file"] = prior_file
            if quantize_args is not None:
                from i6_experiments.users.hilmes.tools.onnx import ModelQuantizeStaticJob
                returnn_export_config = get_search_config(**train_args, decoder_args=search_args, decoder=decoder, export=True)
                onnx_job = TorchOnnxExportJob(
                    returnn_config=returnn_export_config,
                    checkpoint=train_job.out_checkpoints[epoch],
                    returnn_root=MINI_RETURNN_ROOT,
                    returnn_python_exe=RETURNN_EXE,
                )
                onnx_job.add_alias(ft_name + f"/onnx_export_{epoch}")
                quant_job = ModelQuantizeStaticJob(
                    dataset=datasets.train.as_returnn_opts(),
                    model=onnx_job.out_onnx_model,
                    **quantize_args
                )
                quant_job.add_alias(ft_name + f"/quantization_{epoch}")
                decoder_args = copy.deepcopy(search_args)
                decoder_args["quantized_model"] = quant_job.out_model
                returnn_search_config = get_search_config(**train_args, decoder_args=decoder_args, decoder=decoder)
                format_string_report, values_report, search_jobs = search(
                    ft_name + "/quantized_%i" % epoch,
                    returnn_search_config,
                    train_job.out_checkpoints[epoch],
                    test_dataset_tuples,
                    RETURNN_EXE,
                    MINI_RETURNN_ROOT,
                )
                #for search_job in search_jobs:
                #    search_job.add_input(quant_job.out_model)
                search_job_ls += search_jobs
                report.update(values_report)
            else:
                format_string_report, values_report, search_jobs = search(
                    ft_name + "/default_%i" % epoch,
                    returnn_search_config,
                    train_job.out_checkpoints[epoch],
                    test_dataset_tuples,
                    RETURNN_EXE,
                    MINI_RETURNN_ROOT,
                )
                search_job_ls += search_jobs
                report.update(values_report)

        best_job = GetBestPtCheckpointJob(train_job.out_model_dir, train_job.out_learning_rates, key="dev_loss_ctc")
        best_job.add_alias(ft_name + "/get_best_job")
        format_string_report, values_report, search_jobs = search(
            ft_name + "/best_chkpt",
            returnn_search_config,
            best_job.out_checkpoint,
            test_dataset_tuples,
            RETURNN_EXE,
            MINI_RETURNN_ROOT,
        )
        search_job_ls += search_jobs
        report.update(values_report)

        return train_job, search_job_ls, format_string_report, report

    def generate_report(results, exp_name):
        from i6_core.report import GenerateReportStringJob, MailJob

        report = GenerateReportStringJob(report_values=results, report_template=flash_bpe_ctc_report_format)
        report.add_alias(f"report/report/{exp_name}")
        mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
        mail.add_alias(f"report/mail/{exp_name}")
        tk.register_output("mail/" + exp_name, mail.out_status)

    # from here on onwards, use default Adam with same OCLR
    default_search_args = {
        "lexicon": get_text_lexicon(bpe_size=BPE_SIZE),
        "returnn_vocab": label_datastream.vocab,
        "beam_size": 1024,
        "arpa_lm": arpa_ted_lm,
        "beam_threshold": 14,
    }

    from ..pytorch_networks.ctc.conformer_0923 import whisper_pretrained_v2_cfg

    whisper_cfg_2 = whisper_pretrained_v2_cfg.WhisperConfig(
        just_encoder=True,
        finetune_layer=6,
        split_seq=True,
        name="base.en",
        dropout=0,
    )
    model_config_whisper_base_v1 = whisper_pretrained_v2_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        whisper_config=whisper_cfg_2,
    )
    train_args_whisper_adam_accum50_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
                              + list(np.linspace(7e-4, 7e-5, 110))
                              + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 3,
            "accum_grad_multiple_step": 50,
        },
        "debug": True,
    }
    eval_epochs = [50, 75, 100, 150, 200, 250]
    train_args = {
        **copy.deepcopy(train_args_whisper_adam_accum50_jjlr),
        "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
        "net_args": {"model_config_dict": asdict(model_config_whisper_base_v1)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/whisper_base_pretrain_v5_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
            )
            #train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "conformer_0923/whisper_base_pretrain_v5_jjlr"
    )
    del results

    # whisper_cfg_1 = whisper_pretrained_v2_cfg.WhisperConfig(
    #     just_encoder=True,
    #     finetune_layer=1,
    #     split_seq=True,
    #     name="base.en",
    #     dropout=0,
    # )
    # model_config_whisper_v2 = whisper_pretrained_v2_cfg.ModelConfig(
    #     specauc_start_epoch=0,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     whisper_config=whisper_cfg_1,
    # )
    # train_args_whisper_adam_accum30_lr2e5 = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": [2e-5],
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 5,
    #         "accum_grad_multiple_step": 30,
    #     },
    #     "debug": False,
    # }
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_whisper_adam_accum30_lr2e5),
    #     "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
    #     "net_args": {"model_config_dict": asdict(model_config_whisper_v2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/whisper_pretrain_v5_base_1e-5/lm%.1f_prior%.2f_bs1024_th14" % (
    #             lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         # train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(
    #     results=results, exp_name=prefix_name + "conformer_0923/whisper_pretrain_v5_base_1e-5"
    # )
    # del results
    #
    # whisper_cfg_1 = whisper_pretrained_v2_cfg.WhisperConfig(
    #     just_encoder=True,
    #     finetune_layer=1,
    #     split_seq=True,
    #     name="base.en",
    #     dropout=0,
    # )
    # model_config_whisper_v2_later_spec = whisper_pretrained_v2_cfg.ModelConfig(
    #     specauc_start_epoch=11,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     whisper_config=whisper_cfg_1,
    # )
    # train_args_whisper_adam_accum30_lr1e5 = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": [1e-5],
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 5,
    #         "accum_grad_multiple_step": 30,
    #     },
    #     "debug": False,
    # }
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_whisper_adam_accum30_lr1e5),
    #     "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
    #     "net_args": {"model_config_dict": asdict(model_config_whisper_v2_later_spec)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/whisper_pretrain_v5_base_1e-5_specstart11/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         # train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(
    #     results=results, exp_name=prefix_name + "conformer_0923/whisper_pretrain_v5_base_1e-5_specstart11"
    # )
    # del results
    #
    # model_config_whisper_v2_no_spec = whisper_pretrained_v2_cfg.ModelConfig(
    #     specauc_start_epoch=5000,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     whisper_config=whisper_cfg_1,
    # )
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_whisper_adam_accum30_lr1e5),
    #     "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
    #     "net_args": {"model_config_dict": asdict(model_config_whisper_v2_no_spec)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/whisper_pretrain_v5_base_1e-5_nospec/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         # train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(
    #     results=results, exp_name=prefix_name + "conformer_0923/whisper_pretrain_v5_base_1e-5_nospec"
    # )
    # del results
    #
    # train_args = {
    #     **copy.deepcopy(train_args_whisper_adam_accum30_lr2e5),
    #     "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
    #     "net_args": {"model_config_dict": asdict(model_config_whisper_v2_no_spec)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/whisper_pretrain_v5_base_2e-5_nospec/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         # train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(
    #     results=results, exp_name=prefix_name + "conformer_0923/whisper_pretrain_v5_base_2e-5_nospec"
    # )
    # del results
    #
    # whisper_cfg_tune_2 = whisper_pretrained_v2_cfg.WhisperConfig(
    #     just_encoder=True,
    #     finetune_layer=2,
    #     split_seq=True,
    #     name="base.en",
    #     dropout=0,
    # )
    # model_config_whisper_v2 = whisper_pretrained_v2_cfg.ModelConfig(
    #     specauc_start_epoch=0,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     whisper_config=whisper_cfg_tune_2,
    # )
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_whisper_adam_accum30_lr1e5),
    #     "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
    #     "net_args": {"model_config_dict": asdict(model_config_whisper_v2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/whisper_pretrain_v5_base_2_1e-5/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         # train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(
    #     results=results, exp_name=prefix_name + "conformer_0923/whisper_pretrain_v5_base_2_1e-5"
    # )
    # del results
    #
    # whisper_cfg_tune_3 = whisper_pretrained_v2_cfg.WhisperConfig(
    #     just_encoder=True,
    #     finetune_layer=3,
    #     split_seq=True,
    #     name="base.en",
    #     dropout=0,
    # )
    # model_config_whisper_v2 = whisper_pretrained_v2_cfg.ModelConfig(
    #     specauc_start_epoch=0,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     whisper_config=whisper_cfg_tune_3,
    # )
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_whisper_adam_accum30_lr1e5),
    #     "network_module": "ctc.conformer_0923.whisper_pretrained_v5",
    #     "net_args": {"model_config_dict": asdict(model_config_whisper_v2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/whisper_pretrain_v5_base_3_1e-5/lm%.1f_prior%.2f_bs1024_th14" % (
    #                 lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         # train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(
    #     results=results, exp_name=prefix_name + "conformer_0923/whisper_pretrain_v5_base_3_1e-5"
    # )
    # del results

    from ..pytorch_networks.ctc.conformer_0923 import hubert_pretrained_v1_cfg

    hubert_cfg_1 = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=1,
        name="base-ls960",
    )
    model_config_hubert_v1 = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_1,
    )
    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
    #                           + list(np.linspace(7e-4, 7e-5, 110))
    #                           + list(np.linspace(7e-5, 1e-8, 30)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 25,
    #     },
    #     "debug": True,
    # }
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_v1)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/hubert_pretrain_v3_base_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 8.2
    #     results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_base_jjlr"
    # )
    # del results
    #
    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
    #                           + list(np.linspace(7e-4, 7e-5, 110))
    #                           + list(np.linspace(7e-5, 1e-8, 30)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 10,
    #     },
    #     "debug": False,
    # }
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_v1)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/hubert_pretrain_v3_base_smallaccum_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 7.9
    #     results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_base_smallaccum_jjlr"
    # )
    # del results
    #
    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
    #                           + list(np.linspace(7e-4, 7e-5, 110))
    #                           + list(np.linspace(7e-5, 1e-8, 30)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 100,
    #     },
    #     "debug": False,
    # }
    # eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_v1)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/hubert_pretrain_v3_base_largeaccum_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    #
    #         if prior_scale == 0.5 and lm_weight == 1.6:
    #             train_job, _, _, wer_values = run_exp(
    #                 prefix_name
    #                 + "conformer_0923/hubert_pretrain_v3_base_largeaccum_jjlr/lm%.1f_prior%.2f_bs1024_th14_onnx" % (
    #                 lm_weight, prior_scale),
    #                 datasets=train_data,
    #                 train_args=train_args,
    #                 search_args=search_args,
    #                 with_prior=True,
    #                 eval_epochs=eval_epochs,
    #                 decoder="ctc.decoder.flashlight_onnx_bpe_ctc"
    #             )
    #             results.update(wer_values)
    #             del wer_values
    # generate_report(  # 8.1
    #     results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_base_largeaccum_jjlr"
    # )
    # del results


    hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=2,
        name="base-ls960",
    )
    model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_2,
    )
    train_args_hubert_adam_accum25_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
                              + list(np.linspace(7e-4, 7e-5, 110))
                              + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 3,
            "accum_grad_multiple_step": 25,
        },
        "debug": True,
    }
    eval_epochs = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/hubert_pretrain_v3_base_tune2_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
            if lm_weight == 1.8 and prior_scale == 0.5:
                epochs = [200]
                #num_seqs_ls = [10, 100, 1000]
                num_seqs_ls = [10]
                quant_modes = [CalibrationMethod.MinMax]
                activation_types = [QuantType.QInt8]
                weight_types = [QuantType.QInt8]
                #average_modes = [True, False]
                average_modes = [True]
                #sym_modes = [True, False]
                sym_modes = [True]
                #quant_ops_ls = [None, ["Conv"], ["Linear"], ["Conv", "Linear"]]
                quant_ops_ls = [None]
                #quant_formats = [QuantFormat.QDQ, QuantFormat.QOperator]
                quant_formats = [QuantFormat.QDQ]
                for num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format in (
                    itertools.product(
                        num_seqs_ls, quant_modes, activation_types, weight_types, average_modes,
                        sym_modes, quant_ops_ls, quant_formats)):
                    quant_str = get_quant_str(num_seqs, quant_mode, activation_type, weight_type, average, sym, quant_ops, quant_format)
                    train_job, _, _, wer_values = run_exp(
                        prefix_name
                        + f"conformer_0923/hubert_pretrain_v3_base_tune2_jjlr/lm%.1f_prior%.2f_bs1024_th14_quant/{quant_str}" % (
                            lm_weight, prior_scale),
                        datasets=train_data,
                        train_args=train_args,
                        search_args=search_args,
                        with_prior=True,
                        eval_epochs=epochs,
                        decoder="ctc.decoder.flashlight_quantized_bpe_ctc",
                        quantize_args={
                            "num_seqs": num_seqs,
                            "num_parallel_seqs": 10,
                            "calibrate_method": CalibrationMethod.MinMax,
                            "moving_average": average,
                            "symmetric": sym,
                            "activation_type": activation_type,
                            "weight_type": weight_type,
                            "ops_to_quant": quant_ops,
                            "quant_format": quant_format,
                        }
                    )
                    results.update(wer_values)
                    del wer_values
    generate_report(  # 7.0
        results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_base_tune2_jjlr"
    )
    del results

    # hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
    #     finetune_layer=3,
    #     name="base-ls960",
    # )
    # model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
    #     specauc_start_epoch=0,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     hubert_cfg=hubert_cfg_2,
    # )
    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
    #                           + list(np.linspace(7e-4, 7e-5, 110))
    #                           + list(np.linspace(7e-5, 1e-8, 30)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 25,
    #     },
    #     "debug": True,
    # }
    # eval_epochs = [100, 150, 200, 225, 250]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/hubert_pretrain_v3_base_tune3_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 7.1
    #     results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_base_tune3_jjlr"
    # )
    # del results

    # hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
    #     finetune_layer=2,
    #     name="base-ls960",
    # )
    # model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
    #     specauc_start_epoch=0,
    #     label_target_size=vocab_size_without_blank,
    #     final_dropout=0.2,
    #     hubert_cfg=hubert_cfg_2,
    # )
    # train_args_hubert_adam_accum25_jjlr = {
    #     "config": {
    #         "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
    #         "learning_rates": list(np.linspace(7e-6, 7e-4, 220))
    #                           + list(np.linspace(7e-4, 7e-5, 220))
    #                           + list(np.linspace(7e-5, 1e-8, 60)),
    #         #############
    #         "batch_size": 180 * 16000,
    #         "max_seq_length": {"audio_features": 35 * 16000},
    #         "max_seqs": 3,
    #         "accum_grad_multiple_step": 25,
    #     },
    #     "debug": True,
    # }
    # eval_epochs = [100, 200, 250, 400, 500]
    # train_args = {
    #     **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
    #     "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
    #     "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    # }
    # results = {}
    # for lm_weight in [1.6, 1.8, 2.0, 2.2]:
    #     for prior_scale in [0.3, 0.5]:
    #         search_args = {
    #             **default_search_args,
    #             "lm_weight": lm_weight,
    #             "prior_scale": prior_scale,
    #             "beam_size_token": 128,
    #         }
    #         train_job, _, _, wer_values = run_exp(
    #             prefix_name
    #             + "conformer_0923/hubert_pretrain_v3_base_tune2_longer_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
    #             datasets=train_data,
    #             train_args=train_args,
    #             search_args=search_args,
    #             with_prior=True,
    #             eval_epochs=eval_epochs,
    #             num_epochs=500
    #         )
    #         train_job.rqmt["gpu_mem"] = 24
    #         results.update(wer_values)
    #         del wer_values
    # generate_report(  # 7.2
    #     results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_base_tune2_longer_jjlr"
    # )
    # del results

    hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=2,
        name="large-ls960-ft",
    )
    model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_2,
    )
    train_args_hubert_adam_accum25_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 110))
                              + list(np.linspace(7e-4, 7e-5, 110))
                              + list(np.linspace(7e-5, 1e-8, 30)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 3,
            "accum_grad_multiple_step": 25,
        },
        "debug": True,
    }
    eval_epochs = [200, 250]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/hubert_pretrain_v3_large960_tune2_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(  # 5.5
        results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_large960_tune2_jjlr"
    )
    del results

    train_args_hubert_adam_accum25_jjlr_longflat = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 130))
                              + list(np.linspace(7e-4, 7e-5, 230))
                              + list(np.linspace(7e-5, 1e-8, 140)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 3,
            "accum_grad_multiple_step": 25,
        },
        "debug": False,
    }
    eval_epochs = [250, 400, 450, 500]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr_longflat),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/hubert_pretrain_v3_large960_tune2_jjlr_longflat/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
                num_epochs=500
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(  # TODO 5.3 !!
        results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_large960_tune2_jjlr_longflat"
    )
    del results

    hubert_cfg_6 = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=6,
        name="large-ls960-ft",
    )
    model_config_hubert_6 = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_6,
    )
    eval_epochs = [250, 400, 450, 500]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr_longflat),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_6)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/hubert_pretrain_v3_large960_tune6_jjlr_longflat/lm%.1f_prior%.2f_bs1024_th14" % (lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
                num_epochs=500
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_large960_tune6_jjlr_longflat"
    )
    del results

    hubert_cfg_2 = hubert_pretrained_v1_cfg.HubertConfig(
        finetune_layer=2,
        name="large-ll60k",
    )
    model_config_hubert_2 = hubert_pretrained_v1_cfg.ModelConfig(
        specauc_start_epoch=0,
        label_target_size=vocab_size_without_blank,
        final_dropout=0.2,
        hubert_cfg=hubert_cfg_2,
    )
    train_args_hubert_adam_accum25_jjlr = {
        "config": {
            "optimizer": {"class": "adam", "epsilon": 1e-08, "betas": (0.9, 0.98)},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 130))
                              + list(np.linspace(7e-4, 7e-5, 230))
                              + list(np.linspace(7e-5, 1e-8, 140)),
            #############
            "batch_size": 180 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "max_seqs": 3,
            "accum_grad_multiple_step": 25,
        },
        "debug": True,
    }
    eval_epochs = [250, 300, 400, 500]
    train_args = {
        **copy.deepcopy(train_args_hubert_adam_accum25_jjlr),
        "network_module": "ctc.conformer_0923.hubert_pretrained_v3",
        "net_args": {"model_config_dict": asdict(model_config_hubert_2)},
    }
    results = {}
    for lm_weight in [1.6, 1.8, 2.0, 2.2]:
        for prior_scale in [0.3, 0.5]:
            search_args = {
                **default_search_args,
                "lm_weight": lm_weight,
                "prior_scale": prior_scale,
                "beam_size_token": 128,
            }
            train_job, _, _, wer_values = run_exp(
                prefix_name
                + "conformer_0923/hubert_pretrain_v3_large60k_tune2_jjlr/lm%.1f_prior%.2f_bs1024_th14" % (
                lm_weight, prior_scale),
                datasets=train_data,
                train_args=train_args,
                search_args=search_args,
                with_prior=True,
                eval_epochs=eval_epochs,
                num_epochs=500
            )
            train_job.rqmt["gpu_mem"] = 24
            results.update(wer_values)
            del wer_values
    generate_report(
        results=results, exp_name=prefix_name + "conformer_0923/hubert_pretrain_v3_large60k_tune2_jjlr"
    )
    del results

