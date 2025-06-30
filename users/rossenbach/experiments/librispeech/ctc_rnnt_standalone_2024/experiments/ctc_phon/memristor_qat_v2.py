from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search
from ...report import tune_and_evalue_report

from .tune_eval import tune_and_evaluate_helper, eval_model



from sisyphus import tk
import numpy as np
from i6_core.report.report import GenerateReportStringJob, MailJob, _Report_Type
import copy
from typing import Dict
from i6_core.util import instanciate_delayed


def calc_stat(ls):
    avrg = np.average([float(x[1]) for x in ls])
    min = np.min([float(x[1]) for x in ls])
    max = np.max([float(x[1]) for x in ls])
    median = np.median([float(x[1]) for x in ls])
    std = np.std([float(x[1]) for x in ls])
    ex_str = f"Avrg: {avrg}, Min {min}, Max {max}, Median {median}, Std {std},    ({avrg},{min},{max},{median},{std}) Num Values: {len(ls)}"
    return ex_str


def baseline_report_format(report: _Report_Type) -> str:
    """
    Example report format for the baseline , extra ls can be set in order to filter out certain results
    :param report:
    :return:
    """
    extra_ls = ["quantize_static"]
    sets = set()
    for recog in report:
        sets.add(recog.split("/")[-1])
    out = [
        (" ".join(recog.split("/")[3:]), str(report[recog]))
        for recog in report
        if not any(extra in recog for extra in extra_ls)
    ]
    out = sorted(out, key=lambda x: float(x[1]))
    best_ls = [out[0]]
    if any("cycle" in x[0] for x in best_ls):
        ex_str = calc_stat(out)
        out.insert(0, ("Cycle Statistics: ", ex_str))
    for dataset in sets:
        for extra in extra_ls:
            if extra == "quantize_static":
                tmp = {recog: report[recog] for recog in report if extra in recog and dataset in recog}
                iters = set()
                for recog in tmp:
                    x = recog.split("/")
                    for sub in x:
                        if "samples" in sub:
                            iters.add(sub[len("samples_") :])
                for samples in iters:
                    out2 = [
                        (" ".join(recog.split("/")[3:]), str(report[recog]))
                        for recog in report
                        if f"samples_{samples}/" in recog and dataset in recog
                    ]
                    out2 = sorted(out2, key=lambda x: float(x[1]))
                    if len(out2) > 0:
                        ex_str = calc_stat(out2)
                        out.append(("", ""))
                        out.append((dataset + " " + extra + f"_samples_{samples}", ex_str))
                        # out.extend(out2[:3])
                        # out.extend(out2[-3:])
                        out.extend(out2)
                        best_ls.append(out2[0])
            else:
                out2 = [
                    (" ".join(recog.split("/")[3:]), str(report[recog]))
                    for recog in report
                    if extra in recog and dataset in recog
                ]
                out2 = sorted(out2, key=lambda x: float(x[1]))
                if len(out2) > 0:
                    out.append(("", ""))
                    out.append((dataset + " " + extra, ""))
                    out.extend(out2)
                    best_ls.append(out2[0])
    best_ls = sorted(best_ls, key=lambda x: float(x[1]))
    best_ls += [("Base Results", "")]
    out = best_ls + out
    out.insert(0, ("Best Results", ""))
    return "\n".join([f"{pair[0]}:  {str(pair[1])}" for pair in out])


def generate_report(results, exp_name, report_template=baseline_report_format):
    report = GenerateReportStringJob(report_values=results, report_template=report_template)
    report.add_alias(f"report/report/{exp_name}")
    mail = MailJob(report.out_report, send_contents=True, subject=exp_name)
    mail.add_alias(f"report/mail/{exp_name}")
    tk.register_output("mail/" + exp_name, mail.out_status)


def build_memristor_base_report(report: Dict):
    from math import ceil

    report = copy.deepcopy(report)
    baselines = report.pop("baselines")
    best_baselines = {}
    for exp, dic in baselines.items():
        instanciate_delayed(dic)
        if all(dic.values()):
            best = min(dic, key=dic.get)
            best_baselines[exp + "/" + best] = dic[best]
        else:
            best_baselines[exp] = "None"
    line = []
    best_dc = {}
    bits = {1.5, 2, 3, 4, 5, 6, 8}
    for exp, best in best_baselines.items():
        line.append(f"{exp.split('/')[5]}: {best}   {' '.join(exp.split('/')[10:])}")
    for exp, dic in report.items():
        tmp = {}
        for bit in bits:
            for name in dic:
                if f"weight_{bit}" in name or (bit == ceil(bit) and f"weight_{int(bit)}" in name):
                    tmp[name] = dic[name]
            if all(tmp.values()):
                instanciate_delayed(tmp)
                best = min(tmp, key=tmp.get)
                best_dc[exp + "/" + best] = tmp[best]
            else:
                best_dc[exp] = "None"
    for exp, value in best_dc.items():
        if isinstance(exp, float):
            line.append(f"{exp}: {value}")
        else:
            line.append(f"{' '.join(exp.split('/')[9:12])}: {value}   {' '.join(exp.split('/')[12:])}")
    return "\n".join(line)


def eow_phon_ls960_0425_memristor():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_memristor_v2"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings,
    )

    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
            dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    def tune_and_evaluate_helper(training_name, asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module="ctc.decoder.flashlight_ctc_v1"):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name, forward_config={}, asr_model=asr_model, decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn
            )
            report_values[key] = wers[training_name + "/" + key]

        tune_and_evalue_report(
            training_name=training_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values
        )

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig as DecoderConfigMemristor

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    decoder_config_memristor = DecoderConfigMemristor(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    decoder_config_no_memristor = copy.deepcopy(decoder_config_memristor)
    decoder_config_no_memristor.turn_off_quant = "leave_as_is"

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )

    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {
            "class": "radam",
            "epsilon": 1e-12,
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True,
        },
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
                          + list(np.linspace(5e-4, 5e-5, 480))
                          + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    # Same with conv first
    network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args_conv_first = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module_conv_first,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }

    name = ".512dim_sub4_48gbgpu_100eps_radam_bs360_sp"
    training_name = prefix_name + "/" + network_module_conv_first + name
    train_job = training(training_name, train_data, train_args_conv_first, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    train_job.hold()
    train_job.move_to_hpc = True

    asr_model = prepare_asr_model(
        training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data,
        get_specific_checkpoint=1000
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])


    # Normal QAT
    base_network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor",
        moving_average=None,
        weight_bit_prec=8,
        activation_bit_prec=8,
        quantize_output=False,
        extra_act_quant=False,
        quantize_bias=None,
        observer_only_in_train=False,
    )


    train_config_24gbgpu = copy.deepcopy(train_config_24gbgpu_amp)
    train_config_24gbgpu.pop("torch_amp_options")
    # Same with conv first
    train_args_base_qat = {
        "config": train_config_24gbgpu,
        "network_module": base_network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }

    name = ".512dim_sub4_48gbgpu_100eps_radam_bs360_sp"
    training_name = prefix_name + "/" + base_network_module_v4 + name
    train_job = training(training_name, train_data, train_args_base_qat, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    #train_job.hold()
    #train_job.move_to_hpc = True

    asr_model = prepare_asr_model(
        training_name, train_job, train_args_base_qat, with_prior=True, datasets=train_data,
        get_specific_checkpoint=1000
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])


    # Memristor QAT trianing

    network_module_mem_v5 = "ctc.qat_0711.memristor_v5"
    from ...pytorch_networks.ctc.qat_0711.memristor_v5_cfg import QuantModelTrainConfigV5 as MemristorModelTrainConfigV5
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    qat_report = {}

    for activation_bit in [8]:
        for weight_bit in [3, 4, 5, 6, 8]:
            for num_cycles in range(10):
                prior_train_dac_settings = DacAdcHardwareSettings(
                    input_bits=0,
                    output_precision_bits=0,
                    output_range_bits=0,
                    hardware_input_vmax=0.6,
                    hardware_output_current_scaling=8020.0,
                )
                recog_dac_settings = DacAdcHardwareSettings(
                    input_bits=8,
                    output_precision_bits=4,
                    output_range_bits=4,
                    hardware_input_vmax=0.6,
                    hardware_output_current_scaling=8020.0,
                )
                model_config = MemristorModelTrainConfigV5(
                    feature_extraction_config=fe_config,
                    frontend_config=frontend_config,
                    specaug_config=specaug_config_full,
                    label_target_size=vocab_size_without_blank,
                    conformer_size=512,
                    num_layers=12,
                    num_heads=8,
                    ff_dim=2048,
                    att_weights_dropout=0.1,
                    conv_dropout=0.1,
                    ff_dropout=0.1,
                    mhsa_dropout=0.1,
                    conv_kernel_size=31,
                    final_dropout=0.1,
                    specauc_start_epoch=1,
                    weight_quant_dtype="qint8",
                    weight_quant_method="per_tensor_symmetric",
                    activation_quant_dtype="qint8",
                    activation_quant_method="per_tensor_symmetric",
                    dot_quant_dtype="qint8",
                    dot_quant_method="per_tensor_symmetric",
                    Av_quant_dtype="qint8",
                    Av_quant_method="per_tensor_symmetric",
                    moving_average=None,
                    weight_bit_prec=weight_bit,
                    activation_bit_prec=activation_bit,
                    quantize_output=False,
                    converter_hardware_settings=prior_train_dac_settings,
                    quant_in_linear=True,
                    num_cycles=0,
                )


                model_config_recog = copy.deepcopy(model_config)
                model_config_recog.converter_hardware_settings = recog_dac_settings
                model_config_recog.num_cycles = num_cycles

                train_args = {
                    "config": train_config_24gbgpu,
                    "network_module": network_module_mem_v5,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                training_name = prefix_name + "/" + network_module_mem_v5 + f"_{weight_bit}_{activation_bit}_{num_cycles}"
                train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
                train_job.rqmt["gpu_mem"] = 48
                #train_job.hold()
                #train_job.move_to_hpc = True

                prior_args = {
                    "config": train_config_24gbgpu,
                    "network_module": network_module_mem_v5,
                    "net_args": {"model_config_dict": asdict(model_config)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }

                train_args_recog = {
                    "config": train_config_24gbgpu,
                    "network_module": network_module_mem_v5,
                    "net_args": {"model_config_dict": asdict(model_config_recog)},
                    "debug": False,
                    "post_config": {"num_workers_per_gpu": 8},
                    "use_speed_perturbation": True,
                }
                results = {}
                results = eval_model(
                    training_name=training_name,
                    train_job=train_job,
                    train_args=train_args_recog,
                    train_data=train_data,
                    decoder_config=decoder_config_memristor,
                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                    result_dict=results,
                    decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                    prior_scales=[0.3],  # TODO 0.7
                    lm_scales=[2.0],
                    use_gpu=True,
                    import_memristor=True,
                    extra_forward_config={
                        "batch_size": 200 * 16000,
                    },
                    run_best_4=False,
                    run_best=False,
                    prior_args=prior_args,
                )
                qat_report[training_name] = copy.deepcopy(results)


                if weight_bit == 3:
                    results = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args_recog,
                        train_data=train_data,
                        decoder_config=decoder_config_memristor,
                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                        result_dict=results,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=[0.3],  # TODO 0.7
                        lm_scales=[2.0],
                        use_gpu=True,
                        import_memristor=True,
                        extra_forward_config={
                            "batch_size": 200 * 16000,
                        },
                        run_best_4=False,
                        run_best=True,
                        prior_args=prior_args,
                    )

                if num_cycles == 0:
                    results = eval_model(
                        training_name=training_name + "_no_memristor",
                        train_job=train_job,
                        train_args=train_args_recog,
                        train_data=train_data,
                        decoder_config=decoder_config_no_memristor,
                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                        result_dict=results,
                        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                        prior_scales=[0.2, 0.3, 0.4],  # TODO 0.7
                        lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                        use_gpu=True,
                        import_memristor=True,
                        extra_forward_config={
                            "batch_size": 200 * 16000,
                        },
                        run_best_4=False,
                        run_best=True if weight_bit == 3 else False,
                        prior_args=prior_args,
                    )


    from .tune_eval import build_qat_report
    from functools import partial
    tk.register_report("reports/memristor", partial(build_qat_report, qat_report), required=qat_report)

    # special experiments
    for activation_bit in [8]:
        for weight_bit in [4, 6]:
            prior_train_dac_settings = DacAdcHardwareSettings(
                input_bits=0,
                output_precision_bits=0,
                output_range_bits=0,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            recog_dac_settings = DacAdcHardwareSettings(
                input_bits=8,
                output_precision_bits=10,
                output_range_bits=10,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_config = MemristorModelTrainConfigV5(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
                label_target_size=vocab_size_without_blank,
                conformer_size=512,
                num_layers=12,
                num_heads=8,
                ff_dim=2048,
                att_weights_dropout=0.1,
                conv_dropout=0.1,
                ff_dropout=0.1,
                mhsa_dropout=0.1,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=1,
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor_symmetric",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor_symmetric",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor_symmetric",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor_symmetric",
                moving_average=None,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                quantize_output=False,
                converter_hardware_settings=prior_train_dac_settings,
                quant_in_linear=True,
                num_cycles=0,
            )

            model_config_recog = copy.deepcopy(model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings

            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v5,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            training_name = prefix_name + "/" + network_module_mem_v5 + f"_ADC10_{weight_bit}_{activation_bit}"
            train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48
            train_job.hold()
            train_job.move_to_hpc = True

            prior_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v5,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            train_args_recog = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v5,
                "net_args": {"model_config_dict": asdict(model_config_recog)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            results = {}
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=decoder_config_memristor,
                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.3],  # TODO 0.7
                lm_scales=[2.0],
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={
                    "batch_size": 200 * 16000,
                },
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
            )

    return
    # Again with precision/range 8/8
    for activation_bit in [8]:
        # for weight_bit in [3, 4, 5]:
        for weight_bit in [4]:
            prior_train_dac_settings = DacAdcHardwareSettings(
                input_bits=0,
                output_precision_bits=0,
                output_range_bits=0,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            recog_dac_settings = DacAdcHardwareSettings(
                input_bits=8,
                output_precision_bits=8,
                output_range_bits=8,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_config = MemristorModelTrainConfigV5(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
                label_target_size=vocab_size_without_blank,
                conformer_size=512,
                num_layers=12,
                num_heads=8,
                ff_dim=2048,
                att_weights_dropout=0.1,
                conv_dropout=0.1,
                ff_dropout=0.1,
                mhsa_dropout=0.1,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=1,
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor_symmetric",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor_symmetric",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor_symmetric",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor_symmetric",
                moving_average=None,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                quantize_output=False,
                converter_hardware_settings=prior_train_dac_settings,
                quant_in_linear=True,
                num_cycles=0,
            )

            model_config_recog = copy.deepcopy(model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings

            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v5,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            training_name = prefix_name + "/" + network_module_mem_v5 + f"_{weight_bit}_{activation_bit}"
            train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48

            prior_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v5,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            train_args_recog = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v5,
                "net_args": {"model_config_dict": asdict(model_config_recog)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            results = {}
            results = eval_model(
                training_name=training_name + "_dac_8_8",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=decoder_config_memristor,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.3],  # TODO 0.7
                lm_scales=[2.0],
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={
                    "batch_size": 200 * 16000,
                },
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
            )

            #tune_and_evaluate_helper(training_name + "_no_memristor", asr_model, decoder_config_no_memristor, lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
            #                         prior_scales=[0.2, 0.3, 0.4], decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc")
            results = eval_model(
                training_name=training_name + "_no_memristor",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=decoder_config_no_memristor,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.3],  # TODO 0.7
                lm_scales=[2.0],
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={
                    "batch_size": 200 * 16000,
                },
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
            )


    return
    ################ no 384 settings for now

    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {
            "class": "radam",
            "epsilon": 1e-16,
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True,
        },
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
                          + list(np.linspace(5e-4, 5e-5, 480))
                          + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }
    train_config_24gbgpu = copy.deepcopy(train_config_24gbgpu_amp)
    train_config_24gbgpu.pop("torch_amp_options")

    # Same with conv first
    network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args_conv_first = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module_conv_first,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }

    name = ".384dim_sub4_48gbgpu_100eps_radam_bs300_sp"
    training_name = prefix_name + "/" + network_module_conv_first + name
    train_job = training(training_name, train_data, train_args_conv_first, num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data,
        get_specific_checkpoint=1000
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0],
                             prior_scales=[0.2, 0.3, 0.4])

    # Normal QAT
    #network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    #from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    #model_config = QuantModelTrainConfigV4(
    #    feature_extraction_config=fe_config,
    #    frontend_config=frontend_config,
    #    specaug_config=specaug_config_full,
    #    label_target_size=vocab_size_without_blank,
    #    conformer_size=,
    #    num_layers=12,
    #    num_heads=8,
    #    ff_dim=2048,
    #    att_weights_dropout=0.1,
    #    conv_dropout=0.1,
    #    ff_dropout=0.1,
    #    mhsa_dropout=0.1,
    #    conv_kernel_size=31,
    #    final_dropout=0.1,
    #    specauc_start_epoch=1,
    #    weight_quant_dtype="qint8",
    #    weight_quant_method="per_tensor",
    #    activation_quant_dtype="qint8",
    #    activation_quant_method="per_tensor",
    #    dot_quant_dtype="qint8",
    #    dot_quant_method="per_tensor",
    #    Av_quant_dtype="qint8",
    #    Av_quant_method="per_tensor",
    #    moving_average=None,
    #    weight_bit_prec=8,
    #    activation_bit_prec=8,
    #    quantize_output=False,
    #    extra_act_quant=False,
    #    quantize_bias=None,
    #    observer_only_in_train=False,
    #)

    #train_config_24gbgpu = copy.deepcopy(train_config_24gbgpu_amp)
    #train_config_24gbgpu.pop("torch_amp_options")
    ## Same with conv first
    #train_args_base_qat = {
    #    "config": train_config_24gbgpu,
    #    "network_module": network_module_v4,
    #    "net_args": {"model_config_dict": asdict(model_config)},
    #    "debug": False,
    #    "use_speed_perturbation": True,
    #    "post_config": {"num_workers_per_gpu": 8}
    #}

    #name = ".512dim_sub4_48gbgpu_100eps_radam_bs300_sp"
    #training_name = prefix_name + "/" + network_module_v4 + name
    #train_job = training(training_name, train_data, train_args_base_qat, num_epochs=1000, **default_returnn)
    #train_job.rqmt["gpu_mem"] = 48
    #asr_model = prepare_asr_model(
    #    training_name, train_job, train_args_base_qat, with_prior=True, datasets=train_data,
    #    get_specific_checkpoint=1000
    #)
    #tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0],
    #                         prior_scales=[0.2, 0.3, 0.4])

    # Memristor QAT trianing

    network_module_mem_v4 = "ctc.qat_0711.memristor_v4"
    from ...pytorch_networks.ctc.qat_0711.memristor_v4_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV4
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    for activation_bit in [8]:
        for weight_bit in [3, 4, 5]:
            prior_train_dac_settings = DacAdcHardwareSettings(
                input_bits=0,
                output_precision_bits=0,
                output_range_bits=0,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            recog_dac_settings = DacAdcHardwareSettings(
                input_bits=8,
                output_precision_bits=4,
                output_range_bits=4,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_config = MemristorModelTrainConfigV4(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
                label_target_size=vocab_size_without_blank,
                conformer_size=384,
                num_layers=12,
                num_heads=4,
                ff_dim=1536,
                att_weights_dropout=0.1,
                conv_dropout=0.1,
                ff_dropout=0.1,
                mhsa_dropout=0.1,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=1,
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor_symmetric",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor_symmetric",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor_symmetric",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor_symmetric",
                moving_average=None,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                quantize_output=False,
                converter_hardware_settings=prior_train_dac_settings,
                quant_in_linear=True,
                num_cycles=0,
            )

            model_config_recog = copy.deepcopy(model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings

            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v4,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            training_name = prefix_name + "/384dim." + network_module_mem_v4 + f"_{weight_bit}_{activation_bit}"
            train_job = training(training_name, train_data, train_args, num_epochs=1000, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48

            prior_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v4,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            train_args_recog = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v4,
                "net_args": {"model_config_dict": asdict(model_config_recog)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            results = {}
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=decoder_config_memristor,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.3],  # TODO 0.7
                lm_scales=[2.0],
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={
                    "batch_size": 200 * 16000,
                },
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
            )
            
            
    ################ 384 50eps settings

    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {
            "class": "radam",
            "epsilon": 1e-16,
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True,
        },
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240))
                          + list(np.linspace(5e-4, 5e-5, 240))
                          + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }
    train_config_24gbgpu = copy.deepcopy(train_config_24gbgpu_amp)
    train_config_24gbgpu.pop("torch_amp_options")

    # Same with conv first
    network_module_conv_first = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first"
    train_args_conv_first = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module_conv_first,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }

    name = ".384dim_sub4_48gbgpu_50eps_radam_bs300_sp"
    training_name = prefix_name + "/" + network_module_conv_first + name
    train_job = training(training_name, train_data, train_args_conv_first, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_conv_first, with_prior=True, datasets=train_data,
        get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0],
                             prior_scales=[0.2, 0.3, 0.4])

    # Normal QAT
    #network_module_v4 = "ctc.qat_0711.baseline_qat_v4"
    #from ...pytorch_networks.ctc.qat_0711.baseline_qat_v4_cfg import QuantModelTrainConfigV4

    #model_config = QuantModelTrainConfigV4(
    #    feature_extraction_config=fe_config,
    #    frontend_config=frontend_config,
    #    specaug_config=specaug_config_full,
    #    label_target_size=vocab_size_without_blank,
    #    conformer_size=,
    #    num_layers=12,
    #    num_heads=8,
    #    ff_dim=2048,
    #    att_weights_dropout=0.1,
    #    conv_dropout=0.1,
    #    ff_dropout=0.1,
    #    mhsa_dropout=0.1,
    #    conv_kernel_size=31,
    #    final_dropout=0.1,
    #    specauc_start_epoch=1,
    #    weight_quant_dtype="qint8",
    #    weight_quant_method="per_tensor",
    #    activation_quant_dtype="qint8",
    #    activation_quant_method="per_tensor",
    #    dot_quant_dtype="qint8",
    #    dot_quant_method="per_tensor",
    #    Av_quant_dtype="qint8",
    #    Av_quant_method="per_tensor",
    #    moving_average=None,
    #    weight_bit_prec=8,
    #    activation_bit_prec=8,
    #    quantize_output=False,
    #    extra_act_quant=False,
    #    quantize_bias=None,
    #    observer_only_in_train=False,
    #)

    #train_config_24gbgpu = copy.deepcopy(train_config_24gbgpu_amp)
    #train_config_24gbgpu.pop("torch_amp_options")
    ## Same with conv first
    #train_args_base_qat = {
    #    "config": train_config_24gbgpu,
    #    "network_module": network_module_v4,
    #    "net_args": {"model_config_dict": asdict(model_config)},
    #    "debug": False,
    #    "use_speed_perturbation": True,
    #    "post_config": {"num_workers_per_gpu": 8}
    #}

    #name = ".512dim_sub4_48gbgpu_100eps_radam_bs300_sp"
    #training_name = prefix_name + "/" + network_module_v4 + name
    #train_job = training(training_name, train_data, train_args_base_qat, num_epochs=1000, **default_returnn)
    #train_job.rqmt["gpu_mem"] = 48
    #asr_model = prepare_asr_model(
    #    training_name, train_job, train_args_base_qat, with_prior=True, datasets=train_data,
    #    get_specific_checkpoint=1000
    #)
    #tune_and_evaluate_helper(training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0],
    #                         prior_scales=[0.2, 0.3, 0.4])

    # Memristor QAT trianing

    network_module_mem_v4 = "ctc.qat_0711.memristor_v4"
    from ...pytorch_networks.ctc.qat_0711.memristor_v4_cfg import QuantModelTrainConfigV4 as MemristorModelTrainConfigV4
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

    for activation_bit in [8]:
        for weight_bit in [3, 4, 5]:
            prior_train_dac_settings = DacAdcHardwareSettings(
                input_bits=0,
                output_precision_bits=0,
                output_range_bits=0,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            recog_dac_settings = DacAdcHardwareSettings(
                input_bits=8,
                output_precision_bits=4,
                output_range_bits=4,
                hardware_input_vmax=0.6,
                hardware_output_current_scaling=8020.0,
            )
            model_config = MemristorModelTrainConfigV4(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
                label_target_size=vocab_size_without_blank,
                conformer_size=384,
                num_layers=12,
                num_heads=4,
                ff_dim=1536,
                att_weights_dropout=0.1,
                conv_dropout=0.1,
                ff_dropout=0.1,
                mhsa_dropout=0.1,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=1,
                weight_quant_dtype="qint8",
                weight_quant_method="per_tensor_symmetric",
                activation_quant_dtype="qint8",
                activation_quant_method="per_tensor_symmetric",
                dot_quant_dtype="qint8",
                dot_quant_method="per_tensor_symmetric",
                Av_quant_dtype="qint8",
                Av_quant_method="per_tensor_symmetric",
                moving_average=None,
                weight_bit_prec=weight_bit,
                activation_bit_prec=activation_bit,
                quantize_output=False,
                converter_hardware_settings=prior_train_dac_settings,
                quant_in_linear=True,
                num_cycles=0,
            )

            model_config_recog = copy.deepcopy(model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings

            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v4,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            training_name = prefix_name + "/384dim_50eps." + network_module_mem_v4 + f"_{weight_bit}_{activation_bit}"
            train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
            train_job.rqmt["gpu_mem"] = 48

            prior_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v4,
                "net_args": {"model_config_dict": asdict(model_config)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }

            train_args_recog = {
                "config": train_config_24gbgpu,
                "network_module": network_module_mem_v4,
                "net_args": {"model_config_dict": asdict(model_config_recog)},
                "debug": False,
                "post_config": {"num_workers_per_gpu": 8},
                "use_speed_perturbation": True,
            }
            results = {}
            results = eval_model(
                training_name=training_name,
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data,
                decoder_config=decoder_config_memristor,
                dev_dataset_tuples=dev_dataset_tuples,
                result_dict=results,
                decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
                prior_scales=[0.3],  # TODO 0.7
                lm_scales=[2.0],
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={
                    "batch_size": 200 * 16000,
                },
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
            )