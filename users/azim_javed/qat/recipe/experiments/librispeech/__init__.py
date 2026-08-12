from typing import Dict, List, Optional, Tuple

from sisyphus import tk
from ...model_pipelines.common.report import register_memristor_report, register_recog_report
from . import training, recognition, finetuning
from ...baseline_rep.experiments.librispeech import training as baseline_training

from .recognition import memristor as memristor_recognition

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings

from ...model_pipelines.common.learning_rates import OCLRConfig


def run_all(filename):
    w8_a8_qat_config = dict(
        weight_bit_prec=8,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    w4_a8_qat_config = dict(
        weight_bit_prec=4,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    models = {
        "ffnn_transducer_bpe": baseline_training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_bpe_highbs_200epochs": baseline_training.ffnn_transducer_bpe_param_sync.run(
            descriptor="ffnn_transducer_bpe_highbs_200epochs",
            train_options=baseline_training.ffnn_transducer_bpe_param_sync.get_train_options(num_epochs=200),
        ),
        "qat_ffnn_transducer_full_quant_v2_lowbs": training.qat_ffnn_transducer_bpe.run(
            descriptor="qat_ffnn_transducer_full_quant_v2_lowbs", qat_args=w8_a8_qat_config
        ),
        "qat_ffnn_transducer_aux_no_quant_bpe": training.qat_ffnn_transducer_aux_no_quant_bpe.run(
            descriptor="qat_ffnn_transducer_aux_no_quant_bpe", qat_args=w8_a8_qat_config
        ),
        "ffnn_transducer_qat_encoder": training.ffnn_transducer_qat_encoder_bpe.run(
            descriptor="ffnn_transducer_qat_encoder", qat_args=w8_a8_qat_config
        ),
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs": training.ffnn_transducer_qat_encoder_prediction_bpe.run(
            descriptor="ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs", qat_args=w8_a8_qat_config
        ),
        "qat_ctc_bpe_param_sync": training.qat_ctc_bpe_param_sync.run(
            descriptor="qat_ctc_bpe_param_sync", qat_args=w8_a8_qat_config
        ),
        "qat_ctc_bpe_output_param_sync": training.qat_ctc_bpe_output_param_sync.run(
            descriptor="qat_ctc_bpe_output_param_sync", qat_args=w8_a8_qat_config
        ),
        "qat_ctc_bpe_w4_a8": training.qat_ctc_bpe_param_sync.run(
            descriptor="qat_ctc_bpe_w4_a8", qat_args=w4_a8_qat_config
        ),
        "qat_full_ctx_transducer_bpe_v2_low_bs": training.qat_full_ctx_transducer_bpe.run(
            descriptor="qat_full_ctx_transducer_bpe_v2_low_bs", qat_args=w8_a8_qat_config
        ),
        "full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs": training.full_ctx_transducer_qat_encoder_prediction_bpe.run(
            descriptor="full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs", qat_args=w8_a8_qat_config
        ),
        "ffnn_transducer_qat_encoder_bpe_param_sync": training.ffnn_transducer_qat_encoder_bpe_param_sync.run(
            descriptor="ffnn_transducer_qat_encoder_bpe_param_sync", qat_args=w8_a8_qat_config
        ),
        "ffnn_transducer_qat_encoder_bpe_highbs_150epochs": training.ffnn_transducer_qat_encoder_bpe_param_sync.run(
            descriptor="ffnn_transducer_qat_encoder_bpe_highbs_200epochs",
            qat_args=w8_a8_qat_config,
            train_options=training.ffnn_transducer_qat_encoder_bpe_param_sync.get_train_options(num_epochs=200),
        ),
        "full_ctx_transducer_qat_encoder_param_sync_bpe": training.full_ctx_transducer_qat_encoder_bpe.run(
            descriptor="full_ctx_transducer_qat_encoder_param_sync_bpe", qat_args=w8_a8_qat_config
        ),
        "full_ctx_transducer_qat_encoder_bpe": training.full_ctx_transducer_qat_encoder_bpe_low_bs.run(
            descriptor="full_ctx_transducer_qat_encoder_bpe", qat_args=w8_a8_qat_config
        ),
    }

    hilmes_lr_config = OCLRConfig(
        init_lr=7e-06,
        peak_lr=5e-04,
        decayed_lr=1e-07,
        final_lr=1e-07,
        inc_epochs=500 // 2,
        dec_epochs=500 // 2,
        final_epochs=0,
    )

    finetunes = {
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe": finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe",
            qat_args=w8_a8_qat_config,
        ),
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr": finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr",
            qat_args=w8_a8_qat_config,
            train_options=finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.get_train_options(
                learning_rate_config=hilmes_lr_config, gpu_mem_rqmt=24
            ),
        ),
    }
    recog_results = []
    recog_results.extend(
        recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder"], corpora=["dev-other"]
        )
    )
    recog_results.extend(
        recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_bpe_param_sync"], corpora=["dev-other"]
        )
    )
    recog_results.extend(
        recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs"], corpora=["dev-other"]
        )
    )
    recog_results.extend(
        recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_param_sync_bpe"], corpora=["dev-other"]
        )
    )
    recog_results.extend(
        recognition.qat_ctc_bpe_param_sync.run(model=models["qat_ctc_bpe_param_sync"], corpora=["dev-other"])
    )
    recog_results.extend(
        recognition.qat_ctc_bpe_param_sync.run(
            model=models["qat_ctc_bpe_param_sync"], corpora=["dev-other"], batched_decoder=True
        )
    )
    recog_results.extend(
        recognition.qat_ctc_bpe_param_sync.run(model=models["qat_ctc_bpe_w4_a8"], corpora=["dev-other"])
    )
    recog_results.extend(
        recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe"], corpora=["dev-other"]
        )
    )
    recog_results.extend(
        recognition.qat_ctc_output_bpe.run(model=models["qat_ctc_bpe_output_param_sync"], corpora=["dev-other"])
    )
    register_recog_report(recog_results, filename=filename)
    return models, recog_results


def run_test(filename):
    baseline_qat_config = dict(
        weight_bit_prec=8,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    models = {
        # "qat_ffnn_transducer_full_quant": training.qat_ffnn_transducer_bpe.run(
        #     descriptor="qat_ffnn_transducer_full_quant", qat_args=baseline_qat_config
        # ),
        "ffnn_transducer_qat_encoder": training.ffnn_transducer_qat_encoder_bpe.run(
            descriptor="ffnn_transducer_qat_encoder", qat_args=baseline_qat_config
        ),
        # "qat_ctc_bpe": training.qat_ctc_bpe.run(descriptor="qat_ctc_bpe", qat_args=baseline_qat_config),
    }
    recog_results = []
    # recog_results.extend(recognition.ffnn_transducer_qat_encoder_bpe.run(model=models["ffnn_transducer_qat_encoder"]))
    # recog_results.extend(recognition.qat_ctc_bpe.run(model=models["qat_ctc_bpe"]))
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer_full_quant"]))
    register_recog_report(recog_results, filename=filename)
    return models, recog_results


def run_debug(filename):
    w8_a8_qat_config = dict(
        weight_bit_prec=8,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    w4_a8_qat_config = dict(
        weight_bit_prec=4,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )

    models = {
        "qat_ctc_bpe_w4_a8": training.qat_ctc_bpe_param_sync.run(
            descriptor="qat_ctc_bpe_w4_a8", qat_args=w4_a8_qat_config
        ),
        "qat_ctc_bpe_param_sync": training.qat_ctc_bpe_param_sync.run(
            descriptor="qat_ctc_bpe_param_sync", qat_args=w8_a8_qat_config
        ),
    }

    converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    pos_enc_converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    correction_settings = None
    recog_results = []
    max_runs = 5

    recog_results.append(
        memristor_recognition.qat_ctc_bpe_param_sync.run(
            model=models["qat_ctc_bpe_param_sync"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=max_runs,
            memristor_prior=False,
            batched_decoder=True,
        )
    )
    recog_results.append(
        memristor_recognition.qat_ctc_bpe_param_sync.run(
            model=models["qat_ctc_bpe_w4_a8"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=max_runs,
            memristor_prior=False,
            batched_decoder=True,
        )
    )
    memristor_recog_results = [mr for r in recog_results for mr in r[0]]
    all_recog_results = [rr for r in recog_results for rr in r[1]]
    register_memristor_report(
        memristor_recog_results, filename=f"{filename.rsplit('.', 1)[0]}_memristor_report.{filename.rsplit('.', 1)[1]}"
    )
    register_recog_report(all_recog_results, filename=filename)
    return models, recog_results
