from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from sisyphus import tk
from ...model_pipelines.common.report import register_memristor_report, register_recog_report
from . import training, recognition, finetuning
from ...baseline_rep.experiments.librispeech import training as baseline_training
from ...baseline_rep.experiments.librispeech import recognition as baseline_recognition

from .recognition import memristor as memristor_recognition

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

from ...model_pipelines.common.learning_rates import OCLRConfig
from .groups import LazyModels


def _build_models(lazy=False, selection=None):
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

    models = LazyModels(
        {
            "ffnn_transducer_bpe": lambda: baseline_training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
            "full_ctx_transducer_bpe": lambda: baseline_training.full_ctx_transducer_bpe.run(
                descriptor="full_ctx_transducer_bpe"
            ),
            "ffnn_transducer_bpe_125epochs": lambda: baseline_training.ffnn_transducer_bpe.run(
                descriptor="ffnn_transducer_bpe_125epochs",
                train_options=baseline_training.ffnn_transducer_bpe.get_train_options(num_epochs=125),
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
                qat_args={
                    **w8_a8_qat_config,
                    "version_control": "v2-qact-fix-12bs-no-joiner-output-bias",
                },
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_adc8": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_adc8", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_8prec_4range": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_8prec_4range", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_1prec_7range": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_1prec_7range", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_2prec_6range": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_2prec_6range", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_8range": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_4prec_8range", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_12range": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_4prec_12range", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_16range": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_4prec_16range", qat_args=w8_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8", qat_args=w4_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_l1_scale_0.1": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_l1_scale_0.1",
                qat_args=w4_a8_qat_config,
                train_options=replace(
                    training.qat_ffnn_transducer_bpe.get_train_options(),
                    joiner_l1_scale=0.1,
                ),
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_moving_average_0.01": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_moving_average_0.01",                 
                model_config=training.qat_ffnn_transducer_bpe.get_model_config(**w4_a8_qat_config, joiner_weight_moving_average=0.01),qat_args=w4_a8_qat_config
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_moving_average_0.01_l1_scale_0.1": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_moving_average_0.01_l1_scale_0.1",
                qat_args=w4_a8_qat_config,
                model_config=training.qat_ffnn_transducer_bpe.get_model_config(**w4_a8_qat_config, joiner_weight_moving_average=0.01),
                train_options=replace(
                    training.qat_ffnn_transducer_bpe.get_train_options(),
                    joiner_l1_scale=0.1,
                ),
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_outliers_scale_1.0_thresholds_0.5_0.5": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_outliers_scale_1.0_thresholds_0.5_0.5",
                qat_args=w4_a8_qat_config,
                train_options=replace(
                    training.qat_ffnn_transducer_bpe.get_train_options(),
                    joiner_outlier_scale=1.0,
                    joiner_outlier_thresholds=(0.5, 0.5),
                ),
            ),
            "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs": lambda: training.qat_ffnn_transducer_bpe.run(
                descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
                qat_args=w4_a8_qat_config,
                train_options=training.qat_ffnn_transducer_bpe.get_train_options(num_epochs=125),
            ),
            "qat_ffnn_transducer_aux_no_quant_bpe": lambda: training.qat_ffnn_transducer_aux_no_quant_bpe.run(
                descriptor="qat_ffnn_transducer_aux_no_quant_bpe", qat_args=w8_a8_qat_config
            ),
            "ffnn_transducer_qat_encoder": lambda: training.ffnn_transducer_qat_encoder_bpe.run(
                descriptor="ffnn_transducer_qat_encoder", qat_args=w8_a8_qat_config
            ),
            "ffnn_transducer_qat_encoder_125epochs": lambda: training.ffnn_transducer_qat_encoder_bpe.run(
                descriptor="ffnn_transducer_qat_encoder_125epochs",
                qat_args=w8_a8_qat_config,
                train_options=training.ffnn_transducer_qat_encoder_bpe.get_train_options2(num_epochs=125),
            ),
            "ffnn_transducer_qat_encoder_w4_a8": lambda: training.ffnn_transducer_qat_encoder_bpe.run(
                descriptor="ffnn_transducer_qat_encoder_w4_a8", qat_args=w4_a8_qat_config
            ),
            "ffnn_transducer_qat_encoder_v2_seed": lambda: training.ffnn_transducer_qat_encoder_bpe.run(
                descriptor="ffnn_transducer_qat_encoder_v2_seed", qat_args=w8_a8_qat_config, seed=383
            ),
            "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs": lambda: training.ffnn_transducer_qat_encoder_prediction_bpe.run(
                descriptor="ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs", qat_args=w8_a8_qat_config
            ),
            "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8": lambda: training.ffnn_transducer_qat_encoder_prediction_bpe.run(
                descriptor="ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8", qat_args=w4_a8_qat_config
            ),
            "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8_125epochs": lambda: training.ffnn_transducer_qat_encoder_prediction_bpe.run(
                descriptor="ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8_125epochs",
                qat_args=w4_a8_qat_config,
                train_options=training.ffnn_transducer_qat_encoder_prediction_bpe.get_train_options(num_epochs=125),
            ),
            "qat_ctc_bpe_param_sync": lambda: training.qat_ctc_bpe_param_sync.run(
                descriptor="qat_ctc_bpe_param_sync", qat_args=w8_a8_qat_config
            ),
            "qat_ctc_bpe_output_param_sync": lambda: training.qat_ctc_bpe_output_param_sync.run(
                descriptor="qat_ctc_bpe_output_param_sync", qat_args=w8_a8_qat_config
            ),
            "qat_ctc_bpe_w4_a8": lambda: training.qat_ctc_bpe_param_sync.run(
                descriptor="qat_ctc_bpe_w4_a8", qat_args=w4_a8_qat_config
            ),
            "qat_full_ctx_transducer_bpe_v2_low_bs": lambda: training.qat_full_ctx_transducer_bpe.run(
                descriptor="qat_full_ctx_transducer_bpe_v2_low_bs", qat_args=w8_a8_qat_config
            ),
            "qat_full_ctx_transducer_bpe_v2_low_bs_w4_a8": lambda: training.qat_full_ctx_transducer_bpe.run(
                descriptor="qat_full_ctx_transducer_bpe_v2_low_bs_w4_a8", qat_args=w4_a8_qat_config
            ),
            "full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs": lambda: training.full_ctx_transducer_qat_encoder_prediction_bpe.run(
                descriptor="full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs", qat_args=w8_a8_qat_config
            ),
            "full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8": lambda: training.full_ctx_transducer_qat_encoder_prediction_bpe.run(
                descriptor="full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8", qat_args=w4_a8_qat_config
            ),
            "ffnn_transducer_qat_encoder_bpe_param_sync": lambda: training.ffnn_transducer_qat_encoder_bpe_param_sync.run(
                descriptor="ffnn_transducer_qat_encoder_bpe_param_sync", qat_args=w8_a8_qat_config
            ),
            "full_ctx_transducer_qat_encoder_param_sync_bpe": lambda: training.full_ctx_transducer_qat_encoder_bpe.run(
                descriptor="full_ctx_transducer_qat_encoder_param_sync_bpe", qat_args=w8_a8_qat_config
            ),
            "full_ctx_transducer_qat_encoder_bpe": lambda: training.full_ctx_transducer_qat_encoder_bpe_low_bs.run(
                descriptor="full_ctx_transducer_qat_encoder_bpe", qat_args=w8_a8_qat_config
            ),
            "full_ctx_transducer_qat_encoder_bpe_w4_a8": lambda: training.full_ctx_transducer_qat_encoder_bpe_low_bs.run(
                descriptor="full_ctx_transducer_qat_encoder_bpe_w4_a8", qat_args=w4_a8_qat_config
            ),
        }
    )
    if not lazy:
        models = dict(models)

    hilmes_lr_config = OCLRConfig(
        init_lr=7e-06,
        peak_lr=5e-04,
        decayed_lr=1e-07,
        final_lr=1e-07,
        inc_epochs=500 // 2,
        dec_epochs=500 // 2,
        final_epochs=0,
    )

    finetune_factories = {
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe",
            qat_args=w8_a8_qat_config,
        ),
        "ffnn_transducer_qat_encoder_pred_w8_a8_bpe___ffnn_transducer_bpe": lambda: finetuning.ffnn_transducer_qat_encoder_pred_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_pred_bpe___ffnn_transducer_bpe",
            qat_args=w8_a8_qat_config,
        ),
        "ffnn_transducer_qat_encoder_pred_w4_a8_bpe___ffnn_transducer_bpe": lambda: finetuning.ffnn_transducer_qat_encoder_pred_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_pred_bpe_w4_a8_bpe___ffnn_transducer_bpe",
            qat_args=w4_a8_qat_config,
        ),
        "qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe": lambda: finetuning.qat_ffnn_transducer_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe",
            qat_args=w8_a8_qat_config,
        ),
        "qat_ffnn_transducer_w4_a8_bpe___ffnn_transducer_bpe": lambda: finetuning.qat_ffnn_transducer_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="qat_ffnn_transducer_w4_a8_bpe___ffnn_transducer_bpe",
            qat_args=w4_a8_qat_config,
        ),
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep": lambda: finetuning.qat_ffnn_transducer_bpe__qat_ffnn_transducer_bpe.run(
            base_model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            descriptor="qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
            qat_args=w4_a8_qat_config,
            train_options=finetuning.qat_ffnn_transducer_bpe__qat_ffnn_transducer_bpe.get_train_options(
                num_epochs=25, gpu_mem_rqmt=48
            ),
        ),
        "ffnn_transducer_qat_encoder_bpe_w8a8___ffnn_transducer_qat_encoder_bpe_w8a8": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_qat_encoder_bpe.run(
            base_model=models["ffnn_transducer_qat_encoder"],
            descriptor="ffnn_transducer_qat_encoder_bpe_w8a8___ffnn_transducer_qat_encoder_bpe_w8a8",
            train_options=finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_qat_encoder_bpe.get_train_options(
                num_epochs=25, gpu_mem_rqmt=24
            ),
            qat_args=w8_a8_qat_config,
        ),
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe_50ep": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe_50ep",
            qat_args=w4_a8_qat_config,
            train_options=finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.get_train_options(
                num_epochs=50, gpu_mem_rqmt=24
            ),
        ),
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe",
            qat_args=w4_a8_qat_config,
        ),
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_w4_a8": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_w4_a8",
            qat_args=w4_a8_qat_config,
        ),
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_v2": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_v2",
            qat_args=w8_a8_qat_config,
            seed=383,
        ),
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr": lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.run(
            base_model=models["ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr",
            qat_args=w8_a8_qat_config,
            train_options=finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.get_train_options(
                learning_rate_config=hilmes_lr_config, gpu_mem_rqmt=48
            ),
        ),
        "full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe": lambda: finetuning.full_ctx_transducer_qat_encoder_bpe__full_ctx_transducer_bpe.run(
            base_model=models["full_ctx_transducer_bpe"],
            descriptor="full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe",
            qat_args=w8_a8_qat_config,
            train_options=finetuning.full_ctx_transducer_qat_encoder_bpe__full_ctx_transducer_bpe.get_train_options(
                gpu_mem_rqmt=48
            ),
        ),
        "full_ctx_transducer_qat_encoder_pred_bpe_w8a8___full_ctx_transducer_bpe": lambda: finetuning.full_ctx_transducer_qat_encoder_pred_bpe__full_ctx_transducer_bpe.run(
            base_model=models["full_ctx_transducer_bpe"],
            descriptor="full_ctx_transducer_qat_encoder_pred_bpe_w8a8___full_ctx_transducer_bpe",
            qat_args=w8_a8_qat_config,
            train_options=finetuning.full_ctx_transducer_qat_encoder_pred_bpe__full_ctx_transducer_bpe.get_train_options(
                gpu_mem_rqmt=48
            ),
        ),
    }
    finetune_factories["ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_qat_encoder_w8_a8"] = (
        lambda: finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_qat_encoder_bpe.run(
            base_model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe"],
            descriptor="ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_qat_encoder_w8_a8",
            qat_args=w4_a8_qat_config,
            train_options=finetuning.ffnn_transducer_qat_encoder_bpe__ffnn_transducer_bpe.get_train_options(
                num_epochs=50, gpu_mem_rqmt=24
            ),
        )
    )

    finetunes = LazyModels(finetune_factories)
    if not lazy:
        finetunes = dict(finetunes)
    if selection is not None:
        for registry in (models, finetunes):
            for key in registry:
                if selection(key):
                    registry[key]
    return models, finetunes


def run_adapter(filename, selection=None):
    models, _ = _build_models(lazy=True)
    recog_results = []

    def add(key, build):
        if selection is None or selection(key):
            recog_results.extend(build())

    joint_adc = DacAdcHardwareSettings(8, 2, 6, 0.6, 8020.0)

    def run(base_model, descriptor, *, decoder_cycles, joint_adc=None):
        hardware_settings = {}
        if joint_adc is not None:
            hardware_settings = dict(
                joint_l1_converter_hardware_settings=joint_adc,
                joint_l2_converter_hardware_settings=joint_adc,
            )

        adapter_model = finetuning.memristor_adapter_bpe.run(
            descriptor=descriptor,
            base_model=base_model,
            **hardware_settings,
        )
        base_config = adapter_model.model_config.base_config
        recog = memristor_recognition.qat_ffnn_transducer_bpe
        _, results = recog.run(
            model=base_model,
            adapter_model=adapter_model,
            variants=[
                recog.default_offline_lexfree_recog_variant(num_shards=10, resumable=True, cycles=3),
                recog.mbs1024_offline_lexfree_recog_variant(num_shards=5, resumable=True, cycles=decoder_cycles),
                recog.ilm04_elm05_tree_4gram_recog_variant(num_shards=5, resumable=True, cycles=decoder_cycles),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=base_config.converter_hardware_settings,
            pos_enc_converter_hardware_settings=base_config.pos_enc_converter_hardware_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            **hardware_settings,
        )
        return results

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: run(
            base_model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            descriptor="memristor_output_adapter_bpe_pilot",
            decoder_cycles=1,
        ),
    )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: run(
            base_model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            descriptor="memristor_output_adapter_bpe_joint_adc_2prec_6range",
            decoder_cycles=3,
            joint_adc=joint_adc,
        ),
    )

    if recog_results:
        report = register_recog_report(recog_results, filename=filename)
        if selection is not None:
            report.write_report()
    return models, recog_results


def run_all(filename, selection=None):
    models, finetunes = _build_models(lazy=selection is not None, selection=selection)
    recog_results = []

    def add(key, build):
        if selection is None or selection(key):
            recog_results.extend(build())

    # recog_results.extend(
    #     baseline_recognition.ffnn_transducer_bpe.run(
    #         model=models["ffnn_transducer_bpe_highbs_200epochs"], corpora=["dev-other"]
    #     )
    # )

    add(
        "ffnn_transducer_bpe_125epochs",
        lambda: baseline_recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe_125epochs"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_v2_seed",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_v2_seed"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_qat_encoder_w8_a8",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_qat_encoder_w8_a8"],
            corpora=["dev-other"],
        ),
    )
    # recog_results.extend(
    #     recognition.ffnn_transducer_qat_encoder_bpe.run(
    #         model=models["ffnn_transducer_qat_encoder_bpe_highbs_200epochs"], corpora=["dev-other"]
    #     )
    # )
    add(
        "ffnn_transducer_qat_encoder_w4_a8",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_w4_a8"],
            corpora=["dev-other"],
            variants=[
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_lexfree_recog_variant(num_shards=1),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_tree_recog_variant(num_shards=1),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_lexfree_lstm_recog_variant(num_shards=1),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_tree_4gram_recog_variant(num_shards=1),
                # recognition.ffnn_transducer_qat_encoder_bpe.default_offline_tree_lstm_4gram_recog_variant(num_shards=1),
            ],
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_w4_a8",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_w4_a8"],
            corpora=["dev-other"],
            variants=[
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_lexfree_recog_variant(num_shards=3),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_tree_recog_variant(num_shards=5),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_lexfree_lstm_recog_variant(num_shards=2),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_tree_4gram_recog_variant(num_shards=4),
                recognition.ffnn_transducer_qat_encoder_bpe.default_offline_tree_lstm_4gram_recog_variant(
                    num_shards=10
                ),
            ],
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_125epochs",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_125epochs"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_param_sync",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_bpe_param_sync"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8",
        lambda: recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8_125epochs",
        lambda: recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8_125epochs"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs",
        lambda: recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_pred_w8_a8_bpe___ffnn_transducer_bpe",
        lambda: recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_pred_w8_a8_bpe___ffnn_transducer_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            corpora=["dev-other"],
            variants=[
                recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=-0.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=-0.2),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=-0.1),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=0.1),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=0.2),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=0.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=1.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=1.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=2.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=2.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(blank_penalty=3.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=-0.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=-0.2),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=-0.1),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=0.1),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=0.2),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=0.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=1.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=1.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=2.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=2.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_recog_variant(blank_penalty=3.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=-0.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=-0.2),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=-0.1),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=0.1),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=0.2),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=0.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=1.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=1.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=2.0),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=2.5),
                recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_tree_4gram_recog_variant(blank_penalty=3.0),
            ],
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs"],
            corpora=["dev-other"],
        ),
    )
    add(
        "qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_ffnn_transducer_w4_a8_bpe___ffnn_transducer_bpe",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___ffnn_transducer_bpe"],
            corpora=["dev-other"],
            variants=[recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(use_cache=True)],
        ),
    )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"], corpora=["dev-other"]
        ),
    )
    for key in (
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_l1_scale_0.1",
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_joiner_outliers_scale_1.0_thresholds_0.5_0.5",
    ):
        add(
            key,
            lambda key=key: recognition.qat_ffnn_transducer_bpe.run(model=models[key], corpora=["dev-other"]),
        )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            corpora=["dev-other"],
            variants=[
                recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(),
                recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(
                    hash_salt="predictor_linear1_debug_v1"
                ),
                recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(
                    hash_salt="predictor_linear1_debug_v1", use_cache=True
                ),
            ],
        ),
    )
    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            corpora=["dev-other"],
        ),
    )

    add(
        "qat_ffnn_transducer_aux_no_quant_bpe",
        lambda: recognition.qat_ffnn_transducer_aux_no_quant_bpe.run(
            model=models["qat_ffnn_transducer_aux_no_quant_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_param_sync_bpe",
        lambda: recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_param_sync_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe",
        lambda: recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=finetunes["full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs",
        lambda: recognition.full_ctx_transducer_qat_encoder_prediction_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs"], corpora=["dev-other"]
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_pred_bpe_w8a8___full_ctx_transducer_bpe",
        lambda: recognition.full_ctx_transducer_qat_encoder_prediction_bpe.run(
            model=finetunes["full_ctx_transducer_qat_encoder_pred_bpe_w8a8___full_ctx_transducer_bpe"],
            corpora=["dev-other"],
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_pred_bpe_w8a8___full_ctx_transducer_bpe",
        lambda: recognition.full_ctx_transducer_qat_encoder_prediction_bpe.run(
            model=finetunes["full_ctx_transducer_qat_encoder_pred_bpe_w8a8___full_ctx_transducer_bpe"],
            corpora=["dev-other"],
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe",
        lambda: recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=finetunes["full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe_50ep",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe_50ep"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"], corpora=["dev-other"]
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_bpe",
        lambda: recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_bpe_w4_a8",
        lambda: recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_bpe_w4_a8"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_ctc_bpe_param_sync",
        lambda: recognition.qat_ctc_bpe_param_sync.run(model=models["qat_ctc_bpe_param_sync"], corpora=["dev-other"]),
    )
    add(
        "qat_ctc_bpe_param_sync",
        lambda: recognition.qat_ctc_bpe_param_sync.run(
            model=models["qat_ctc_bpe_param_sync"], corpora=["dev-other"], batched_decoder=True
        ),
    )
    add(
        "qat_ctc_bpe_w4_a8",
        lambda: recognition.qat_ctc_bpe_param_sync.run(model=models["qat_ctc_bpe_w4_a8"], corpora=["dev-other"]),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_w8a8___ffnn_transducer_qat_encoder_bpe_w8a8",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe_w8a8___ffnn_transducer_qat_encoder_bpe_w8a8"],
            corpora=["dev-other"],
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_v2",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_v2"], corpora=["dev-other"]
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe",
        lambda: recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe"], corpora=["dev-other"]
        ),
    )
    add(
        "qat_ctc_bpe_output_param_sync",
        lambda: recognition.qat_ctc_output_bpe.run(
            model=models["qat_ctc_bpe_output_param_sync"], corpora=["dev-other"]
        ),
    )

    if recog_results:
        report = register_recog_report(recog_results, filename=filename)
        if selection is not None:
            report.write_report()
    return models, recog_results


async def run_mem(filename, selection=None):
    models, finetunes = _build_models(lazy=selection is not None, selection=selection)

    converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_ideal = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=5476.0,
    )
    pos_enc_converter_hardware_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    pos_enc_converter_hardware_settings_ideal = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=5476.0,
    )
    converter_hardware_settings_8out = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=8,
        output_range_bits=8,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_1prec_7range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_1prec_7range_ideal = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=5476.0,
    )
    converter_hardware_settings_2prec_6range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=2,
        output_range_bits=6,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    converter_hardware_settings_2prec_6range_ideal = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=2,
        output_range_bits=6,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=5476.0,
    )

    converter_hardware_settings_3prec_5range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=3,
        output_range_bits=5,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_5prec_3range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=5,
        output_range_bits=3,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    converter_hardware_settings_8prec_4range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=8,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_8prec_8range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=8,
        output_range_bits=8,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_4prec_8range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=8,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_4prec_8range_ideal = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=8,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=5476.0,
    )
    converter_hardware_settings_4prec_12range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=12,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    converter_hardware_settings_4prec_16range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=16,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    correction_settings = None
    recog_results = []
    max_runs = 5

    def add(key, build):
        if selection is None or selection(key):
            recog_results.append(build())

    # recog_results.append(
    #     memristor_recognition.qat_ctc_bpe_param_sync.run(
    #         model=models["qat_ctc_bpe_param_sync"],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         correction_settings=correction_settings,
    #         max_runs=max_runs,
    #         memristor_prior=False,
    #         batched_decoder=True,
    #     )
    # )
    add(
        "qat_ctc_bpe_output_param_sync",
        lambda: memristor_recognition.qat_ctc_output_bpe_param_sync.run(
            model=models["qat_ctc_bpe_output_param_sync"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=max_runs,
            memristor_prior=False,
            batched_decoder=True,
        ),
    )
    # recog_results.append(
    #     memristor_recognition.qat_ctc_bpe_param_sync.run(
    #         model=models["qat_ctc_bpe_w4_a8"],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         correction_settings=correction_settings,
    #         max_runs=max_runs,
    #         memristor_prior=False,
    #         batched_decoder=True,
    #     )
    # )
    add(
        "ffnn_transducer_qat_encoder",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder"],
            corpora=["dev-other"],
            variants=memristor_recognition.ffnn_transducer_qat_encoder_bpe.param_sweep_tree_lstm_recog_variants(),
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_qat_encoder_w8_a8",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_qat_encoder_w8_a8"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe_50ep",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe_w4_a8___ffnn_transducer_bpe_50ep"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    # recog_results.append(
    #     memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
    #         model=models["ffnn_transducer_qat_encoder_v2_seed"],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         correction_settings=correction_settings,
    #         max_runs=3,
    #         batched_decoder=True,
    #     )
    # )
    add(
        "ffnn_transducer_qat_encoder_125epochs",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_125epochs"],
            variants=[
                memristor_recognition.ffnn_transducer_qat_encoder_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.ffnn_transducer_qat_encoder_bpe.mbs1024_offline_tree_recog_variant(),
            ]
            + memristor_recognition.ffnn_transducer_qat_encoder_bpe.param_sweep_tree_lstm_recog_variants()
            + memristor_recognition.ffnn_transducer_qat_encoder_bpe.param_sweep_tree_4gram_recog_variants(),
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    # recog_results.append(
    #     memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
    #         model=finetunes["ffnn_transducer_qat_encoder_bpe_w8a8___ffnn_transducer_qat_encoder_bpe_w8a8"],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         correction_settings=correction_settings,
    #         max_runs=3,
    #         batched_decoder=True,
    #     )
    # )
    add(
        "ffnn_transducer_qat_encoder_w4_a8",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=models["ffnn_transducer_qat_encoder_w4_a8"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs"],
            corpora=["dev-other"],
            variants=memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.param_sweep_tree_4gram_recog_variants(),
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            batch_size_seconds=360,
        ),
    )

    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8_125epochs",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8_125epochs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.default_offline_lexfree_recog_variant(
                    num_shards=2, resumable=True
                ),
                memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=3, resumable=True
                ),
            ]
            + memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.param_sweep_tree_4gram_recog_variants(
                num_shards=3, resumable=True
            ),
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            use_cache=True,
            batched_decoder=True,
            batch_size_seconds=360,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_pred_w8_a8_bpe___ffnn_transducer_bpe",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_pred_w8_a8_bpe___ffnn_transducer_bpe"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            batch_size_seconds=360,
            use_cache=True,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs_w4_a8"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            batch_size_seconds=360,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.run(
            model=models["ffnn_transducer_qat_encoder_prediction_bpe_v2_lowbs"],
            corpora=["dev-other"],
            variants=memristor_recognition.ffnn_transducer_qat_encoder_prediction_bpe.param_sweep_tree_4gram_recog_variants(),
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            batch_size_seconds=360,
            use_cache=True,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            batch_size_seconds=360,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_v2",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_v2"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            batch_size_seconds=360,
        ),
    )
    add(
        "ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_w4_a8",
        lambda: memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
            model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_w4_a8"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            batch_size_seconds=360,
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs",
        lambda: memristor_recognition.full_ctx_transducer_qat_encoder_pred_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs"],
            variants=[
                memristor_recognition.full_ctx_transducer_qat_encoder_pred_bpe.default_offline_lexfree_recog_variant()
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )
    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_full_ctx_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_16range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )
    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_full_ctx_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
                memristor_recognition.qat_full_ctx_transducer_bpe.ilm05_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_full_ctx_transducer_bpe_v2_low_bs",
        lambda: memristor_recognition.qat_full_ctx_transducer_bpe.run(
            model=models["qat_full_ctx_transducer_bpe_v2_low_bs"],
            variants=[
                memristor_recognition.qat_full_ctx_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_16range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    # recog_results.append(
    #     memristor_recognition.ffnn_transducer_qat_encoder_bpe.run(
    #         model=finetunes["ffnn_transducer_qat_encoder_bpe___ffnn_transducer_bpe_hilmeslr"],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         correction_settings=correction_settings,
    #         max_runs=3,
    #         batched_decoder=True,
    #         batch_size_seconds=360,
    #     )
    # )
    # recog_results.append(
    #     memristor_recognition.full_ctx_transducer_qat_encoder_bpe.run(
    #         model=models["full_ctx_transducer_qat_encoder_param_sync_bpe"],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         correction_settings=correction_settings,
    #         max_runs=3,
    #         batched_decoder=True,
    #     )
    # )
    add(
        "full_ctx_transducer_qat_encoder_bpe",
        lambda: memristor_recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_bpe"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.full_ctx_transducer_qat_encoder_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.full_ctx_transducer_qat_encoder_bpe.mbs1024_offline_tree_recog_variant(),
            ]
            + memristor_recognition.full_ctx_transducer_qat_encoder_bpe.param_sweep_tree_4gram_recog_variants()
            + memristor_recognition.full_ctx_transducer_qat_encoder_bpe.param_sweep_tree_lstm_recog_variants(),
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_bpe_w4_a8",
        lambda: memristor_recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=models["full_ctx_transducer_qat_encoder_bpe_w4_a8"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.full_ctx_transducer_qat_encoder_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.full_ctx_transducer_qat_encoder_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.full_ctx_transducer_qat_encoder_bpe.ilm05_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )
    add(
        "full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe",
        lambda: memristor_recognition.full_ctx_transducer_qat_encoder_bpe.run(
            model=finetunes["full_ctx_transducer_qat_encoder_bpe_w8a8___full_ctx_transducer_bpe"],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_ideal,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_no_joiner_output_bias"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_3prec_5range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_3prec_5range,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=0.52, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=10, resumable=True
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=0.52, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=10, resumable=True
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_ideal,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings_ideal,
            correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_ideal,
            joint_l2_converter_hardware_settings=converter_hardware_settings_ideal,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_8range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )
    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___qat_ffnn_transducer_w8_a8_bpe_25ep"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_ideal,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=3,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=2,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range_ideal,
            joint_l2_converter_hardware_settings=converter_hardware_settings,
            correction_settings=correction_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            joint_correction_layer=1,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_8range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_8prec_8range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    # Matched J2-only weight regrid ablation; keep the existing run above unchanged.
    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
            regrid_joint_net_7=True,
        ),
    )

    add(
        "qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w8_a8_bpe___ffnn_transducer_bpe"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_ideal,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings_ideal,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range_ideal,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range_ideal,
            correction_settings=CycleCorrectionSettings(
                num_cycles=None,
                test_input_value=None,
                relative_deviation=None,
                ideal_programming=True,
            ),
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.nbs_mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=False, cycles=1, nbest_size=20, blank_penalty=1.44
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=0.82, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=0.9, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_ilm04_elm05_tree_4gram_recog_variant(
                    blank_penalty=0.2, cycles=3, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_ilm04_elm05_tree_4gram_recog_variant(
                    blank_penalty=0.33, cycles=3, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_ilm04_elm05_tree_4gram_recog_variant(
                    blank_penalty=0.5, cycles=3, num_shards=10, resumable=True
                ),
            ]
            + [
                memristor_recognition.qat_ffnn_transducer_bpe.pre_softmax_blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=blank_penalty, cycles=1, num_shards=5, resumable=True
                )
                for blank_penalty in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]
            ]
            + [
                memristor_recognition.qat_ffnn_transducer_bpe.pre_softmax_blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=blank_penalty, cycles=1, num_shards=5, resumable=True
                )
                for blank_penalty in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]
            ]
            + [
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_ilm04_elm05_tree_4gram_recog_variant(
                    blank_penalty=blank_penalty, cycles=1, num_shards=10, resumable=True
                )
                for blank_penalty in [1.0, 1.5, 2.0]
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    # add("qat_ffnn_transducer_full_quant_v2_lowbs",
    #     lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
    #         model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
    #         variants=[
    #             memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
    #                 num_shards=10, resumable=True, cycles=3
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
    #                 num_shards=10, resumable=True, cycles=3
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
    #                 num_shards=10, resumable=True, cycles=3
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
    #                 blank_penalty=0.5, cycles=1, num_shards=5, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
    #                 blank_penalty=1.0, cycles=1, num_shards=5, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
    #                 blank_penalty=1.5, cycles=1, num_shards=5, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
    #                 blank_penalty=2.0, cycles=1, num_shards=5, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
    #                 blank_penalty=2.5, cycles=1, num_shards=5, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
    #                 blank_penalty=0.5, cycles=1, num_shards=10, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
    #                 blank_penalty=1.0, cycles=1, num_shards=10, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
    #                 blank_penalty=1.5, cycles=1, num_shards=10, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
    #                 blank_penalty=2.0, cycles=1, num_shards=10, resumable=True
    #             ),
    #             memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
    #                 blank_penalty=2.5, cycles=1, num_shards=10, resumable=True
    #             ),
    #         ],
    #         corpora=["dev-other"],
    #         converter_hardware_settings=converter_hardware_settings,
    #         pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
    #         joint_l1_converter_hardware_settings=converter_hardware_settings,
    #         joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
    #         correction_settings=correction_settings,
    #         max_runs=3,
    #         batched_decoder=True,
    #         use_cache=True,
    #         regrid_embedding=False,
    #     )
    # )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.5, cycles=3, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, resumable=True, cycles=3
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            regrid_embedding=False,
        ),
    )

    add(
        "qat_ffnn_transducer_w4_a8_bpe___ffnn_transducer_bpe",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=finetunes["qat_ffnn_transducer_w4_a8_bpe___ffnn_transducer_bpe"],
            variants=[memristor_recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant()],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=1,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(
                    hash_salt="predictor_linear1_debug_v1"
                ),
            ]
            + memristor_recognition.qat_ffnn_transducer_bpe.default_recog_variants(),
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(
                    hash_salt="predictor_linear1_debug_v1"
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=1,
            batched_decoder=True,
            use_cache=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=3, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_net_7=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.joint_debug_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.0
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.41, cycles=3, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.5, cycles=3, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=3.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=3.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=4.0, cycles=1, num_shards=5
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=8.0, cycles=1, num_shards=5
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=7.3, cycles=1, num_shards=5
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=3.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=3.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=4.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=3, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, cycles=3, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                    num_shards=10, cycles=3, resumable=True
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_1prec_7range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=1, resumable=True
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=3, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=5, cycles=3, resumable=True
                ),
                # memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                #     num_shards=5, cycles=3, resumable=True
                # ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.0, cycles=3, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.44, cycles=3, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=1.5, cycles=3, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=3.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=3.5, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                    blank_penalty=4.0, cycles=1, num_shards=5, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=0.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=1.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=2.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=3.0, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=3.5, cycles=1, num_shards=10, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_mbs1024_offline_tree_recog_variant(
                    blank_penalty=4.0, cycles=1, num_shards=10, resumable=True
                ),
            ]
            + memristor_recognition.qat_ffnn_transducer_bpe.param_sweep_tree_4gram_recog_variants(
                num_shards=5, cycles=1, resumable=True
            ),
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=3, resumable=True
                ),
                # memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                #     num_shards=5, cycles=1, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(
                #     num_shards=5, cycles=1, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=1.0, cycles=1, num_shards=5, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=1.5, cycles=1, num_shards=5, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=2.0, cycles=1, num_shards=5, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=2.5, cycles=1, num_shards=5, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=3.0, cycles=1, num_shards=5, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=3.5, cycles=1, num_shards=5, resumable=True
                # ),
                # memristor_recognition.qat_ffnn_transducer_bpe.blank_penalty_offline_lexfree_recog_variant(
                #     blank_penalty=4.0, cycles=1, num_shards=5
                # ),
            ],
            # + [
            #     memristor_recognition.qat_ffnn_transducer_bpe.pre_softmax_blank_penalty_mbs1024_offline_tree_recog_variant(
            #         blank_penalty=blank_penalty, cycles=1, num_shards=5, resumable=True
            #     )
            #     for blank_penalty in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
            # ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_3prec_5range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=1, resumable=True
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_5prec_3range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(num_shards=5),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_lexfree_recog_variant(
                    num_shards=5, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_adc8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_adc8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_8out,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_prediction_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.param_sweep_tree_4gram_recog_variants()[0],
                memristor_recognition.qat_ffnn_transducer_bpe.param_sweep_tree_4gram_recog_variants()[2],
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=False,
            offload_prediction_network=False,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=3, resumable=True
                ),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_prediction_network=True,
            offload_joint_network=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=1, resumable=True
                ),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_prediction_network=True,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8_125epochs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=5, cycles=1, resumable=True
                ),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_prediction_network=False,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                # memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            output_quantizer=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_adc8",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_adc8"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_8out,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_8prec_4range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_8prec_4range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_8prec_4range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_2prec_6range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_2prec_6range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(num_shards=5),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_2prec_6range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_1prec_7range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_1prec_7range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(
                    num_shards=10, resumable=True, cycles=1
                ),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_1prec_7range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_8range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_4prec_8range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_8range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=False,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_8range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_4prec_8range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(
                    num_shards=10, cycles=1, resumable=True
                ),
                memristor_recognition.qat_ffnn_transducer_bpe.param_sweep_tree_4gram_recog_variants(
                    num_shards=10, cycles=1, resumable=True
                )[0],
                memristor_recognition.qat_ffnn_transducer_bpe.param_sweep_tree_4gram_recog_variants(
                    num_shards=10, cycles=1, resumable=True
                )[2],
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_8range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_12range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_4prec_12range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(num_shards=5),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_12range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs_4prec_16range",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs_4prec_16range"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(num_shards=5),
            ],
            corpora=["dev-other"],
            converter_hardware_settings=converter_hardware_settings_4prec_16range,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_network=True,
        ),
    )

    add(
        "qat_ffnn_transducer_full_quant_v2_lowbs",
        lambda: memristor_recognition.qat_ffnn_transducer_bpe.run(
            model=models["qat_ffnn_transducer_full_quant_v2_lowbs"],
            corpora=["dev-other"],
            variants=[
                memristor_recognition.qat_ffnn_transducer_bpe.default_offline_lexfree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.mbs1024_offline_tree_recog_variant(),
                memristor_recognition.qat_ffnn_transducer_bpe.ilm04_elm05_tree_4gram_recog_variant(),
            ],
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            correction_settings=correction_settings,
            max_runs=3,
            batched_decoder=True,
            use_cache=True,
            offload_joint_net_7=False,
        ),
    )

    memristor_recog_results = [mr for r in recog_results for mr in r[0]]
    all_recog_results = [rr for r in recog_results for rr in r[1]]
    report_base, report_ext = filename.rsplit(".", 1)
    memristor_report = None
    recog_report = None

    def refresh_reports(*_, write=True):
        nonlocal memristor_report, recog_report
        if memristor_recog_results:
            if memristor_report is None:
                memristor_report = register_memristor_report(
                    memristor_recog_results, filename=f"{report_base}_aggregated_report.{report_ext}", live=True
                )
            if write:
                memristor_report.write_report()
        if all_recog_results:
            if recog_report is None:
                recog_report = register_recog_report(all_recog_results, filename=filename, live=True)
            if write:
                recog_report.write_report()

    refresh_reports(write=selection is not None)

    async def run_lazy_sweep(sweep):
        sweep_memristor_results = []
        sweep_recog_results = []
        async for memristor_results, results in sweep:
            memristor_recog_results.extend(memristor_results)
            all_recog_results.extend(results)
            sweep_memristor_results.extend(memristor_results)
            sweep_recog_results.extend(results)
            refresh_reports()  # Include rows immediately, even before scoring finishes.
            for result in (*memristor_results, *results):
                tk.register_callback(refresh_reports, result)
        return sweep_memristor_results, sweep_recog_results

    # Each sweep can wait for its own selection scores without delaying the
    # independent reports or other sweeps.
    lazy_sweeps = (
        [
            memristor_recognition.lazy_sweep.run(
                recognition=memristor_recognition.full_ctx_transducer_qat_encoder_pred_bpe,
                model=models["full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs"],
                regular_variants=[
                    memristor_recognition.full_ctx_transducer_qat_encoder_pred_bpe.default_offline_lexfree_recog_variant(),
                    memristor_recognition.full_ctx_transducer_qat_encoder_pred_bpe.mbs1024_offline_tree_recog_variant(),
                ],
                variants=memristor_recognition.full_ctx_transducer_qat_encoder_pred_bpe.param_sweep_tree_4gram_recog_variants(
                    num_shards=10
                ),
                corpora=["dev-other"],
                selection_corpus="dev-other",
                converter_hardware_settings=converter_hardware_settings,
                pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
                correction_settings=correction_settings,
                top_k=2,
                max_runs=3,
                batched_decoder=True,
                use_cache=True,
            ),
        ]
        if selection is None or selection("full_ctx_transducer_qat_encoder_prediction_bpe_v2_lowbs")
        else []
    )
    recog_results.extend(await tk.async_gather(*(run_lazy_sweep(sweep) for sweep in lazy_sweeps)))
    return models, recog_results
