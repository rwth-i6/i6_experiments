from typing import Dict, List, Optional, Tuple

from sisyphus import tk
from ...model_pipelines.common.report import register_recog_report
from . import recognition, training


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
        "qat_ffnn_transducer_full_quant": training.qat_ffnn_transducer_bpe.run(
            descriptor="qat_ffnn_transducer_full_quant", qat_args=w8_a8_qat_config
        ),
        "ffnn_transducer_qat_encoder": training.ffnn_transducer_qat_encoder_bpe.run(
            descriptor="ffnn_transducer_qat_encoder", qat_args=w8_a8_qat_config
        ),
        "qat_ctc_bpe_param_sync": training.qat_ctc_bpe_param_sync.run(
            descriptor="qat_ctc_bpe_param_sync", qat_args=w8_a8_qat_config
        ),
        "qat_ctc_bpe_w4_a8": training.qat_ctc_bpe_param_sync.run(
            descriptor="qat_ctc_bpe_w4_a8", qat_args=w4_a8_qat_config
        ),
        "ffnn_transducer_qat_encoder_bpe_param_sync": training.ffnn_transducer_qat_encoder_bpe_param_sync.run(
            descriptor="ffnn_transducer_qat_encoder_bpe_param_sync", qat_args=w8_a8_qat_config
        ),
        "full_ctx_transducer_qat_encoder_bpe": training.full_ctx_transducer_qat_encoder_bpe.run(
            descriptor="full_ctx_transducer_qat_encoder_bpe", qat_args=w8_a8_qat_config
        )
    }
    recog_results = []
    recog_results.extend(recognition.ffnn_transducer_qat_encoder_bpe.run(model=models["ffnn_transducer_qat_encoder"]))
    recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer_full_quant"]))
    recog_results.extend(recognition.qat_ctc_bpe_param_sync.run(model=models["qat_ctc_bpe_param_sync"], corpora=["dev-other"]))
    recog_results.extend(recognition.qat_ctc_bpe_param_sync.run(model=models["qat_ctc_bpe_w4_a8"], corpora=["dev-other"]))
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe_param_sync.run(model=models["qat_ffnn_transducer_bpe_param_sync"]))
    
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
    recog_results.extend(recognition.ffnn_transducer_qat_encoder_bpe.run(model=models["ffnn_transducer_qat_encoder"]))
    # recog_results.extend(recognition.qat_ctc_bpe.run(model=models["qat_ctc_bpe"]))
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer_full_quant"]))
    register_recog_report(recog_results, filename=filename)
    return models, recog_results


def run_debug(filename):
    from ...model_pipelines.common.train import TrainedModel
    from synaptogen_ml.memristor_modules import DacAdcHardwareSettings

    baseline_qat_config = dict(
        weight_bit_prec=8,
        activation_bit_prec=8,
        weight_dropout=0.0,
        weight_pruning_config=None,
    )
    from sisyphus import tk

    # ffnnt_qat_encoder_trained_model = TrainedModel(
    #     descriptor="ffnn_transducer_qat_encoder",
    #     model_config=training.ffnn_transducer_qat_encoder_bpe.get_model_config(**baseline_qat_config),
    #     checkpoints={
    #         196: tk.Path("/u/azim.javed/experiments/training/qat/schkpt/ffnn_transducer_qat_encoder_chkpt.pt")
    #     },
    # )
    qat_ctc_trained_model = TrainedModel(
        descriptor="qat_ctc_bpe_debug",
        model_config=training.qat_ctc_bpe.get_model_config(**baseline_qat_config),
        checkpoints={
            2000: tk.Path("/u/azim.javed/experiments/training/qat/output/training/qat_ctc_bpe/final_checkpoint")
        },
    )
    # qat_ffnn_transducer_trained_model = TrainedModel(
    #     descriptor="qat_ffnn_transducer_full_quant",
    #     model_config=training.qat_ffnn_transducer_bpe.get_model_config(**baseline_qat_config),
    #     checkpoints={115: tk.Path("/u/azim.javed/experiments/training/qat/schkpt/qat_ffnn_transducer_chkpt.pt")},
    # )
    models = {
        # "ffnn_transducer_qat_encoder": ffnnt_qat_encoder_trained_model,
        # "qat_ctc_bpe": qat_ctc_trained_model,
        # "qat_ffnn_transducer_full_quant": qat_ffnn_transducer_trained_model,
        # "full_ctx_transducer_qat_encoder_bpe": training.full_ctx_transducer_qat_encoder_bpe.run(
        #     descriptor="full_ctx_transducer_qat_encoder_bpe", qat_args=baseline_qat_config
        # )
    }
    from .training import qat_ctc_bpe_param_sync as training_qat_ctc_bpe_param_sync
    from .recognition.memristor import qat_ctc_bpe_param_sync as recognition_qat_ctc_bpe_param_sync
    # qat_ctc_real_model = training_qat_ctc_bpe_param_sync.run(
    #         descriptor="qat_ctc_bpe_param_sync", qat_args=baseline_qat_config
    #     )
    
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
    num_cycles = 0
    recog_results = []
    recog_results.extend(recognition_qat_ctc_bpe_param_sync.run(model=qat_ctc_trained_model, corpora=["dev-other"], converter_hardware_settings=converter_hardware_settings, pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings, correction_settings=correction_settings, num_cycles=num_cycles))
    # recog_results.extend(recognition.ffnn_transducer_qat_encoder_bpe.run(model=models["ffnn_transducer_qat_encoder"]))
    # recog_results.extend(recognition.qat_ctc_bpe.run(model=models["qat_ctc_bpe"]))
    # recog_results.extend(recognition.qat_ffnn_transducer_bpe.run(model=models["qat_ffnn_transducer_full_quant"]))
    register_recog_report(recog_results, filename=filename)
    return models, recog_results


def run_hilmes(filename):
    from i6_core.returnn import PtCheckpoint
    from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
    from i6_experiments.users.hilmes.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.qat_0711.memristor_v8_cfg import (
        QuantModelTrainConfigV8,
        VGG4LayerActFrontendV1Config_mod,
        SpecaugConfig,
        ConformerPosEmbConfig,
    )
    from ...model_pipelines.common.train import TrainedModel
    from sisyphus import tk

    try:
        from torch_memristor.memristor_modules import DacAdcHardwareSettings
    except ModuleNotFoundError:
        from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings

    # reconstruct the training config
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

    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )

    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    prior_train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    model_config = QuantModelTrainConfigV8(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        pos_emb_config=pos_emb_cfg,
        specauc_start_epoch=11,
        label_target_size=184,
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
        dropout_broadcast_axes=None,
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor_symmetric",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor_symmetric",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor_symmetric",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor_symmetric",
        moving_average=None,
        weight_bit_prec=8,
        activation_bit_prec=8,
        quantize_output=False,
        quant_in_linear=True,
        converter_hardware_settings=prior_train_dac_settings,
        num_cycles=0,
        correction_settings=None,
        weight_noise_func=None,
        weight_noise_values=None,
        weight_noise_start_epoch=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )

    checkpoint = PtCheckpoint(
        tk.Path(
            "/work/asr4/hilmes/sis_work_folder/asr_2023/i6_core/returnn/training/ReturnnTrainingJob.b4JUc3ZKBZ4S/output/models/epoch.1000.pt"
        )
    )

    model = TrainedModel(
        descriptor="hilmes_memristor_v10_0607_v2",
        model_config=model_config,
        checkpoints={1000: checkpoint},
    )

    recog_results = list(recognition.hilmes_ctc.run(model=model))
    register_recog_report(recog_results, filename=filename)
    return model, recog_results
