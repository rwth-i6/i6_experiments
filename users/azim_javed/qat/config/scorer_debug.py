from diagnostics.qat_ffnn_scores.reference_scoring import prepare_scores, run_blank_penalty_fit
from diagnostics.qat_ffnn_scores.optimal_blank_penalty import run_optimal_blank_penalty
from recipe.experiments.librispeech.training import qat_ffnn_transducer_bpe
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings

def py():
    model = qat_ffnn_transducer_bpe.run(
        descriptor="qat_ffnn_transducer_full_quant_v2_lowbs",
        qat_args={
            "weight_bit_prec": 8,
            "activation_bit_prec": 8,
            "weight_dropout": 0.0,
            "weight_pruning_config": None,
        },
    )
    model_w4a8 = qat_ffnn_transducer_bpe.run(
        descriptor="qat_ffnn_transducer_full_quant_v2_lowbs_w4_a8",
        qat_args={
            "weight_bit_prec": 4,
            "activation_bit_prec": 8,
            "weight_dropout": 0.0,
            "weight_pruning_config": None,
        },
    )
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
    converter_hardware_settings_2prec_6range = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=2,
        output_range_bits=6,
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
    for run_name, num_segments in (("full_corpus", 1),):
        prepared = prepare_scores(
            model=model,
            corpus_name="dev-other",
            num_segments=num_segments,
            model_names=["base_model"],
            output_prefix=f"diagnostics/qat_ffnn_scores/{run_name}",
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
        )

        prepared_separate = prepare_scores(
            model=model,
            corpus_name="dev-other",
            num_segments=num_segments,
            model_names=["base_model"],
            output_prefix=f"diagnostics/qat_ffnn_scores_jl2_2prec_6range_no_embedding_regrid/{run_name}",
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings,
            joint_l2_converter_hardware_settings=converter_hardware_settings_2prec_6range,
            regrid_embedding=False,
        )
        run_blank_penalty_fit(prepared)
        prepared_w4a8 = prepare_scores(
            model=model_w4a8,
            corpus_name="dev-other",
            num_segments=num_segments,
            model_names=["base_model"],
            output_prefix=f"diagnostics/qat_ffnn_scores_w4a8_joint_correction_no_embedding_regrid/{run_name}",
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            regrid_embedding=False,
        )
        prepared_w4a8_4p8r = prepare_scores(
            model=model_w4a8,
            corpus_name="dev-other",
            num_segments=num_segments,
            model_names=["base_model"],
            output_prefix=f"diagnostics/qat_ffnn_scores_w4a8_joint_l1_4prec_8range_l2_4prec_8range_joint_correction_no_embedding_regrid/{run_name}",
            converter_hardware_settings=converter_hardware_settings,
            pos_enc_converter_hardware_settings=pos_enc_converter_hardware_settings,
            joint_l1_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_l2_converter_hardware_settings=converter_hardware_settings_4prec_8range,
            joint_correction_settings=CycleCorrectionSettings(
                num_cycles=10,
                test_input_value=0.6,
                relative_deviation=0.10,
                ideal_programming=False,
            ),
            regrid_embedding=False,
        )
        run_optimal_blank_penalty(prepared)
        run_optimal_blank_penalty(prepared_separate)
        run_optimal_blank_penalty(prepared_w4a8)
        run_optimal_blank_penalty(prepared_w4a8_4p8r)
