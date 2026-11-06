from diagnostics.qat_ffnn_scores.reference_scoring import prepare_scores
from recipe.experiments.librispeech.training import qat_ffnn_transducer_bpe
from synaptogen_ml.memristor_modules import DacAdcHardwareSettings


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
    segments = [
        "dev-other/116-288048-0014/116-288048-0014",
        "dev-other/116-288048-0020/116-288048-0020",
        "dev-other/1701-141760-0024/1701-141760-0024",
        "dev-other/3660-172182-0037/3660-172182-0037",
        "dev-other/3663-172528-0020/3663-172528-0020",
        "dev-other/4831-18525-0028/4831-18525-0028",
        "dev-other/4831-29134-0012/4831-29134-0012",
        "dev-other/6467-94831-0006/6467-94831-0006",
        "dev-other/700-122866-0011/700-122866-0011",
        "dev-other/700-122868-0015/700-122868-0015",
        "dev-other/8254-84205-0043/8254-84205-0043",
        "dev-other/8288-274162-0025/8288-274162-0025",
    ]
    joint_adc = DacAdcHardwareSettings(8, 2, 6, 0.6, 8020.0)
    prepare_scores(
        model=model,
        corpus_name="dev-other",
        segments=segments,
        model_names=["base_model"],
        output_prefix="diagnostics/adapter_joint_2prec_6range",
        converter_hardware_settings=DacAdcHardwareSettings(8, 4, 4, 0.6, 8020.0),
        pos_enc_converter_hardware_settings=DacAdcHardwareSettings(8, 1, 7, 0.6, 8020.0),
        joint_l1_converter_hardware_settings=joint_adc,
        joint_l2_converter_hardware_settings=joint_adc,
    )
