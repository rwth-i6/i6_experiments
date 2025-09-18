from sisyphus import tk

import copy
from dataclasses import asdict
from typing import Dict, List, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from .pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig as FlashlightDecoderConfig, ExtraConfig as FlashlightDecoderExtraConfig
from .pipeline import search, ASRModel
from .report import tune_and_evalue_report


DecoderConfig = FlashlightDecoderConfig
ExtraConfig = FlashlightDecoderExtraConfig


def tune_and_evaluate_helper(
    training_name: str,
    asr_model: ASRModel,
    base_decoder_config: DecoderConfig,
    dev_dataset_tuples,
    test_dataset_tuples,
    default_returnn: Dict[str, tk.Path],
    lm_scales: List[float],
    prior_scales: List[float],
    unhashed_decoder_config: Optional[ExtraConfig] = None,
    extra_forward_config=None,
    use_gpu=False,
):
    """
    Example helper to execute tuning over lm_scales and prior scales.
    With the best values runs test-clean and test-other.

    This is just a reference helper and can (should) be freely changed, copied, modified etc...

    :param training_name: for alias and output names
    :param asr_model: ASR model to use
    :param base_decoder_config: any decoder config dataclass
    :param lm_scales: lm scales for tuning
    :param prior_scales: prior scales for tuning, same length as lm scales
    :param unhashed_decoder_config: decoder config without hashing, used for BeamSearch configs
    :param extra_forward_config: additional args to the ReturnnForwardJob, e.g. to modify batch size
    :param use_gpu: for GPU decoding
    """

    # Automatic selection of decoder module
    if isinstance(base_decoder_config, FlashlightDecoderConfig):
        decoder_module = "ctc.decoder.flashlight_ctc_v2"
    else:
        assert False, "Invalid decoder config"

    tune_parameters = []
    tune_values_dev = []
    report_values = {}
    for lm_scale in lm_scales:
        for prior_scale in prior_scales:
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_scale = lm_scale
            decoder_config.prior_scale = prior_scale
            search_name = training_name + "/search_lm%.2f_prior%.2f" % (lm_scale, prior_scale)
            search_jobs, wers = search(
                search_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)}
                if unhashed_decoder_config
                else None,
                use_gpu=use_gpu,
                **default_returnn,
            )
            tune_parameters.append((lm_scale, prior_scale))
            tune_values_dev.append((wers[search_name + "/dev"]))

    for key, tune_values in [("test", tune_values_dev)]:
        pick_optimal_params_job = GetOptimalParametersAsVariableJob(
            parameters=tune_parameters, values=tune_values, mode="minimize"

        )
        pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
        decoder_config = copy.deepcopy(base_decoder_config)
        decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
        decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
        search_jobs, wers = search(
            training_name,
            forward_config=extra_forward_config if extra_forward_config else {},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={key: test_dataset_tuples[key]},
            unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)}
            if unhashed_decoder_config
            else None,
            use_gpu=use_gpu,
            **default_returnn,
        )
        report_values[key] = wers[training_name + "/" + key]

    tune_and_evalue_report(
        training_name=training_name,
        tune_parameters=tune_parameters,
        tuning_names=["LM", "Prior"],
        tune_values_dev=tune_values_dev,
        report_values=report_values,
    )