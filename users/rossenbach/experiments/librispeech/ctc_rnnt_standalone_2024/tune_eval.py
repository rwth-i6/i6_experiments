from sisyphus import tk

import copy
from dataclasses import asdict
from typing import Dict, List, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from .pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig as FlashlightDecoderConfig, ExtraConfig as FlashlightDecoderExtraConfig
from .pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v5 import \
    DecoderConfig as BeamSearchDecoderConfigv5
from .pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v5 import DecoderConfig as RNNTBeamSearchDecoderConfigv5
from .pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v6 import DecoderConfig as RNNTBeamSearchDecoderConfigv6
from .pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v8 import DecoderConfig as RNNTBeamSearchDecoderConfigv8
from .pytorch_networks.rnnt.decoder.experimental_rnnt_decoder_v9 import DecoderConfig as RNNTBeamSearchDecoderConfigv9
from .pipeline import search, ASRModel


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
    extra_rqmt=None,
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
    :param extra_rqmt: update search job rqmt
    """

    # Automatic selection of decoder module
    if isinstance(base_decoder_config, FlashlightDecoderConfig):
        decoder_module = "ctc.decoder.flashlight_ctc_v1"
    elif isinstance(base_decoder_config, BeamSearchDecoderConfigv5):
        decoder_module = "ctc.decoder.beam_search_bpe_ctc_v5"
    elif isinstance(base_decoder_config, RNNTBeamSearchDecoderConfigv5):
        decoder_module = "rnnt.decoder.experimental_rnnt_decoder_v5"
    elif isinstance(base_decoder_config, RNNTBeamSearchDecoderConfigv6):
        decoder_module = "rnnt.decoder.experimental_rnnt_decoder_v6"
    elif isinstance(base_decoder_config, RNNTBeamSearchDecoderConfigv8):
        decoder_module = "rnnt.decoder.experimental_rnnt_decoder_v8"
    elif isinstance(base_decoder_config, RNNTBeamSearchDecoderConfigv9):
        decoder_module = "rnnt.decoder.experimental_rnnt_decoder_v9"
    else:
        assert False, "Invalid decoder config"

    # to avoid unwanted behavior, make sure there is only one dev set
    assert len(dev_dataset_tuples) == 1

    tune_parameters = []
    tune_values_dev = []
    tune_value_dev_key = list(dev_dataset_tuples.keys())[0]
    report_values = {}
    for lm_scale in lm_scales:
        for prior_scale in prior_scales:
            decoder_config = copy.deepcopy(base_decoder_config)
            if hasattr(decoder_config, "lm_weight"):
                decoder_config.lm_weight = lm_scale
            elif hasattr(decoder_config, "lm_scale"):
                decoder_config.lm_scale = lm_scale
            else:
                assert False, "could not determine config value to set for lm values"
            if hasattr(decoder_config, "prior_scale"):
                decoder_config.prior_scale = prior_scale
            elif hasattr(decoder_config, "zero_ilm_scale"):
                decoder_config.zero_ilm_scale = prior_scale
            else:
                assert False, "could not determine config value to set for prior values"
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
            tune_values_dev.append((wers[search_name+"/"+tune_value_dev_key]))
            if extra_rqmt:
                for search_job in search_jobs.values():
                    search_job.rqmt.update(extra_rqmt)

    pick_optimal_params_job = GetOptimalParametersAsVariableJob(
        parameters=tune_parameters, values=tune_values_dev, mode="minimize"

    )
    pick_optimal_params_job.add_alias(training_name + f"/pick_best_{tune_value_dev_key}")
    decoder_config = copy.deepcopy(base_decoder_config)
    if hasattr(decoder_config, "lm_weight"):
        decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
    elif hasattr(decoder_config, "lm_scale"):
        decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]

    if hasattr(decoder_config, "prior_scale"):
        decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
    elif hasattr(decoder_config, "zero_ilm_scale"):
        decoder_config.zero_ilm_scale = pick_optimal_params_job.out_optimal_parameters[1]

    for key in test_dataset_tuples.keys():
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
        report_values[key] = wers[training_name+"/"+key]
