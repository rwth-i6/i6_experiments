from ...default_tools import RETURNN_EXE, QUANT_RETURNN, MINI_RETURNN_ROOT
from ...pipeline import search, ASRModel, quantize_static
from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
from typing import List, Optional, Dict, Any
from ...data.common import TrainingDatasets
from dataclasses import dataclass, asdict
from ...config import get_static_quant_config
import copy
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

@dataclass
class QuantArgs:
    sample_ls: List[int]
    quant_config_dict: Dict[str, Any]
    decoder: str
    num_iterations: int
    datasets: TrainingDatasets
    network_module: str
    filter_args: Optional[Dict[str, Any]] = None

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": MINI_RETURNN_ROOT,
}

def tune_and_evaluate_helper(
    training_name: str,
    asr_model: ASRModel,
    base_decoder_config: DecoderConfig,
    lm_scales: List[float],
    prior_scales: List[float],
    dev_dataset_tuples: Dict[str, Any],
    quant_str: Optional[str] = None,
    test_dataset_tuples: Optional[Dict[str, Any]] = None,
    quant_args: Optional[QuantArgs] = None,
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
    """
    tune_parameters = []
    tune_values = []
    results = {}
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
                decoder_module="ctc.decoder.flashlight_ctc_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples=dev_dataset_tuples,
                **default_returnn,
            )
            tune_parameters.append((lm_weight, prior_scale))
            tune_values.append((wers[search_name + "/dev"]))
            results.update(wers)
    if quant_args is not None:
        assert quant_str is not None, "You want your quant to have a name"
        for num_samples in quant_args.sample_ls:
            for seed in range(quant_args.num_iterations):
                it_name = training_name + quant_str + f"/quantize_static/samples_{num_samples}/seed_{seed}"
                quant_config = get_static_quant_config(
                    training_datasets=quant_args.datasets,
                    network_module=quant_args.network_module,
                    net_args=asr_model.net_args,
                    quant_args=quant_args.quant_config_dict,
                    config={},
                    num_samples=num_samples,
                    dataset_seed=seed,
                    debug=False,
                    dataset_filter_args=quant_args.filter_args
                )
                quant_chkpt = quantize_static(
                    prefix_name=it_name,
                    returnn_config=quant_config,
                    checkpoint=asr_model.checkpoint,
                    returnn_exe=RETURNN_EXE,
                    returnn_root=QUANT_RETURNN,
                )
                quant_model = ASRModel(
                    checkpoint=quant_chkpt,
                    net_args=asr_model.net_args | quant_args.quant_config_dict,
                    network_module=quant_args.network_module,
                    prior_file=asr_model.prior_file,
                    prefix_name=it_name
                )
                for lm_weight in lm_scales:
                    for prior_scale in prior_scales:
                        decoder_config = copy.deepcopy(base_decoder_config)
                        decoder_config.lm_weight = lm_weight
                        decoder_config.prior_scale = prior_scale
                        search_name = it_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                        search_jobs, wers = search(
                            search_name,
                            forward_config={},
                            asr_model=quant_model,
                            decoder_module=quant_args.decoder,
                            decoder_args={"config": asdict(decoder_config)},
                            test_dataset_tuples=dev_dataset_tuples,
                            **default_returnn,
                        )
                        results.update(wers)
    pick_optimal_params_job = GetOptimalParametersAsVariableJob(
        parameters=tune_parameters, values=tune_values, mode="minimize"
    )
    pick_optimal_params_job.add_alias(training_name + f"/pick_best_dev")
    if test_dataset_tuples is not None:
        for key, tune_values in [("test", tune_values)]:
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module="ctc.decoder.flashlight_ctc_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn,
            )
            results.update(wers)
    return results, pick_optimal_params_job
