import copy
import os

from sisyphus import *

from i6_core.returnn.search import ReturnnComputeWERJob

from i6_experiments.users.zeineldeen.experiments.aed_beam_search.default_tools import PYTHON_EXE, SCTK_BINARY_PATH
from i6_experiments.users.zeineldeen.recipe.espnet.search import (
    EspnetBeamSearchJob,
    ConvertHypRefToDict,
    EspnetScliteScoringJob,
)

# also contains downloaded audio files
librispeech_data_path = tk.Path(
    "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-10-14--espnet/espnet/egs2/librispeech/asr1"
)
beam_search_script = tk.Path(
    "recipe/i6_experiments/users/zeineldeen/experiments/aed_beam_search/espnet_beam_search.py",
    hash_overwrite="beam_search_v1",
)

baseline_search_args = {
    "device": "cuda",
    "nbest": 1,
    "beam_size": 8,
    "len_reward": 0.0,
    "ctc_weight": 0.0,
    "lm_weight": 0.0,
}


def run_espnet_search(search_args, rqmts=None):
    assert "device" in search_args
    assert "dataset" in search_args
    dataset = search_args["dataset"]
    exp_name = f"espnet_beam_search/"
    for k, v in sorted(search_args.items()):
        if k == "dataset":
            continue
        if isinstance(v, bool):
            exp_name += k
        else:
            assert isinstance(v, (int, float, str))
            exp_name += f"{k}_{v}-"
    if exp_name[-1] == "-":
        exp_name = exp_name[:-1]

    if search_args["device"] == "cpu" and rqmts:
        exp_name += f"-cpu_core_{rqmts['cpu_rqmt']}"

    exp_name += f"/{dataset}"
    if rqmts is None:
        rqmts = {}
    espnet_search_job = EspnetBeamSearchJob(
        beam_search_script=beam_search_script,
        data_path=librispeech_data_path,
        search_args={"dataset": dataset, **search_args},
        python_exe=PYTHON_EXE,
        **rqmts,
    )
    espnet_search_job.add_alias(exp_name)
    tk.register_output(exp_name + "/hyp", espnet_search_job.out_hyp)

    ref_path = tk.Path(os.path.join(librispeech_data_path, f"data/{dataset}/text"))

    sclite_job = EspnetScliteScoringJob(hyp=espnet_search_job.out_hyp, ref=ref_path, sclite_exe=SCTK_BINARY_PATH)
    tk.register_output(exp_name + "/sclite_wer", sclite_job.out_wer_report)

    ref_dict = ConvertHypRefToDict(ref_path).out_dict
    hyp_dict = ConvertHypRefToDict(espnet_search_job.out_hyp).out_dict
    wer_j = ReturnnComputeWERJob(hyp_dict, ref_dict)
    tk.register_output(exp_name + "/wer", wer_j.out_wer)


# CPU nodes:
#
# cn-30: AMD EPYC 7313P 16-Core Processor
# cn-31: Intel(R) Xeon(R) Gold 5416S


def py():

    # model is broken without joint CTC
    #
    # beam size {18,20}, length reward = 0.1
    # dev-other: 4.6
    # test-other: 6.2 (broken segments)
    #
    # same but with max_len_ratio 0.3
    # dev-other: 4.6
    # test-other: 5.2

    # running only with CTC

    # output/espnet_beam_search/beam_size_20-ctc_weight_0.3-device_cuda-len_reward_0.0-lm_weight_0.0-maxlenratio_1.0-nbest_1/dev_other/wer
    # 4.59
    # output/espnet_beam_search/beam_size_20-ctc_weight_0.3-device_cuda-len_reward_0.1-lm_weight_0.0-maxlenratio_1.0-nbest_1/dev_other/wer
    # 4.59
    # output/espnet_beam_search/beam_size_20-ctc_weight_0.3-device_cuda-len_reward_0.0-lm_weight_0.0-maxlenratio_1.0-nbest_1/test_other/wer
    # 4.60
    # output/espnet_beam_search/beam_size_20-ctc_weight_0.3-device_cuda-len_reward_0.1-lm_weight_0.0-maxlenratio_1.0-nbest_1/test_other/wer
    # 4.60

    for max_len in [1.0]:
        for dataset in ["dev_other", "test_other"]:
            for lm_weight in [0.0]:
                for ctc_weight in [0.3]:
                    for beam_size in [20, 60]:
                        for len_reward in [0.0, 0.1]:
                            search_args_ = copy.deepcopy(baseline_search_args)
                            search_args_["beam_size"] = beam_size
                            search_args_["len_reward"] = len_reward
                            search_args_["lm_weight"] = lm_weight
                            search_args_["ctc_weight"] = ctc_weight
                            search_args_["dataset"] = dataset
                            search_args_["maxlenratio"] = max_len
                            run_espnet_search(search_args_)
