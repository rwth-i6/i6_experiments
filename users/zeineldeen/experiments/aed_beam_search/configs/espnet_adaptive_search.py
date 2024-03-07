import copy
import os

from sisyphus import *

from i6_core.returnn.search import ReturnnComputeWERJob

from i6_experiments.users.zeineldeen.experiments.aed_beam_search.default_tools import PYTHON_EXE, SCTK_BINARY_PATH
from i6_experiments.users.zeineldeen.recipe.espnet.search import (
    EspnetBeamSearchJob,
    ConvertHypRefToDict,
    EspnetScliteScoringJob,
    EspnetCalculateRtfJob,
)

# also contains downloaded audio files
librispeech_data_path = tk.Path(
    "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2023-10-14--espnet/espnet/egs2/librispeech/asr1"
)
beam_search_script = tk.Path(
    "recipe/i6_experiments/users/zeineldeen/experiments/aed_beam_search/espnet_beam_search.py",
    hash_overwrite="beam_search_v2",
)

baseline_search_args = {
    "device": "cuda",
    "nbest": 1,
    "beam_size": 8,
    "len_reward": 0.0,
    "ctc_weight": 0.0,
    "lm_weight": 0.0,
}


def run_espnet_search(search_args, suffix="", rqmts=None, ted2_scoring=False):
    assert "dataset" in search_args
    pylasr_recog_args = search_args["pylasr_recog_args"]
    dataset = search_args["dataset"]
    exp_name = f"pylasr_beam_search/"

    def get_exp_desc(args):
        res = ""
        for k, v in sorted(args.items()):
            if k == "dataset":
                continue
            if isinstance(v, dict):
                res += get_exp_desc(v)
                return res
            if isinstance(v, bool):
                if v:
                    res += k + "-"
                else:
                    res += f"Wo_{k}-"
            else:
                assert isinstance(v, (int, float, str))
                res += f"{k}_{v}-"
        return res

    exp_name += get_exp_desc(pylasr_recog_args)
    assert exp_name[-1] == "-"
    exp_name = exp_name[:-1]

    if search_args["device"] == "cpu" and rqmts:
        exp_name += f"-cpu_core_{rqmts['cpu_rqmt']}"

    if suffix:
        exp_name += f"-{suffix}"

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

    sclite_job = EspnetScliteScoringJob(
        hyp=espnet_search_job.out_hyp, ref=ref_path, sclite_exe=SCTK_BINARY_PATH, upper_case=ted2_scoring
    )
    tk.register_output(exp_name + "/sclite_wer", sclite_job.out_wer_report)

    ref_dict = ConvertHypRefToDict(ref_path).out_dict
    hyp_dict = ConvertHypRefToDict(espnet_search_job.out_hyp, upper_case=ted2_scoring).out_dict
    wer_j = ReturnnComputeWERJob(hyp_dict, ref_dict)
    tk.register_output(exp_name + "/wer", wer_j.out_wer)


def py():
    # # TODO: Use Ted2 model to recognize LibriSpeech dev_other dataset
    # pylasr_search_args = copy.deepcopy(baseline_search_args)
    # pylasr_search_args["dataset"] = "dev_other"
    # for maxlenratio in [0.3, 1.0]:
    #     for beam in [1, 4, 8, 10, 15, 20]:
    #         for prun_thre in [5, 10, 20]:
    #             for len_reward in [0.2]:
    #                 pylasr_search_args["model_tag"] = "pyf98/tedlium2_e_branchformer"
    #                 pylasr_search_args["pylasr_recog_args"] = {
    #                     "beam": beam,
    #                     "lengthReward": len_reward,
    #                     "maxLengthRatio": maxlenratio,
    #                     "pruning": True,
    #                     "pruningThreshold": prun_thre,
    #                     "pruningThresholdAutoTune": True,
    #                 }
    #                 run_espnet_search(pylasr_search_args, suffix="ted2Model", ted2_scoring=True)

    for dataset in ["dev_other", "test_other"]:
        pylasr_search_args = copy.deepcopy(baseline_search_args)
        pylasr_search_args["dataset"] = dataset
        for maxlenratio in [1.0]:
            for beam in [20]:
                for adapt_prun in [True, False]:
                    for prun_thre in [30]:
                        for len_reward in [0.1]:
                            pylasr_search_args["pylasr_recog_args"] = {
                                "beam": beam,
                                "lengthReward": len_reward,
                                "maxLengthRatio": maxlenratio,
                                "pruning": True,
                                "pruningThreshold": prun_thre,
                                "pruningThresholdAutoTune": adapt_prun,
                                "ctcWeight": 0.3,
                            }
                            run_espnet_search(pylasr_search_args)

    # TODO: CPU RTF
    # for maxlenratio in [0.3]:
    #     for beam in [20]:
    #         for lm_weight in [0.0]:
    #             for adapt in [True, False]:
    #                 for prun_thre in [5, 10, 20]:
    #                     for len_reward in [0.2]:
    #                         pylasr_search_args = copy.deepcopy(baseline_search_args)
    #                         pylasr_search_args["dataset"] = "dev_other"
    #                         pylasr_search_args["device"] = "cpu"
    #                         pylasr_search_args["pylasr_recog_args"] = {
    #                             "beam": beam,
    #                             "device": "cpu",
    #                             "lengthReward": len_reward,
    #                             "maxLengthRatio": maxlenratio,
    #                             "pruning": True,
    #                             "lmWeight": lm_weight,
    #                             "pruningThreshold": prun_thre,
    #                             "pruningThresholdAutoTune": adapt,
    #                         }
    #                         run_espnet_search(
    #                             pylasr_search_args, rqmts={"time_rqmt": 24, "cpu_rqmt": 4, "cpu_type": "rescale_intel"}
    #                         )
