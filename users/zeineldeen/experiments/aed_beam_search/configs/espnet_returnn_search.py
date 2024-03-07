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
    hash_overwrite="beam_search_v3",
)

baseline_search_args = {
    "device": "cuda",
    "nbest": 1,
    "beam_size": 8,
    "len_reward": 0.0,
    "ctc_weight": 0.0,
    "lm_weight": 0.0,
}


def run_espnet_search(beam_search_name, search_args):
    assert "dataset" in search_args
    returnn_recog_args = search_args["returnn_recog_args"]
    batch_size = search_args.get("batch_size", 1)
    dataset = search_args["dataset"]
    exp_name = f"{beam_search_name}/"

    def get_exp_desc(args):
        res = ""
        for k, v in sorted(args.items()):
            if k == "dataset":
                continue
            if k == "beam_search_variant":
                continue
            if isinstance(v, dict):
                res += get_exp_desc(v)
                return res
            elif isinstance(v, bool):
                res += k + "-"
            else:
                assert isinstance(v, (int, float, str))
                res += f"{k}_{v}-"
        return res

    exp_name += f"batch_size_{batch_size}-"

    exp_name += get_exp_desc(returnn_recog_args)
    assert exp_name[-1] == "-"
    exp_name = exp_name[:-1] + f"/{dataset}"

    espnet_search_job = EspnetBeamSearchJob(
        beam_search_script=beam_search_script,
        data_path=librispeech_data_path,
        search_args={"dataset": dataset, **search_args},
        python_exe=PYTHON_EXE,
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


def py():
    for dataset in ["dev_other", "test_other"]:
        returnn_search_args = copy.deepcopy(baseline_search_args)
        returnn_search_args["dataset"] = dataset
        for batch_size in [1]:
            for max_seq_len_ratio in [1.0]:
                for beam in [20, 40]:
                    for prun_threshold in [10, 50]:
                        for len_reward in [0.0, 0.1]:
                            returnn_search_args["batch_size"] = batch_size
                            returnn_search_args["returnn_recog_args"] = {
                                "beam_size": beam,
                                "beam_ended_size": 1,
                                "length_reward": len_reward,
                                "pruning_threshold": prun_threshold,
                                "adaptive_pruning": True,
                                "max_seq_len_ratio": max_seq_len_ratio,
                                "beam_search_variant": "sep_ended_keep",
                                "ctc_weight": 0.3,
                            }
                            run_espnet_search("sep_ended_keep", returnn_search_args)

                            returnn_search_args["returnn_recog_args"] = {
                                "beam_size": beam,
                                "length_reward": len_reward,
                                "length_normalization_exponent": 0.0,
                                "max_seq_len_ratio": max_seq_len_ratio,
                                "beam_search_variant": "beam_search_v5",
                                "ctc_weight": 0.3,
                            }
                            run_espnet_search("beam_search_v5", returnn_search_args)
