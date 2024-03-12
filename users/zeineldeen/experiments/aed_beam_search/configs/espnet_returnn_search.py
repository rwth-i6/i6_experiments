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


def run_espnet_search(beam_search_name, search_args, rqmts=None, hash_version=None):
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
                if v:
                    res += k + "-"
            else:
                assert isinstance(v, (int, float, str))
                res += f"{k}_{v}-"
        return res

    exp_name += f"batch_size_{batch_size}-"

    exp_name += get_exp_desc(returnn_recog_args)
    assert exp_name[-1] == "-"
    exp_name = exp_name[:-1]

    if hash_version:
        exp_name += f"-{hash_version}"

    if rqmts is None:
        rqmts = {}

    if rqmts.get("cpu_type", None) is not None:
        exp_name += f"-cpu_core_{rqmts['cpu_rqmt']}"

    exp_name += f"/{dataset}"

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


def py():
    # output/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_0.0-length_reward_0.0-max_seq_len_ratio_1.0/dev_other/wer
    # 4.57
    # output/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_0.0-length_reward_0.1-max_seq_len_ratio_1.0/dev_other/wer
    # 4.58
    # output/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_0.0-length_reward_0.0-max_seq_len_ratio_1.0/test_other/wer
    # 4.59
    # output/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_0.0-length_reward_0.1-max_seq_len_ratio_1.0/test_other/wer
    # 4.60
    # alias/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_0.0-length_reward_0.0-max_seq_len_ratio_1.0/dev_other/log.run.1
    # 2024-03-07 13:26:34,293 (espnet_beam_search:315) INFO: Overall RTF: 0.234
    # alias/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_0.0-length_reward_0.1-max_seq_len_ratio_1.0/dev_other/log.run.1
    # 2024-03-05 23:00:08,460 (espnet_beam_search:310) INFO: Overall RTF: 0.237
    # alias/beam_search_v5/batch_size_1-beam_size_20-ctc_weight_0.3-length_normalization_exponent_1.0-length_reward_0.0-max_seq_len_ratio_1.0/dev_other/log.run.1
    # 2024-03-07 13:34:36,550 (espnet_beam_search:315) INFO: Overall RTF: 0.248

    # output/sep_ended_keep/batch_size_1-adaptive_pruning-beam_ended_size_1-beam_size_20-ctc_weight_0.3-length_reward_0.0-max_seq_len_ratio_1.0-pruning_threshold_20/dev_other/wer
    # 4.57
    # output/sep_ended_keep/batch_size_1-adaptive_pruning-beam_ended_size_1-beam_size_20-ctc_weight_0.3-length_reward_0.0-max_seq_len_ratio_1.0-pruning_threshold_50/dev_other/wer
    # 4.57
    # output/sep_ended_keep/batch_size_1-adaptive_pruning-beam_ended_size_1-beam_size_20-ctc_weight_0.3-length_reward_0.0-max_seq_len_ratio_1.0-pruning_threshold_20/test_other/wer
    # 4.59
    # output/sep_ended_keep/batch_size_1-adaptive_pruning-beam_ended_size_1-beam_size_20-ctc_weight_0.3-length_reward_0.0-max_seq_len_ratio_1.0-pruning_threshold_50/test_other/wer
    # 4.59

    # alias/sep_ended_keep/batch_size_1-adaptive_pruning-beam_ended_size_1-beam_size_20-ctc_weight_0.3-length_reward_0.1-max_seq_len_ratio_1.0-pruning_threshold_20/test_other/log.run.1
    # 2024-03-07 13:30:04,125 (espnet_beam_search:315) INFO: Overall RTF: 0.243
    # alias/sep_ended_keep/batch_size_1-adaptive_pruning-beam_ended_size_1-beam_size_20-ctc_weight_0.3-length_reward_0.1-max_seq_len_ratio_1.0-pruning_threshold_50/test_other/log.run.1
    # 2024-03-07 13:29:16,199 (espnet_beam_search:315) INFO: Overall RTF: 0.241

    # for dataset in ["dev_other"]:
    #     returnn_search_args = copy.deepcopy(baseline_search_args)
    #     returnn_search_args["dataset"] = dataset
    #     for batch_size in [1]:
    #         for max_seq_len_ratio in [0.5, 1.0]:
    #             for beam in [20]:
    #                 for adaptive in [True, False]:
    #                     for prun_threshold in [20, 50]:
    #                         for len_reward in [0.1, 0.2]:
    #                             returnn_search_args["batch_size"] = batch_size
    #                             returnn_search_args["returnn_recog_args"] = {
    #                                 "beam_size": beam,
    #                                 "beam_ended_size": 1,
    #                                 "length_reward": len_reward,
    #                                 "pruning_threshold": prun_threshold,
    #                                 "adaptive_pruning": adaptive,
    #                                 "max_seq_len_ratio": max_seq_len_ratio,
    #                                 "beam_search_variant": "sep_ended_keep",
    #                                 "ctc_weight": 0.3,
    #                                 "max_seq_len_offset": -1,
    #                             }
    #                             run_espnet_search("sep_ended_keep", returnn_search_args, hash_version="offset_1")
    #
    #                         # returnn_search_args["returnn_recog_args"] = {
    #                         #     "beam_size": beam,
    #                         #     "length_reward": len_reward,
    #                         #     "length_normalization_exponent": 0.0,
    #                         #     "max_seq_len_ratio": max_seq_len_ratio,
    #                         #     "beam_search_variant": "beam_search_v5",
    #                         #     "ctc_weight": 0.3,
    #                         # }
    #                         # run_espnet_search("beam_search_v5", returnn_search_args)
    #
    # # -------------------------- Experiments ------------------------------------- #
    #
    # # TODO: joint CTC
    # for dataset in ["dev_other"]:
    #     returnn_search_args = copy.deepcopy(baseline_search_args)
    #     returnn_search_args["dataset"] = dataset
    #     for batch_size in [1]:
    #         for max_seq_len_ratio in [0.3, 0.5]:
    #             for beam in [20]:
    #                 for adaptive in [True, False]:
    #                     for prun_threshold in [10, 20, 50]:
    #                         for len_reward in [0.1, 0.2]:
    #                             returnn_search_args["batch_size"] = batch_size
    #                             returnn_search_args["returnn_recog_args"] = {
    #                                 "beam_size": beam,
    #                                 "beam_ended_size": 1,
    #                                 "length_reward": len_reward,
    #                                 "pruning_threshold": prun_threshold,
    #                                 "adaptive_pruning": adaptive,
    #                                 "max_seq_len_ratio": max_seq_len_ratio,
    #                                 "beam_search_variant": "sep_ended_keep",
    #                                 "ctc_weight": 0.3,
    #                             }
    #                             run_espnet_search("sep_ended_keep", returnn_search_args)
    #
    #                         returnn_search_args["returnn_recog_args"] = {
    #                             "beam_size": beam,
    #                             "length_reward": len_reward,
    #                             "length_normalization_exponent": 0.0,
    #                             "max_seq_len_ratio": max_seq_len_ratio,
    #                             "beam_search_variant": "beam_search_v5",
    #                             "ctc_weight": 0.3,
    #                         }
    #                         run_espnet_search("beam_search_v5", returnn_search_args)
    #
    # # TODO: analysis
    # for dataset in ["dev_other"]:
    #     returnn_search_args = copy.deepcopy(baseline_search_args)
    #     returnn_search_args["dataset"] = dataset
    #     for batch_size in [1]:
    #         for max_seq_len_ratio in [0.5]:
    #             for beam in [20]:
    #                 for adaptive in [True, False]:
    #                     for prun_threshold in [20]:
    #                         for len_reward in [0.1]:
    #                             returnn_search_args["batch_size"] = batch_size
    #                             returnn_search_args["returnn_recog_args"] = {
    #                                 "beam_size": beam,
    #                                 "beam_ended_size": 1,
    #                                 "length_reward": len_reward,
    #                                 "pruning_threshold": prun_threshold,
    #                                 "adaptive_pruning": adaptive,
    #                                 "max_seq_len_ratio": max_seq_len_ratio,
    #                                 "beam_search_variant": "sep_ended_keep",
    #                                 "ctc_weight": 0.3,
    #                                 "debug": True,
    #                             }
    #                             run_espnet_search("sep_ended_keep", returnn_search_args)
    #                             returnn_search_args["returnn_recog_args"]["max_seq_len_offset"] = -1
    #                             run_espnet_search("sep_ended_keep", returnn_search_args)
    #
    #                         returnn_search_args["returnn_recog_args"] = {
    #                             "beam_size": beam,
    #                             "length_reward": len_reward,
    #                             "length_normalization_exponent": 0.0,
    #                             "max_seq_len_ratio": max_seq_len_ratio,
    #                             "beam_search_variant": "beam_search_v5",
    #                             "ctc_weight": 0.3,
    #                             "debug": True,
    #                         }
    #                         run_espnet_search("beam_search_v5", returnn_search_args, hash_version="debug")
    #
    # # TODO: + LM
    # for dataset in ["dev_other"]:
    #     returnn_search_args = copy.deepcopy(baseline_search_args)
    #     returnn_search_args["dataset"] = dataset
    #     for batch_size in [1]:
    #         for maxlenratio in [0.5]:
    #             for beam in [20]:
    #                 for adapt_prun in [True]:
    #                     for prun_thre in [5]:
    #                         for len_reward in [0.1, 0.2, 0.6]:
    #                             for lm_weight in [0.1, 0.12, 0.14]:
    #                                 returnn_search_args["batch_size"] = batch_size
    #                                 # returnn_search_args["returnn_recog_args"] = {
    #                                 #     "beam_size": beam,
    #                                 #     "beam_ended_size": 1,
    #                                 #     "length_reward": len_reward,
    #                                 #     "pruning_threshold": prun_thre,
    #                                 #     "adaptive_pruning": adapt_prun,
    #                                 #     "max_seq_len_ratio": maxlenratio,
    #                                 #     "beam_search_variant": "sep_ended_keep",
    #                                 #     "lm_weight": lm_weight,
    #                                 # }
    #                                 # run_espnet_search("sep_ended_keep", returnn_search_args)
    #
    #                                 returnn_search_args["returnn_recog_args"] = {
    #                                     "beam_size": beam,
    #                                     "length_reward": len_reward,
    #                                     "max_seq_len_ratio": maxlenratio,
    #                                     "length_normalization_exponent": 0.0,
    #                                     "beam_search_variant": "beam_search_v5",
    #                                     "lm_weight": lm_weight,
    #                                 }
    #                                 run_espnet_search("beam_search_v5", returnn_search_args)
    #
    #                                 returnn_search_args["returnn_recog_args"] = {
    #                                     "beam_size": beam,
    #                                     "max_seq_len_ratio": maxlenratio,
    #                                     "length_normalization_exponent": 1.0,
    #                                     "beam_search_variant": "beam_search_v5",
    #                                     "lm_weight": lm_weight,
    #                                 }
    #                                 run_espnet_search("beam_search_v5", returnn_search_args)

    # TODO: CPU RTF
    for dataset in ["dev_other"]:
        returnn_search_args = copy.deepcopy(baseline_search_args)
        returnn_search_args["dataset"] = dataset
        returnn_search_args["device"] = "cpu"

        returnn_search_args["batch_size"] = 1
        returnn_search_args["returnn_recog_args"] = {
            "beam_size": 20,
            "length_reward": 0.0,
            "max_seq_len_ratio": 0.5,
            "length_normalization_exponent": 0.0,
            "beam_search_variant": "beam_search_v5",
            "pruning_threshold": 10,
            "adaptive_pruning": True,
            "lm_weight": 0.6,
            "ctc_weight": 0.3,
        }
        run_espnet_search(
            "beam_search_v5", returnn_search_args, rqmts={"cpu_type": "rescale_intel", "cpu_rqmt": 4, "time_rqmt": 24}
        )

        run_espnet_search("beam_search_v5", returnn_search_args, hash_version="gpu_maxseqs1")

        # returnn_search_args["batch_size"] = 1
        # returnn_search_args["returnn_recog_args"] = {
        #     "beam_size": 20,
        #     "length_reward": 1.0,
        #     "max_seq_len_ratio": 0.5,
        #     "beam_ended_size": 1,
        #     "beam_search_variant": "sep_ended_keep",
        #     "lm_weight": 0.6,
        #     "ctc_weight": 0.3,
        # }
        # run_espnet_search(
        #     "sep_ended_keep", returnn_search_args, rqmts={"cpu_type": "rescale_intel", "cpu_rqmt": 4, "time_rqmt": 24}
        # )
