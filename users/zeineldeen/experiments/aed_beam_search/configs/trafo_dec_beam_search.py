"""
Script to run our torch beam search implementations inside returnn with torch as engine backend
"""

from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.aed import (
    config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    speed_pert_librosa_config,
    train_exp,
    _recog,
    model_recog_pure_torch,
)
from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.conformer_import_moh_att_2023_06_30 import model_warmup


trafo_dec_model = train_exp(  # 5.11 (!!)
    "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
    config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    model_config={"behavior_version": 20},  # new Trafo decoder defaults
    config_updates={
        "optimizer.weight_decay": 1e-2,
        "__train_audio_preprocess": speed_pert_librosa_config,
        "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    },
)


def get_name(args):
    ignored_keys = ["__recog_def_ext", "__batch_size_dependent", "beam_search_collect_individual_seq_scores"]
    res = ""
    for k, v in sorted(args.items()):
        if k in ignored_keys:
            continue
        if isinstance(v, dict):
            res += get_name(v)
            return res
        elif isinstance(v, bool):
            if v:
                res += k + "-"
        else:
            assert isinstance(v, (int, float, str))
            res += f"{k}_{v}-"
    assert res[-1] == "-"
    res = res[:-1]
    return res


def py():
    # TODO: effect of adaptive threshold pruning
    for max_seq_len_factor in [0.3, 0.5, 1.0]:
        for batch_size in [10_000 * 160]:
            for threshold in [5, 10, 20, 30, 50]:
                for adaptive in [True, False]:
                    for beam in [12, 32, 50]:
                        for beam_ended in [1]:
                            for len_reward in [0.1, 0.2, 0.3, 0.4]:
                                recog_config = {}
                                recog_config["beam_search_version"] = "sep_ended_keep_v5"
                                recog_config["__batch_size_dependent"] = True
                                recog_config["__recog_def_ext"] = True
                                recog_config["beam_search_collect_individual_seq_scores"] = True
                                recog_config["beam_search_opts"] = {
                                    "beam_size": beam,
                                    "length_normalization_exponent": 0.0,
                                    "length_reward": len_reward,
                                    "beam_ended_size": beam_ended,
                                    "pruning_threshold": threshold,
                                    "adaptive_pruning": adaptive,
                                }
                                if max_seq_len_factor != 1.0:
                                    recog_config["beam_search_opts"]["max_seq_len_factor"] = max_seq_len_factor
                                name = get_name(recog_config["beam_search_opts"])
                                recog_config["load_model_post_hooks"] = [model_warmup]
                                recog_config["run"] = 1
                                if batch_size:
                                    recog_config["batch_size"] = batch_size
                                    name += f"_batch{batch_size}"
                                _recog(
                                    f"trafo_dec_model/sep_ended_keep_v5/{name}",
                                    trafo_dec_model.get_last_fixed_epoch(),
                                    model_recog_pure_torch,
                                    recog_config,
                                    search_rqmt={
                                        "cpu": 2,
                                        "time": 0.35,
                                        "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"],
                                    },
                                    dev_sets=["dev-other"],
                                )

    # TODO: debug
    for max_seq_len_factor in [0.5, 1.0]:
        for batch_size in [1000 * 160]:  # 1 second to have only 1 seq for debugging
            for threshold in [50]:
                for adaptive in [True]:
                    for beam in [50]:
                        for beam_ended in [1]:
                            for len_reward in [0.1, 0.2, 0.3]:
                                recog_config = {}
                                recog_config["beam_search_version"] = "sep_ended_keep_v5"
                                recog_config["__batch_size_dependent"] = True
                                recog_config["__recog_def_ext"] = True
                                recog_config["beam_search_collect_individual_seq_scores"] = True
                                recog_config["beam_search_opts"] = {
                                    "beam_size": beam,
                                    "length_normalization_exponent": 0.0,
                                    "length_reward": len_reward,
                                    "beam_ended_size": beam_ended,
                                    "pruning_threshold": threshold,
                                    "adaptive_pruning": adaptive,
                                }
                                if max_seq_len_factor != 1.0:
                                    recog_config["beam_search_opts"]["max_seq_len_factor"] = max_seq_len_factor
                                name = get_name(recog_config["beam_search_opts"])
                                recog_config["load_model_post_hooks"] = [model_warmup]
                                recog_config["run"] = 1
                                if batch_size:
                                    recog_config["batch_size"] = batch_size
                                    name += f"_batch{batch_size}"
                                _recog(
                                    f"trafo_dec_model/sep_ended_keep_v5/{name}",
                                    trafo_dec_model.get_last_fixed_epoch(),
                                    model_recog_pure_torch,
                                    recog_config,
                                    search_rqmt={
                                        "cpu": 2,
                                        "time": 0.35,
                                        "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"],
                                    },
                                    dev_sets=["dev-other"],
                                )

    for batch_size in [10_000 * 160]:
        for beam_search_variant in [5]:
            for beam in [12, 32, 50]:
                for len_exp in [1.0]:
                    recog_config = {}
                    recog_config["beam_search_version"] = beam_search_variant
                    recog_config["__batch_size_dependent"] = True
                    recog_config["__recog_def_ext"] = True
                    recog_config["beam_search_collect_individual_seq_scores"] = True
                    recog_config["beam_search_opts"] = {
                        "beam_size": beam,
                        "length_normalization_exponent": len_exp,
                    }
                    if beam_search_variant == "sep_ended":
                        recog_config["beam_search_opts"]["beam_and_ended_size"] = beam
                    name = get_name(recog_config["beam_search_opts"])
                    recog_config["load_model_post_hooks"] = [model_warmup]
                    recog_config["run"] = 1
                    if batch_size:
                        recog_config["batch_size"] = batch_size
                        name += f"_batch{batch_size}"
                    _recog(
                        f"base-24gb-v6-lrlin1e_5_600k/{beam_search_variant}/{name}",
                        trafo_dec_model.get_last_fixed_epoch(),
                        model_recog_pure_torch,
                        recog_config,
                        search_rqmt={"cpu": 2, "time": 0.35, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                        dev_sets=["dev-other"],
                    )
                for len_reward in [0.4]:
                    recog_config = {}
                    recog_config["beam_search_version"] = beam_search_variant
                    recog_config["__batch_size_dependent"] = True
                    recog_config["__recog_def_ext"] = True
                    recog_config["beam_search_collect_individual_seq_scores"] = True
                    recog_config["beam_search_opts"] = {
                        "beam_size": beam,
                        "length_normalization_exponent": 0.0,
                        "length_reward": len_reward,
                    }
                    if beam_search_variant == "sep_ended":
                        recog_config["beam_search_opts"]["beam_and_ended_size"] = beam
                    name = get_name(recog_config["beam_search_opts"])
                    recog_config["load_model_post_hooks"] = [model_warmup]
                    recog_config["run"] = 1
                    _recog(
                        f"trafo_dec_model/{beam_search_variant}/{name}",
                        trafo_dec_model.get_last_fixed_epoch(),
                        model_recog_pure_torch,
                        recog_config,
                        search_rqmt={"cpu": 2, "time": 0.35, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                        dev_sets=["dev-other"],
                    )
