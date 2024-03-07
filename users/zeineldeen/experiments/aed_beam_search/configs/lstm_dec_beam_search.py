"""
Script to run our torch beam search implementations inside returnn with torch as engine backend
"""

from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.conformer_import_moh_att_2023_06_30 import (
    train_exp,
    config_24gb_v6,
    trafo_lm_kazuki_import,
    dyn_lr_piecewise_linear,
    model_recog_pure_torch,
    _recog,
)


lstm_dec_model = train_exp(  # 5.41
    "base-24gb-v6-lrlin1e_5_600k",
    config_24gb_v6,
    config_updates={
        "learning_rate": 1.0,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        # total steps after 2000 epochs: 982.312
        "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
        "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
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
    # TODO: effect of adaptive pruning
    for beam_search_variant in ["sep_ended_keep_v6"]:
        for beam in [20, 40, 60]:
            for prun_threshold in [10, 50]:
                for len_reward in [0.1, 0.4]:
                    recog_config = {}
                    recog_config["beam_search_version"] = beam_search_variant
                    recog_config["__batch_size_dependent"] = True
                    recog_config["__recog_def_ext"] = True
                    recog_config["beam_search_collect_individual_seq_scores"] = True
                    recog_config["beam_search_opts"] = {
                        "beam_size": beam,
                        "beam_ended_size": 1,
                        "length_reward": len_reward,
                        "pruning_threshold": prun_threshold,
                        "adaptive_pruning": True,
                    }
                    name = get_name(recog_config["beam_search_opts"])
                    _recog(
                        f"lstm_dec/{beam_search_variant}/{name}",
                        lstm_dec_model.get_last_fixed_epoch(),
                        model_recog_pure_torch,
                        recog_config,
                        search_rqmt={"cpu": 16, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                        dev_sets=["dev-other"],
                    )

    # TODO: using Trafo LM
    for beam_search_variant in [5]:
        for beam in [32, 50]:
            for lm_scale in [0.3]:
                recog_config = {}
                recog_config["beam_search_version"] = beam_search_variant
                recog_config["__batch_size_dependent"] = True
                recog_config["__recog_def_ext"] = True
                recog_config["beam_search_collect_individual_seq_scores"] = True
                recog_config["batch_size"] = 4_000 * 160
                recog_config["beam_search_opts"] = {
                    "beam_size": beam,
                    "length_normalization_exponent": 1.0,
                    "lm_scale": lm_scale,
                    "max_seq_len_factor": 0.5,
                }
                recog_config["external_language_model"] = {
                    "class": "TransformerDecoder",
                    **trafo_lm_kazuki_import.TrafoLmOpts,
                }
                recog_config["preload_from_files"] = {
                    "01_trafo_lm": {
                        "prefix": "language_model.",
                        "filename": trafo_lm_kazuki_import.get_pt_checkpoint_path(),
                    }
                }
                name = get_name(recog_config["beam_search_opts"])
                name += "_trafoLM"
                _recog(
                    f"lstm_dec/{beam_search_variant}/{name}",
                    lstm_dec_model.get_last_fixed_epoch(),
                    model_recog_pure_torch,
                    recog_config,
                    dev_sets=["dev-other"],
                )

    for beam_search_variant in ["sep_ended_keep_v6"]:
        for beam in [32, 50]:
            for prun_threshold in [10, 50]:
                for adapt in [False, True]:
                    for lm_scale in [0.3, 0.4]:
                        for len_reward in [0.2, 0.4, 1.0]:
                            recog_config = {}
                            recog_config["beam_search_version"] = beam_search_variant
                            recog_config["__batch_size_dependent"] = True
                            recog_config["__recog_def_ext"] = True
                            recog_config["beam_search_collect_individual_seq_scores"] = True
                            recog_config["batch_size"] = 4_000 * 160
                            recog_config["beam_search_opts"] = {
                                "beam_size": beam,
                                "beam_ended_size": 1,
                                "pruning_threshold": prun_threshold,
                                "adaptive_pruning": adapt,
                                "lm_scale": lm_scale,
                                "length_reward": len_reward,
                                "max_seq_len_factor": 0.5,
                            }
                            recog_config["external_language_model"] = {
                                "class": "TransformerDecoder",
                                **trafo_lm_kazuki_import.TrafoLmOpts,
                            }
                            recog_config["preload_from_files"] = {
                                "01_trafo_lm": {
                                    "prefix": "language_model.",
                                    "filename": trafo_lm_kazuki_import.get_pt_checkpoint_path(),
                                }
                            }
                            name = get_name(recog_config["beam_search_opts"])
                            name += "_trafoLM"
                            _recog(
                                f"lstm_dec/{beam_search_variant}/{name}",
                                lstm_dec_model.get_last_fixed_epoch(),
                                model_recog_pure_torch,
                                recog_config,
                                dev_sets=["dev-other"],
                            )

    for beam_search_variant in [5]:
        for beam in [32, 50]:
            for len_reward in [0.2, 0.4]:
                recog_config = {}
                recog_config["beam_search_version"] = beam_search_variant
                recog_config["__batch_size_dependent"] = True
                recog_config["__recog_def_ext"] = True
                recog_config["beam_search_collect_individual_seq_scores"] = True
                recog_config["batch_size"] = 4_000 * 160
                recog_config["beam_search_opts"] = {
                    "beam_size": beam,
                    "lm_scale": lm_scale,
                    "length_reward": len_reward,
                    "max_seq_len_factor": 0.5,
                }
                recog_config["external_language_model"] = {
                    "class": "TransformerDecoder",
                    **trafo_lm_kazuki_import.TrafoLmOpts,
                }
                recog_config["preload_from_files"] = {
                    "01_trafo_lm": {
                        "prefix": "language_model.",
                        "filename": trafo_lm_kazuki_import.get_pt_checkpoint_path(),
                    }
                }
                name = get_name(recog_config["beam_search_opts"])
                name += "_trafoLM"
                _recog(
                    f"lstm_dec/{beam_search_variant}/{name}",
                    lstm_dec_model.get_last_fixed_epoch(),
                    model_recog_pure_torch,
                    recog_config,
                    dev_sets=["dev-other"],
                )

                recog_config.pop("length_reward", None)
                recog_config["beam_search_opts"]["length_normalization_exponent"] = 1.0
                _recog(
                    f"lstm_dec/{beam_search_variant}/{name}",
                    lstm_dec_model.get_last_fixed_epoch(),
                    model_recog_pure_torch,
                    recog_config,
                    dev_sets=["dev-other"],
                )

    # # TODO: compare on isolated gpu node cn-262
    # for run in [1, 2, 3]:
    #     for threshold in [0, 5, 10, 15, 20]:
    #         for adaptive in [True, False]:
    #             for beam in [12]:
    #                 for beam_ended in [1, 12]:
    #                     for len_reward in [0.4]:
    #                         recog_config = {}
    #                         recog_config["beam_search_version"] = "sep_ended_keep_v5"
    #                         recog_config["__batch_size_dependent"] = True
    #                         recog_config["__recog_def_ext"] = True
    #                         recog_config["beam_search_collect_individual_seq_scores"] = True
    #                         recog_config["beam_search_opts"] = {
    #                             "beam_size": beam,
    #                             "length_normalization_exponent": 0.0,
    #                             "length_reward": len_reward,
    #                             "beam_ended_size": beam_ended,
    #                             "pruning_threshold": threshold,
    #                             "adaptive_pruning": adaptive,
    #                         }
    #                         recog_config["run"] = run  # to break the hashes
    #                         name = get_name(recog_config["beam_search_opts"])
    #                         _recog(
    #                             f"base-24gb-v6-lrlin1e_5_600k/sep_ended_keep_v5/{name}_run{run}",
    #                             lstm_dec_model.get_last_fixed_epoch(),
    #                             model_recog_pure_torch,
    #                             recog_config,
    #                             search_rqmt={"cpu": 16, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
    #                             dev_sets=["dev-other"],
    #                         )

    # for beam_search_variant in [5, "sep_ended"]:
    #     for run in [1, 2, 3, 4, 5, 6]:
    #         for beam in [12]:
    #             for len_exp in [1.0]:
    #                 recog_config = {}
    #                 recog_config["beam_search_version"] = beam_search_variant
    #                 recog_config["__batch_size_dependent"] = True
    #                 recog_config["__recog_def_ext"] = True
    #                 recog_config["beam_search_collect_individual_seq_scores"] = True
    #                 recog_config["beam_search_opts"] = {
    #                     "beam_size": beam,
    #                     "length_normalization_exponent": len_exp,
    #                 }
    #                 if beam_search_variant == "sep_ended":
    #                     recog_config["beam_search_opts"]["beam_and_ended_size"] = beam
    #                 recog_config["run"] = run  # to break the hashes
    #                 name = get_name(recog_config["beam_search_opts"])
    #                 _recog(
    #                     f"base-24gb-v6-lrlin1e_5_600k/{beam_search_variant}/{name}_run{run}",
    #                     lstm_dec_model.get_last_fixed_epoch(),
    #                     model_recog_pure_torch,
    #                     recog_config,
    #                     search_rqmt={"cpu": 16, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
    #                     dev_sets=["dev-other"],
    #                 )
    #             for len_reward in [0.4]:
    #                 recog_config = {}
    #                 recog_config["beam_search_version"] = beam_search_variant
    #                 recog_config["__batch_size_dependent"] = True
    #                 recog_config["__recog_def_ext"] = True
    #                 recog_config["beam_search_collect_individual_seq_scores"] = True
    #                 recog_config["beam_search_opts"] = {
    #                     "beam_size": beam,
    #                     "length_normalization_exponent": 0.0,
    #                     "length_reward": len_reward,
    #                 }
    #                 if beam_search_variant == "sep_ended":
    #                     recog_config["beam_search_opts"]["beam_and_ended_size"] = beam
    #                 recog_config["run"] = run  # to break the hashes
    #                 name = get_name(recog_config["beam_search_opts"])
    #                 _recog(
    #                     f"base-24gb-v6-lrlin1e_5_600k/{beam_search_variant}/{name}_run{run}",
    #                     lstm_dec_model.get_last_fixed_epoch(),
    #                     model_recog_pure_torch,
    #                     recog_config,
    #                     search_rqmt={"cpu": 16, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
    #                     dev_sets=["dev-other"],
    #                 )
