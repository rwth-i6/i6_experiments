"""
Script to run our torch beam search implementations inside returnn with torch as engine backend
"""

from i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.espnet import (
    _recog,
    model_recog_our,
    _get_orig_e_branchformer_model,
    _sis_setup_global_prefix,
    _get_orig_e_branchformer_lm_model_config,
    _get_orig_e_branchformer_lm_model_preload_opts,
)

_sis_setup_global_prefix()
model = _get_orig_e_branchformer_model()


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
    # TODO: v5 + CTC + LM
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in [5]:
            for beam in [20]:
                recog_config = dict()
                recog_config["beam_search_version"] = beam_search_variant
                recog_config["__batch_size_dependent"] = True
                recog_config["__recog_def_ext"] = True
                recog_config["beam_search_collect_individual_seq_scores"] = True

                recog_config["batch_size"] = 5000 * 160  # CTC
                recog_config["beam_search_opts"] = {
                    "beam_size": beam,
                    "ctc_weight": 0.3,
                    "max_seq_len_factor": max_seq_len_factor,
                }
                name = get_name(recog_config["beam_search_opts"])
                # _recog(
                #     f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{5000}",
                #     model,
                #     model_recog_our,
                #     recog_config=recog_config,
                #     vocab="spm_espnet_5k",
                #     audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                #     dev_sets=["dev-other", "test-other"],
                #     audio_format="old_flac_tar_zip",
                # )

                recog_config["max_seqs"] = 1
                name = get_name(recog_config["beam_search_opts"])
                _recog(
                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_maxseqs1",
                    model,
                    model_recog_our,
                    recog_config=recog_config,
                    vocab="spm_espnet_5k",
                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                    audio_format="old_flac_tar_zip",
                    # search_rqmt={"sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                    dev_sets=["dev-other", "test-other"],
                )

                recog_config["beam_search_opts"]["lm_scale"] = 0.6
                recog_config["batch_size"] = 2000 * 160
                recog_config.update(_get_orig_e_branchformer_lm_model_config())
                recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                # name = get_name(recog_config["beam_search_opts"])
                # _recog(
                #     f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{batch_size}",
                #     model,
                #     model_recog_our,
                #     recog_config=recog_config,
                #     vocab="spm_espnet_5k",
                #     audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                #     dev_sets=["dev-other", "test-other"],
                #     audio_format="old_flac_tar_zip",
                # )

                name = get_name(recog_config["beam_search_opts"])
                recog_config["max_seqs"] = 1
                _recog(
                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_maxseqs1",
                    model,
                    model_recog_our,
                    recog_config=recog_config,
                    vocab="spm_espnet_5k",
                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                    audio_format="old_flac_tar_zip",
                    # search_rqmt={"sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                    dev_sets=["dev-other", "test-other"],
                )

    # TODO: sep_ended + CTC
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended"]:
            for beam in [20]:
                for use_end_detect in [True, False]:
                    for beam_and_ended_size in [1000, 20]:
                        if beam_and_ended_size == 1000 and not use_end_detect:
                            continue  # does not make sense
                        recog_config = dict()
                        recog_config["beam_search_version"] = beam_search_variant
                        recog_config["__batch_size_dependent"] = True
                        recog_config["__recog_def_ext"] = True
                        recog_config["beam_search_collect_individual_seq_scores"] = True

                        recog_config["batch_size"] = 5000 * 160  # CTC
                        recog_config["beam_search_opts"] = {
                            "beam_size": beam,
                            "beam_and_ended_size": beam_and_ended_size,
                            "ctc_weight": 0.3,
                            "max_seq_len_factor": max_seq_len_factor,
                            "use_espnet_end_detect": use_end_detect,
                        }
                        name = get_name(recog_config["beam_search_opts"])
                        recog_config["run"] = 2
                        _recog(
                            f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_run2",
                            model,
                            model_recog_our,
                            recog_config=recog_config,
                            vocab="spm_espnet_5k",
                            audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                            dev_sets=["dev-other", "test-other"],
                            audio_format="old_flac_tar_zip",
                        )

                        recog_config["beam_search_opts"]["lm_scale"] = 0.6
                        recog_config.update(_get_orig_e_branchformer_lm_model_config())
                        recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                        name = get_name(recog_config["beam_search_opts"])
                        recog_config["batch_size"] = 2000 * 160
                        _recog(
                            f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{2000}",
                            model,
                            model_recog_our,
                            recog_config=recog_config,
                            vocab="spm_espnet_5k",
                            audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                            dev_sets=["dev-other", "test-other"],
                            audio_format="old_flac_tar_zip",
                        )

    # TODO: sep_ended_keep + CTC
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [20]:
                for ctc_weight in [0.3]:
                    for pruning_threshold in [0, 5, 10, 20, 50, 1000000]:
                        for adaptive in [True, False]:
                            for len_reward in [0.1]:
                                recog_config = dict()
                                recog_config["beam_search_version"] = beam_search_variant
                                recog_config["__batch_size_dependent"] = True
                                recog_config["__recog_def_ext"] = True

                                recog_config["batch_size"] = 5000 * 160  # CTC
                                recog_config["beam_search_collect_individual_seq_scores"] = True
                                recog_config["beam_search_opts"] = {
                                    "beam_size": beam,
                                    "length_reward": len_reward,
                                    "ctc_weight": ctc_weight,
                                    "max_seq_len_factor": max_seq_len_factor,
                                    "pruning_threshold": pruning_threshold,
                                    "adaptive_pruning": adaptive,
                                    "beam_ended_size": 1,
                                }
                                name = get_name(recog_config["beam_search_opts"])
                                _recog(
                                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                    model,
                                    model_recog_our,
                                    recog_config=recog_config,
                                    vocab="spm_espnet_5k",
                                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                    audio_format="old_flac_tar_zip",
                                    dev_sets=["dev-other"],
                                    search_rqmt={"time": 24} if pruning_threshold == 1000000 else None,
                                )

                                if adaptive and pruning_threshold == 5:
                                    recog_config["max_seqs"] = 1
                                    _recog(
                                        f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_gpu_maxseqs1",
                                        model,
                                        model_recog_our,
                                        recog_config=recog_config,
                                        vocab="spm_espnet_5k",
                                        audio_opts={
                                            "peak_normalization": False
                                        },  # speech_volume_normalize=False in ESPnet
                                        audio_format="old_flac_tar_zip",
                                        dev_sets=["dev-other", "test-other"],
                                        # search_rqmt={"sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                                    )

                                    _recog(
                                        f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_maxseqs1",
                                        model,
                                        model_recog_our,
                                        recog_config=recog_config,
                                        vocab="spm_espnet_5k",
                                        audio_opts={
                                            "peak_normalization": False
                                        },  # speech_volume_normalize=False in ESPnet
                                        audio_format="old_flac_tar_zip",
                                        dev_sets=["dev-other", "test-other"],
                                        # search_rqmt={"sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                                    )

                                # TODO: reserved node
                                # _recog(
                                #     f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_cn262",
                                #     model,
                                #     model_recog_our,
                                #     recog_config=recog_config,
                                #     vocab="spm_espnet_5k",
                                #     audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                #     audio_format="old_flac_tar_zip",
                                #     dev_sets=["dev-other"],
                                #     search_rqmt={"cpu": 4, "sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                                # )

    # TODO: sep_ended_keep + CTC+LM adaptive/non-adaptive
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [20, 40]:
                for ctc_weight in [0.3]:
                    for lm_weight in [0.6]:
                        for pruning_threshold in [0, 5, 10, 20, 1000000]:
                            for adaptive in [True, False]:
                                for len_reward in [1.0]:
                                    assert lm_weight
                                    recog_config = dict()
                                    recog_config["beam_search_version"] = beam_search_variant
                                    recog_config["__batch_size_dependent"] = True
                                    recog_config["__recog_def_ext"] = True
                                    recog_config["beam_search_collect_individual_seq_scores"] = True

                                    if beam < 50:
                                        recog_config["batch_size"] = 2000 * 160  # LM + CTC
                                    else:
                                        recog_config["batch_size"] = 1000 * 160
                                    recog_config["max_seqs"] = 20

                                    recog_config["beam_search_opts"] = {
                                        "beam_size": beam,
                                        "length_reward": len_reward,
                                        "max_seq_len_factor": max_seq_len_factor,
                                        "ctc_weight": ctc_weight,
                                        "lm_scale": lm_weight,
                                        "pruning_threshold": pruning_threshold,
                                        "adaptive_pruning": adaptive,
                                        "beam_ended_size": 1,
                                    }
                                    recog_config.update(_get_orig_e_branchformer_lm_model_config())
                                    recog_config[
                                        "preload_from_files"
                                    ] = _get_orig_e_branchformer_lm_model_preload_opts()

                                    name = get_name(recog_config["beam_search_opts"])
                                    _recog(
                                        f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                        model,
                                        model_recog_our,
                                        recog_config=recog_config,
                                        vocab="spm_espnet_5k",
                                        audio_opts={
                                            "peak_normalization": False
                                        },  # speech_volume_normalize=False in ESPnet
                                        dev_sets=["dev-other"],
                                        audio_format="old_flac_tar_zip",
                                        search_rqmt={"time": 24} if pruning_threshold == 1000000 else None,
                                    )

                                    # run on other sets
                                    if beam == 20 and len_reward == 1.0 and pruning_threshold in [5, 10] and adaptive:
                                        # _recog(
                                        #     f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                        #     model,
                                        #     model_recog_our,
                                        #     recog_config=recog_config,
                                        #     vocab="spm_espnet_5k",
                                        #     audio_opts={
                                        #         "peak_normalization": False
                                        #     },  # speech_volume_normalize=False in ESPnet
                                        #     audio_format="old_flac_tar_zip",
                                        # )

                                        recog_config["max_seqs"] = 1
                                        recog_config["run"] = 2
                                        _recog(
                                            f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_cpu_run2",
                                            model,
                                            model_recog_our,
                                            recog_config=recog_config,
                                            vocab="spm_espnet_5k",
                                            audio_opts={
                                                "peak_normalization": False
                                            },  # speech_volume_normalize=False in ESPnet
                                            audio_format="old_flac_tar_zip",
                                            search_rqmt={
                                                "cpu": 4,
                                                "gpu": 0,
                                                "time": 24,
                                                "sbatch_args": ["-p", "rescale_intel", "-A", "rescale_speed"],
                                            },
                                            dev_sets=["dev-other"],
                                        )

                                        if pruning_threshold != 5:
                                            continue
                                        recog_config["run"] = 1
                                        _recog(
                                            f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_maxseqs1",
                                            model,
                                            model_recog_our,
                                            recog_config=recog_config,
                                            vocab="spm_espnet_5k",
                                            audio_opts={
                                                "peak_normalization": False
                                            },  # speech_volume_normalize=False in ESPnet
                                            audio_format="old_flac_tar_zip",
                                            # search_rqmt={"sbatch_args": ["-w", "cn-262", "--reservation", "hlt_4"]},
                                            dev_sets=["dev-other", "test-other"],
                                        )

    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [1, 4, 10, 20, 30]:
                for ctc_weight in [0.3]:
                    for lm_weight in [0.6]:
                        for pruning_threshold in [5, 10, 20]:
                            for adaptive in [True, False]:
                                for len_reward in [1.0]:
                                    assert lm_weight
                                    recog_config = dict()
                                    recog_config["beam_search_version"] = beam_search_variant
                                    recog_config["__batch_size_dependent"] = True
                                    recog_config["__recog_def_ext"] = True
                                    recog_config["beam_search_collect_individual_seq_scores"] = True

                                    if beam < 50:
                                        recog_config["batch_size"] = 2000 * 160  # LM + CTC
                                    else:
                                        recog_config["batch_size"] = 1000 * 160
                                    recog_config["max_seqs"] = 20

                                    recog_config["beam_search_opts"] = {
                                        "beam_size": beam,
                                        "length_reward": len_reward,
                                        "max_seq_len_factor": max_seq_len_factor,
                                        "ctc_weight": ctc_weight,
                                        "lm_scale": lm_weight,
                                        "pruning_threshold": pruning_threshold,
                                        "adaptive_pruning": adaptive,
                                        "beam_ended_size": 1,
                                    }
                                    recog_config.update(_get_orig_e_branchformer_lm_model_config())
                                    recog_config[
                                        "preload_from_files"
                                    ] = _get_orig_e_branchformer_lm_model_preload_opts()

                                    name = get_name(recog_config["beam_search_opts"])
                                    if beam == 4 or beam == 10:
                                        recog_config["run"] = 2
                                        name += "_run2"
                                    _recog(
                                        f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                        model,
                                        model_recog_our,
                                        recog_config=recog_config,
                                        vocab="spm_espnet_5k",
                                        audio_opts={
                                            "peak_normalization": False
                                        },  # speech_volume_normalize=False in ESPnet
                                        dev_sets=["dev-other"],
                                        audio_format="old_flac_tar_zip",
                                    )

    # TODO: sep_ended_keep pure AED
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [1, 4, 10, 20, 30, 40]:
                for ctc_weight in [0.0]:
                    for lm_weight in [0.0]:
                        for pruning_threshold in [10]:
                            for adaptive in [True, False]:
                                for len_reward in [0.1, 0.2]:
                                    recog_config = dict()
                                    recog_config["beam_search_version"] = beam_search_variant
                                    recog_config["__batch_size_dependent"] = True
                                    recog_config["__recog_def_ext"] = True
                                    recog_config["beam_search_collect_individual_seq_scores"] = True

                                    if beam < 50:
                                        recog_config["batch_size"] = 2000 * 160  # LM + CTC
                                    else:
                                        recog_config["batch_size"] = 1000 * 160
                                    recog_config["max_seqs"] = 20

                                    recog_config["beam_search_opts"] = {
                                        "beam_size": beam,
                                        "length_reward": len_reward,
                                        "max_seq_len_factor": max_seq_len_factor,
                                        "pruning_threshold": pruning_threshold,
                                        "adaptive_pruning": adaptive,
                                        "ctc_weight": ctc_weight,
                                        "beam_ended_size": 1,
                                    }
                                    # recog_config.update(_get_orig_e_branchformer_lm_model_config())
                                    # recog_config[
                                    #     "preload_from_files"
                                    # ] = _get_orig_e_branchformer_lm_model_preload_opts()
                                    #
                                    name = get_name(recog_config["beam_search_opts"])
                                    # if beam == 4 or beam == 10:
                                    #     recog_config["run"] = 2
                                    #     name += "_run2"
                                    _recog(
                                        f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                        model,
                                        model_recog_our,
                                        recog_config=recog_config,
                                        vocab="spm_espnet_5k",
                                        audio_opts={
                                            "peak_normalization": False
                                        },  # speech_volume_normalize=False in ESPnet
                                        dev_sets=["dev-other"],
                                        audio_format="old_flac_tar_zip",
                                    )

    for max_seq_len_factor in [0.5]:
        for beam_search_variant in [5]:
            for beam in [1, 4, 10, 20, 30, 40]:
                recog_config = dict()
                recog_config["beam_search_version"] = beam_search_variant
                recog_config["__batch_size_dependent"] = True
                recog_config["__recog_def_ext"] = True
                recog_config["beam_search_collect_individual_seq_scores"] = True

                recog_config["beam_search_opts"] = {
                    "beam_size": beam,
                    "ctc_weight": 0.3,
                    "max_seq_len_factor": max_seq_len_factor,
                }
                recog_config["beam_search_opts"]["lm_scale"] = 0.6
                recog_config["batch_size"] = 2000 * 160
                recog_config.update(_get_orig_e_branchformer_lm_model_config())
                recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                name = get_name(recog_config["beam_search_opts"])
                _recog(
                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                    model,
                    model_recog_our,
                    recog_config=recog_config,
                    vocab="spm_espnet_5k",
                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                    dev_sets=["dev-other"],
                    audio_format="old_flac_tar_zip",
                )

    # TODO: pruning in v5
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in [5]:
            for beam in [20]:
                for pruning_threshold in [0, 5, 10, 20, 1000000]:
                    for adaptive in [True, False]:
                        for len_reward in [0.1, None]:
                            recog_config = dict()
                            recog_config["beam_search_version"] = beam_search_variant
                            recog_config["__batch_size_dependent"] = True
                            recog_config["__recog_def_ext"] = True
                            recog_config["beam_search_collect_individual_seq_scores"] = True

                            recog_config["batch_size"] = 5000 * 160  # CTC
                            recog_config["beam_search_opts"] = {
                                "beam_size": beam,
                                "pruning_threshold": pruning_threshold,
                                "adaptive_pruning": adaptive,
                                "ctc_weight": 0.3,
                                "max_seq_len_factor": max_seq_len_factor,
                            }
                            if len_reward:
                                recog_config["beam_search_opts"]["length_reward"] = len_reward
                            name = get_name(recog_config["beam_search_opts"])
                            _recog(
                                f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_run2",
                                model,
                                model_recog_our,
                                recog_config=recog_config,
                                vocab="spm_espnet_5k",
                                audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                dev_sets=["dev-other", "test-other"],
                                audio_format="old_flac_tar_zip",
                            )

                        for len_reward in [1.0, None]:
                            recog_config["beam_search_opts"]["lm_scale"] = 0.6
                            recog_config.update(_get_orig_e_branchformer_lm_model_config())
                            recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                            recog_config["batch_size"] = 2000 * 160
                            if len_reward:
                                recog_config["beam_search_opts"]["length_reward"] = len_reward
                            name = get_name(recog_config["beam_search_opts"])
                            _recog(
                                f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{2000}",
                                model,
                                model_recog_our,
                                recog_config=recog_config,
                                vocab="spm_espnet_5k",
                                audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                dev_sets=["dev-other", "test-other"],
                                audio_format="old_flac_tar_zip",
                            )

    # TODO: simple + adaptive threshold pruning
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in [5]:
            for beam in [1, 4, 10, 20, 30, 40]:
                for pruning_threshold in [5, 10, 20]:
                    for adaptive in [True, False]:
                        for len_reward in [None]:
                            recog_config = dict()
                            recog_config["beam_search_version"] = beam_search_variant
                            recog_config["__batch_size_dependent"] = True
                            recog_config["__recog_def_ext"] = True
                            recog_config["beam_search_collect_individual_seq_scores"] = True

                            recog_config["batch_size"] = 5000 * 160  # CTC
                            recog_config["beam_search_opts"] = {
                                "beam_size": beam,
                                "pruning_threshold": pruning_threshold,
                                "adaptive_pruning": adaptive,
                                "ctc_weight": 0.3,
                                "max_seq_len_factor": max_seq_len_factor,
                            }
                            if len_reward:
                                recog_config["beam_search_opts"]["length_reward"] = len_reward
                            # name = get_name(recog_config["beam_search_opts"])
                            # _recog(
                            #     f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_run2",
                            #     model,
                            #     model_recog_our,
                            #     recog_config=recog_config,
                            #     vocab="spm_espnet_5k",
                            #     audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                            #     dev_sets=["dev-other", "test-other"],
                            #     audio_format="old_flac_tar_zip",
                            # )

                        for len_reward in [None]:
                            recog_config["beam_search_opts"]["lm_scale"] = 0.6
                            recog_config.update(_get_orig_e_branchformer_lm_model_config())
                            recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                            recog_config["batch_size"] = 2000 * 160
                            if len_reward:
                                recog_config["beam_search_opts"]["length_reward"] = len_reward
                            name = get_name(recog_config["beam_search_opts"])
                            _recog(
                                f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{2000}",
                                model,
                                model_recog_our,
                                recog_config=recog_config,
                                vocab="spm_espnet_5k",
                                audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                dev_sets=["dev-other"],
                                audio_format="old_flac_tar_zip",
                            )

                        if beam == 20 and pruning_threshold == 10 and adaptive:
                            recog_config["max_seqs"] = 1
                            name = get_name(recog_config["beam_search_opts"])
                            _recog(
                                f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{2000}_maxseqs1",
                                model,
                                model_recog_our,
                                recog_config=recog_config,
                                vocab="spm_espnet_5k",
                                audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                dev_sets=["dev-other"],
                                audio_format="old_flac_tar_zip",
                            )

                            name = get_name(recog_config["beam_search_opts"])
                            recog_config["run"] = "cpu_run_1"
                            _recog(
                                f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{2000}_maxseqs1_cpu",
                                model,
                                model_recog_our,
                                recog_config=recog_config,
                                vocab="spm_espnet_5k",
                                audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                dev_sets=["dev-other"],
                                audio_format="old_flac_tar_zip",
                                search_rqmt={
                                    "cpu": 4,
                                    "gpu": 0,
                                    "time": 24,
                                    "sbatch_args": ["-p", "rescale_intel", "-A", "rescale_speed"],
                                },
                            )

    # TODO: vary beam_and_ended_size
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended"]:
            for beam in [20]:
                for beam_and_ended_size in [21]:
                    for use_len_reward in [True, False]:
                        recog_config = dict()
                        recog_config["beam_search_version"] = beam_search_variant
                        recog_config["__batch_size_dependent"] = True
                        recog_config["__recog_def_ext"] = True
                        recog_config["beam_search_collect_individual_seq_scores"] = True

                        recog_config["batch_size"] = 5000 * 160  # CTC
                        recog_config["beam_search_opts"] = {
                            "beam_size": beam,
                            "beam_and_ended_size": beam_and_ended_size,
                            "ctc_weight": 0.3,
                            "max_seq_len_factor": max_seq_len_factor,
                        }
                        if use_len_reward:
                            recog_config["beam_search_opts"]["length_reward"] = 0.1
                        name = get_name(recog_config["beam_search_opts"])
                        recog_config["run"] = 3
                        _recog(
                            f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_run3",
                            model,
                            model_recog_our,
                            recog_config=recog_config,
                            vocab="spm_espnet_5k",
                            audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                            dev_sets=["dev-other", "test-other"],
                            audio_format="old_flac_tar_zip",
                        )

                        recog_config["beam_search_opts"]["lm_scale"] = 0.6
                        recog_config["batch_size"] = 2000 * 160
                        if use_len_reward:
                            recog_config["beam_search_opts"]["length_reward"] = 1.0
                        recog_config.update(_get_orig_e_branchformer_lm_model_config())
                        recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()
                        name = get_name(recog_config["beam_search_opts"])
                        _recog(
                            f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}_batch{2000}_run3",
                            model,
                            model_recog_our,
                            recog_config=recog_config,
                            vocab="spm_espnet_5k",
                            audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                            dev_sets=["dev-other", "test-other"],
                            audio_format="old_flac_tar_zip",
                        )

    # TODO: pruning worst
    for max_seq_len_factor in [0.5]:
        for beam_search_variant in ["sep_ended_keep_v6"]:
            for beam in [20]:
                for ctc_weight in [0.3]:
                    for lm_weight in [0.6]:
                        for pruning_threshold_worst in [0]:
                            for use_len_reward in [True, False]:
                                assert lm_weight
                                recog_config = dict()
                                recog_config["beam_search_version"] = beam_search_variant
                                recog_config["__batch_size_dependent"] = True
                                recog_config["__recog_def_ext"] = True
                                recog_config["beam_search_collect_individual_seq_scores"] = True

                                recog_config["beam_search_opts"] = {
                                    "beam_size": beam,
                                    "max_seq_len_factor": max_seq_len_factor,
                                    "ctc_weight": ctc_weight,
                                    "pruning_threshold_worst": pruning_threshold_worst,
                                    "beam_ended_size": beam,
                                }
                                if use_len_reward:
                                    recog_config["beam_search_opts"]["length_reward"] = 0.1

                                recog_config["batch_size"] = 5_000 * 160
                                name = get_name(recog_config["beam_search_opts"])
                                _recog(
                                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                    model,
                                    model_recog_our,
                                    recog_config=recog_config,
                                    vocab="spm_espnet_5k",
                                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                    dev_sets=["dev-other", "test-other"],
                                    audio_format="old_flac_tar_zip",
                                )

                                recog_config["batch_size"] = 2_000 * 160
                                recog_config["beam_search_opts"]["lm_scale"] = lm_weight
                                if use_len_reward:
                                    recog_config["beam_search_opts"]["length_reward"] = 1.0
                                recog_config.update(_get_orig_e_branchformer_lm_model_config())
                                recog_config["preload_from_files"] = _get_orig_e_branchformer_lm_model_preload_opts()

                                name = get_name(recog_config["beam_search_opts"])
                                _recog(
                                    f"e_branchformer_raw_en_bpe5000_sp_flac/{beam_search_variant}/{name}",
                                    model,
                                    model_recog_our,
                                    recog_config=recog_config,
                                    vocab="spm_espnet_5k",
                                    audio_opts={"peak_normalization": False},  # speech_volume_normalize=False in ESPnet
                                    dev_sets=["dev-other", "test-other"],
                                    audio_format="old_flac_tar_zip",
                                )
