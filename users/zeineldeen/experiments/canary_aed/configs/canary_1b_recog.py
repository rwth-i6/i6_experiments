from typing import Dict

from sisyphus import *

from i6_core.datasets.huggingface import DownloadAndPrepareHuggingFaceDatasetJob
from i6_experiments.users.zeineldeen.experiments.canary_aed.nemo.download import DownloadNemoModel
from i6_experiments.users.zeineldeen.experiments.canary_aed.nemo.search import SearchJob
from i6_experiments.users.zeineldeen.experiments.canary_aed.nemo.stats import ComputePruningStatsJob

TEST_DATASETS = {"ami": "test", "earnings22": "test", "gigaspeech": "test", "librispeech": "test.other"}

MODEL_ID = "nvidia/canary-1b"


def download_test_datasets() -> Dict[str, tk.Path]:
    # for downloading gigaspeech, a token is required. I login to huggingface and generate a token and then
    # run the command `huggingface-cli login` and paste the token

    out_dirs = {}

    for test_dataset, split in TEST_DATASETS.items():
        j = DownloadAndPrepareHuggingFaceDatasetJob(
            path="open-asr-leaderboard/datasets-test-only",
            name=test_dataset,
            split=split,
            time_rqmt=24,
            mem_rqmt=4,
            cpu_rqmt=4,
            mini_task=True,
            token=True,
        )
        out_dirs[test_dataset] = j.out_dir
        tk.register_output(f"datasets/{test_dataset}", j.out_dir)

    return out_dirs


def download_canary_1b_model() -> tk.Path:
    j = DownloadNemoModel(model_id=MODEL_ID, device=-1)
    tk.register_output("canary_1b_nemo_model", j.out_model_dir)
    # return j.out_model_dir
    # TODO: let the job returns directly the model path instead
    return tk.Path(
        j.out_model_dir.get_path()
        + "/models--nvidia--canary-1b/snapshots/dd32c0c709e2bfc79f583e16b9df4b3a160f7e86/canary-1b.nemo"
    )


def py():
    dataset_paths = download_test_datasets()
    model_path = download_canary_1b_model()

    # to run canary model, this env has installed nemo toolkit with:
    #   pip3 install git+https://github.com/NVIDIA/NeMo.git@r2.0.0rc0#egg=nemo_toolkit[all]
    # related issue: https://github.com/huggingface/open_asr_leaderboard/issues/26
    python_exe = tk.Path(
        "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2024-06-07--canary-aed/nemo_venv/bin/python3"
    )

    def get_search_job(
        *, test_set: str, split: str, search_script: tk.Path, extra_search_args: dict = None, cache_dir_name_suffix: str
    ):
        default_search_args = {"pnc": False, "max_eval_samples": -1}
        if extra_search_args is None:
            extra_search_args = {}
        default_search_args.update(extra_search_args)
        search_job = SearchJob(
            model_id=MODEL_ID,
            model_path=model_path,
            dataset_path=dataset_paths[test_set],
            dataset_name=test_set,
            cache_dir_name_suffix=cache_dir_name_suffix,
            split=split,
            search_script=search_script,
            search_args=default_search_args,
            python_exe=python_exe,
            device="gpu",
            time_rqmt=24,
            mem_rqmt=8,
            cpu_rqmt=4,
        )
        return search_job

    # Greedy decoding:
    #
    # testset     | ours  | huggingface
    # -----------------------------------
    # libri       | 2.95  | 2.94
    # ami         | 13.96 | 14.0
    # earnings22  | 12.23 | 12.25
    # gigaspeech  | 10.14 | 10.19
    for run in range(1):
        huggface_search_script = tk.Path(
            "/u/zeineldeen/setups/ubuntu_22_setups/2024-06-07--canary-aed/recipe/i6_experiments/users/zeineldeen/experiments/canary_aed/nemo/run_eval.py",
            hash_overwrite=f"run_eval_v1_rtf_run{run}",
        )
        for test_set, split in TEST_DATASETS.items():
            for bs in [16, 32, 64]:
                name = f"{test_set}_bs{bs}_greedy_run{run}"
                search_job = get_search_job(
                    test_set=test_set,
                    split=split,
                    search_script=huggface_search_script,
                    extra_search_args={"batch_size": bs},
                    cache_dir_name_suffix=name,
                )
                search_job.rqmt["sbatch_args"] = ["-p", "gpu_test_24gb", "-w", "cn-290"]
                search_job.add_alias(f"canary_1b/huggingface/{name}")
                tk.register_output(f"canary_1b/huggingface/{name}/search_out", search_job.out_search_results)
                tk.register_output(f"canary_1b/huggingface/{name}/wer", search_job.out_wer)

                if bs == 16 and test_set == "ami":
                    for beam_size in [4, 8, 12]:
                        name = f"{test_set}_bs{bs}_run{run}_beam{beam_size}"
                        search_job = get_search_job(
                            test_set=test_set,
                            split=split,
                            search_script=huggface_search_script,
                            extra_search_args={"batch_size": bs, "beam_size": beam_size},
                            cache_dir_name_suffix=name,
                        )
                        search_job.rqmt["sbatch_args"] = ["-p", "gpu_test_24gb", "-w", "cn-290"]
                        search_job.add_alias(f"canary_1b/huggingface/{name}")
                        tk.register_output(f"canary_1b/huggingface/{name}/search_out", search_job.out_search_results)
                        tk.register_output(f"canary_1b/huggingface/{name}/wer", search_job.out_wer)

    # TODO: run with our beam search

    for run in range(2):
        our_beam_search_script = tk.Path(
            "/u/zeineldeen/setups/ubuntu_22_setups/2024-06-07--canary-aed/recipe/i6_experiments/users/zeineldeen/experiments/canary_aed/nemo/run_eval_beam_search.py",
            hash_overwrite=f"run_eval_v2_rtf_run{run}",
        )
        # TODO: just simple beam search w/wo length norm
        for test_set, split in TEST_DATASETS.items():
            if run != 0:
                continue
            if test_set.lower() not in ["ami"]:
                continue
            for beam_size in [1, 4, 8, 12]:
                for bs in [16]:
                    for len_reward in [0.0, 0.1]:
                        name = f"{test_set}_bs{bs}_beam{beam_size}_run{run}"
                        extra_search_args = {"beam_size": beam_size, "batch_size": bs}
                        if len_reward:
                            extra_search_args["length_normalization_exponent"] = 0.0
                            extra_search_args["length_reward"] = len_reward
                            name += f"_lenReward{len_reward}"
                        else:
                            extra_search_args["length_normalization_exponent"] = 1.0
                            name += "_lenNorm"
                        search_job = get_search_job(
                            test_set=test_set,
                            split=split,
                            search_script=our_beam_search_script,
                            extra_search_args=extra_search_args,
                            cache_dir_name_suffix=name,
                        )
                        search_job.rqmt["sbatch_args"] = ["-p", "gpu_test_24gb", "-w", "cn-290"]
                        search_job.add_alias(f"canary_1b/beam_search_v5/{name}")
                        tk.register_output(f"canary_1b/beam_search_v5/{name}/search_out", search_job.out_search_results)
                        tk.register_output(f"canary_1b/beam_search_v5/{name}/wer", search_job.out_wer)

                        stats_job = ComputePruningStatsJob(search_out=search_job.out_search_results)
                        tk.register_output(
                            f"canary_1b/beam_search_v5/{name}/stats/avg_num_steps_per_seq",
                            stats_job.out_avg_num_steps_per_seq,
                        )
                        tk.register_output(
                            f"canary_1b/beam_search_v5/{name}/stats/avg_num_act_hyps_per_step",
                            stats_job.out_avg_num_act_hyps_per_step,
                        )

        # TODO: simple beam search + adaptive pruning
        for test_set, split in TEST_DATASETS.items():
            if test_set.lower() not in ["ami"]:
                continue
            for beam_size in [4, 8, 12]:
                for pruning_thre in [10, 20]:
                    for bs in [16]:
                        for len_reward in [0.1]:
                            name = f"{test_set}_bs{bs}_beam{beam_size}_run{run}_prune{pruning_thre}_adaptive_lenReward{len_reward}"
                            extra_search_args = {
                                "beam_size": beam_size,
                                "length_normalization_exponent": 0.0,
                                "length_reward": len_reward,
                                "pruning_threshold": pruning_thre,
                                "adaptive_pruning": True,
                                "batch_size": bs,
                            }
                            search_job = get_search_job(
                                test_set=test_set,
                                split=split,
                                search_script=our_beam_search_script,
                                extra_search_args=extra_search_args,
                                cache_dir_name_suffix=name,
                            )
                            search_job.rqmt["sbatch_args"] = ["-p", "gpu_test_24gb", "-w", "cn-290"]
                            search_job.add_alias(f"canary_1b/beam_search_v5/{name}")
                            tk.register_output(
                                f"canary_1b/beam_search_v5/{name}/search_out", search_job.out_search_results
                            )
                            tk.register_output(f"canary_1b/beam_search_v5/{name}/wer", search_job.out_wer)

                            stats_job = ComputePruningStatsJob(search_out=search_job.out_search_results)
                            tk.register_output(
                                f"canary_1b/beam_search_v5/{name}/stats/avg_num_steps_per_seq",
                                stats_job.out_avg_num_steps_per_seq,
                            )
                            tk.register_output(
                                f"canary_1b/beam_search_v5/{name}/stats/avg_num_act_hyps_per_step",
                                stats_job.out_avg_num_act_hyps_per_step,
                            )

                            # TODO: sep_ended_keep beam search
                            search_job = get_search_job(
                                test_set=test_set,
                                split=split,
                                search_script=our_beam_search_script,
                                extra_search_args={"beam_search_strategy": "sep_ended_keep", **extra_search_args},
                                cache_dir_name_suffix=name,
                            )
                            search_job.rqmt["sbatch_args"] = ["-p", "gpu_test_24gb", "-w", "cn-290"]
                            search_job.add_alias(f"canary_1b/sep_ended_keep_v6/{name}")
                            tk.register_output(
                                f"canary_1b/sep_ended_keep_v6/{name}/search_out", search_job.out_search_results
                            )
                            tk.register_output(f"canary_1b/sep_ended_keep_v6/{name}/wer", search_job.out_wer)

                            stats_job = ComputePruningStatsJob(search_out=search_job.out_search_results)
                            tk.register_output(
                                f"canary_1b/sep_ended_keep_v6/{name}/stats/avg_num_steps_per_seq",
                                stats_job.out_avg_num_steps_per_seq,
                            )
                            tk.register_output(
                                f"canary_1b/sep_ended_keep_v6/{name}/stats/avg_num_act_hyps_per_step",
                                stats_job.out_avg_num_act_hyps_per_step,
                            )
