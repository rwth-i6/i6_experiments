from typing import Dict

from sisyphus import *

from i6_core.datasets.huggingface import DownloadAndPrepareHuggingFaceDatasetJob
from i6_experiments.users.zeineldeen.experiments.canary_aed.nemo.download import DownloadNemoModel
from i6_experiments.users.zeineldeen.experiments.canary_aed.nemo.search import SearchJob

TEST_DATASETS = ["ami", "earnings22", "gigaspeech"]
MODEL_ID = "nvidia/canary-1b"


def download_test_datasets() -> Dict[str, tk.Path]:
    # for downloading gigaspeech, a token is required. I login to huggingface and generate a token and then
    # run the command `huggingface-cli login` and paste the token

    out_dirs = {}

    for test_dataset in TEST_DATASETS:
        j = DownloadAndPrepareHuggingFaceDatasetJob(
            path="open-asr-leaderboard/datasets-test-only",
            name=test_dataset,
            split="test",
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
    return j.out_model_dir


def py():
    dataset_paths = download_test_datasets()
    model_path = download_canary_1b_model()

    search_script = tk.Path(
        "/u/zeineldeen/setups/ubuntu_22_setups/2024-06-07--canary-aed/recipe/i6_experiments/users/zeineldeen/experiments/canary_aed/nemo/run_eval.py",
        hash_overwrite="run_eval_v1",
    )
    python_exe = tk.Path(
        "/work/asr4/zeineldeen/setups-data/ubuntu_22_setups/2024-06-07--canary-aed/nemo_venv/bin/python3"
    )

    search_job = SearchJob(
        model_id=MODEL_ID,
        model_path=model_path,
        dataset_path=dataset_paths["ami"],
        dataset_name="ami",
        split="test",
        search_script=search_script,
        python_exe=python_exe,
        device="gpu",
        time_rqmt=4,
        mem_rqmt=4,
        cpu_rqmt=2,
    )
    search_job.add_alias("canary_1b_ami")
    tk.register_output("canary_1b_ami/wer", search_job.out_wer)
