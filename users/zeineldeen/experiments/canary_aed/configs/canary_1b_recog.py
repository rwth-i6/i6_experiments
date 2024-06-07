from sisyphus import *

from i6_core.datasets.huggingface import DownloadAndPrepareHuggingFaceDatasetJob

test_sets = ["ami", "earnings22", "gigaspeech"]


def download_test_datasets():
    # for downloading gigaspeech, a token is required. I login to huggingface and generate a token and then
    # run the command `huggingface-cli login` and paste the token

    for test_set in test_sets:
        j = DownloadAndPrepareHuggingFaceDatasetJob(
            path="open-asr-leaderboard/datasets-test-only",
            name=test_set,
            split="test",
            time_rqmt=24,
            mem_rqmt=4,
            cpu_rqmt=4,
            mini_task=True,
            token=True,
        )
        tk.register_output(f"datasets/{test_set}", j.out_dir)


def py():
    download_test_datasets()
