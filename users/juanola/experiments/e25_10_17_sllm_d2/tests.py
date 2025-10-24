from sisyphus import tk

from i6_core.tools.download import DownloadJob
from .default_tools import RETURNN_EXE, RETURNN_ROOT

ROOT_RETURNN_ROOT = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}

def hf_config_download_test(test_name = "hf_config_download_test"):
    download_config_job = DownloadJob("https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/config.json",
                                      target_filename="config-Qwen2-0_5B.json")
    tk.register_output(f"test/{test_name}", download_config_job.out_file)