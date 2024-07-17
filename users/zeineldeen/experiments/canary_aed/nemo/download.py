from sisyphus import *

import os


class DownloadNemoModel(Job):
    def __init__(self, model_id: str, device: int):
        self.model_id = model_id

        import torch

        if device >= 0:
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

        self.out_model_dir = self.output_path("nemo_model", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # the model by default will be downloaded to huggingface cache
        # this can be overridden by setting the HF_HUB_CACHE environment variable:
        #   https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/file_download.py#L1171
        #   https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L123

        for env_var in ["HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "NEMO_CACHE_DIR"]:
            os.environ[env_var] = self.out_model_dir.get_path()

        from nemo.collections.asr.models import ASRModel

        ASRModel.from_pretrained(self.model_id, map_location=self.device)
