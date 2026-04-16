from i6_experiments.users.zeyer.external_models.huggingface import (
    DownloadHuggingFaceRepoJob,
)
from .common import HF_CACHE_DIR 

# None of this is used anywhere i think

def download_moshi():
    # projects/moshi/moshi/moshi/models/loaders.py
    # untested code...

    repo = DownloadHuggingFaceRepoJob(model_id="kyutai/moshiko-pytorch-bf16")
    repo.out_hub_cache_dir = HF_CACHE_DIR
    return repo.out_hub_cache_dir


def moshi_inference_server(model):
    pass
