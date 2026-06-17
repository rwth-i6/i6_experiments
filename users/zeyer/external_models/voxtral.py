"""Download helper for Mistral Voxtral models."""

import functools
from sisyphus import tk

from .huggingface import DownloadHuggingFaceRepoJobV2


@functools.cache
def download_voxtral_mini_3b_model() -> tk.Path:
    """Download ``mistralai/Voxtral-Mini-3B-2507`` via the Sis-managed HF hub
    cache. Hash-stable so multiple recipes can share the same download."""
    dl = DownloadHuggingFaceRepoJobV2(repo_id="mistralai/Voxtral-Mini-3B-2507", repo_type="model")
    tk.register_output("voxtral-mini-3b-model", dl.out_hub_cache_dir)
    return dl.out_hub_cache_dir
