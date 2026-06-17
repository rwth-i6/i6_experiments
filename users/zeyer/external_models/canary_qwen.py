"""Download helpers for NVIDIA NeMo Canary-Qwen models.

Canary-Qwen's SALM wrapper instantiates ``Qwen/Qwen3-1.7B`` via HF AutoTokenizer
at load time, which needs the base LLM in the HF cache (compute nodes are
offline). We download the base LLM separately and let the adapter merge the
two caches at process startup.
"""

import functools
from sisyphus import tk

from .huggingface import DownloadHuggingFaceRepoJobV2


@functools.cache
def download_canary_qwen_2_5b_model() -> tk.Path:
    """Download ``nvidia/canary-qwen-2.5b`` via the Sis-managed HF hub
    cache. Hash-stable so multiple recipes share the same download."""
    dl = DownloadHuggingFaceRepoJobV2(repo_id="nvidia/canary-qwen-2.5b", repo_type="model")
    tk.register_output("canary-qwen-2.5b-model", dl.out_hub_cache_dir)
    return dl.out_hub_cache_dir


@functools.cache
def download_qwen3_1_7b_model() -> tk.Path:
    """Download ``Qwen/Qwen3-1.7B`` (Canary-Qwen's base LLM, used by SALM's
    AutoTokenizer at init time on offline compute nodes)."""
    dl = DownloadHuggingFaceRepoJobV2(repo_id="Qwen/Qwen3-1.7B", repo_type="model")
    tk.register_output("qwen3-1.7b-model", dl.out_hub_cache_dir)
    return dl.out_hub_cache_dir
