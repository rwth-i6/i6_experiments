"""
Continuation of :mod:`exp24_09_16_grad_align`.
"""

from sisyphus import tk
from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJob


def py():
    dl = DownloadHuggingFaceRepoJob(model_id="CohereLabs/aya-expanse-32b")
    tk.register_output("aya", dl.out_hub_cache_dir)
