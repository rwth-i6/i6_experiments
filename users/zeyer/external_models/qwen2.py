"""
Qwen2
"""

import functools
from sisyphus import Path
from i6_experiments.common.utils.fake_job import make_fake_job


def get_lm():
    # /hpcwork/p0023999/hq237549/sisyphus-work-dirs/2026-01-20--llm/work/i6_core/returnn/training/ReturnnTrainingJob.MIU24HbRi60L/output/returnn.config

    get_model = functools.partial(
        _qwen2_get_model,
        **{
            "hf_hub_cache_dir": "/rwthfs/rz/cluster/home/hq237549/experiments/2026-01-20--llm/work/i6_experiments/users/schmitt/external_models/huggingface/DownloadHuggingFaceRepoJob.r7AjtV7muFpk/output/hub_cache",
            "freeze_params": False,
            "lora_opts": None,
            "freeze_embedding_layer": True,
        },
    )

    from speech_llm.prefix_lm.model.custom_missing_load_funcs.qwen import qwen_load_tied_embedding_matrices

    preload_from_files = {
        "qwen": {
            # /rwthfs/rz/cluster/home/hq237549/experiments/2026-01-20--llm/work/i6_core/tools/download/DownloadJob.6SV1LOlUtQMG/output/qwen2-0_5b_model.safetensors
            "filename": Path(
                "qwen2-0_5b_model.safetensors",
                creator=make_fake_job(module="i6_core.tools.download", name="DownloadJob", sis_hash="6SV1LOlUtQMG"),
            ),
            "init_for_train": True,
            "checkpoint_key": None,
            "prefix": "model.",
            "custom_missing_load_func": qwen_load_tied_embedding_matrices,
            "ignore_missing": True,
        }
    }


def _qwen2_get_model(**kwargs):
    """
    Indirection to avoid importing Torch in the Sisyphus manager.
    """
    from speech_llm.prefix_lm.model.definitions.decoders.qwen import Qwen2DecoderV3

    return Qwen2DecoderV3(**kwargs)
