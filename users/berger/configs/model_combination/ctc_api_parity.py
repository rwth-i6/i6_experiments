"""Run only the public Granite/Speech-Qwen request-API parity check."""


def py():
    import os
    from .experiments.open_asr.ctc_api_parity import run

    return run(
        tuple(os.environ.get("ASR_CTC_PARITY_DATASETS", "librispeech-dev-other").split(",")),
        transformers_version=os.environ.get("ASR_CTC_PARITY_TRANSFORMERS", "5.16.1"),
    )
