"""Qwen-ASR 0.6B with RASR on LibriSpeech dev-other."""


def py():
    import os
    from i6_experiments.example_setups.seq2seq_rasr_2025.data.librispeech.datasets import get_default_score_corpus
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.qwen_asr import run

    return run(get_default_score_corpus("dev-other"), cache_dir=os.environ.get("HF_HUB_CACHE"))


def main():
    return py()
