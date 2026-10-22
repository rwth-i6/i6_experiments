"""
Debug config entrypoint for WMT14 German-English Unsupervised NMT (UNdreaMT in RETURNN).
Reuses the pre-computed WMT14 News Crawl Word2Vec embeddings and numeral-initialized cross-lingual alignment,
and runs UNdreaMT training on a small subset of monolingual sentences for 3 epochs for fast interactive debugging.
"""

from i6_experiments.users.oyarzun.experiments.exp2026_10_19_unmt.sis_recipe.unmt_wmt14_de_en_exp import (
    run_wmt14_de_en_unmt_pipeline,
)


def py():
    run_wmt14_de_en_unmt_pipeline(
        prefix_name="exp2026_10_19_unmt/wmt14_de_en_debug",
        news_crawl_years=(2013,),
        max_mono_sentences=2000000,
        unmt_train_sentences=64,
        dev_max_sentences=32,
        vocab_size=50000,
        num_epochs=3,
        partition_epoch=1,
        batch_size=500,
        batch_max_seqs=16,
        device="cpu",
    )

