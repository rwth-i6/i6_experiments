from __future__ import annotations
from typing import Optional, Union, Any, Dict
import os
import sys
from sisyphus import tk, Job, Task, Path

from i6_experiments.users.phan.alignment.albert_alignment_setup import CalcAlignmentMetrics
from i6_experiments.users.phan.configs.bilstm_encoder_align import get_alignment

def py():
    prefix = "alignments_tse/"
    name = "bilstm-ctc-10k-align/metrics"  # 312.2/306.1ms
    gmm_alignment_allophones = Path(
        "/work/common/asr/librispeech/data/sisyphus_export_setup/work/i6_core/lexicon/allophones/StoreAllophonesJob.bY339UmRbGhr/output/allophones"
    )
    gmm_alignment_sprint_cache = Path(
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/mm/alignment/AlignmentJob.oyZ7O0XJcO20/output/alignment.cache.bundle"
    )
    features_sprint_cache = Path(  # for exact timings
        "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/features/extraction/FeatureExtractionJob.VTLN.upmU2hTb8dNH/output/vtln.cache.bundle"
    )
    seq_list = Path(
        "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/corpus/segments/SegmentCorpusJob.AmDlp1YMZF1e/output/segments.1"
    )
    job = CalcAlignmentMetrics(
        seq_list=None,
        alignment_hdf=get_alignment(dataset_key="train"),
        alignment_label_topology="ctc",
        alignment_bpe_vocab=Path(
            "/u/minh-nghia.phan/setups/rf_ctc/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab"
        ),
        alignment_blank_idx=10_025,
        features_sprint_cache=features_sprint_cache,
        ref_alignment_sprint_cache=gmm_alignment_sprint_cache,
        ref_alignment_allophones=gmm_alignment_allophones,
        ref_alignment_len_factor=6,
    )
    # job.run()
    job.add_alias(prefix + name)
    tk.register_output(prefix + name + ".json", job.out_scores)
