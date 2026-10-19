"""
Configuration for testing the self-supervised SpeechT5 Base model directly on LibriSpeech
without fine-tuning.

Tasks:
1. Direct zero-shot ASR evaluation on dev-clean and dev-other.
2. Cross-attention weight extraction and heatmap visualization.
3. Audio vs. Text LM shared-encoder state PCA analysis and cosine similarity metrics.
"""

import os
from sisyphus import tk

from ....speecht5_eval import SpeechT5BaseEvalJob

# Default paths for cluster execution
DEFAULT_SPEECHT5_BASE_DIR = "/rwthfs/rz/cluster/hpcwork/bpd03090/speecht5_base_converted"

# LibriSpeech downloaded directories from existing Sisyphus work dirs
DEFAULT_DEV_CLEAN_DIR = (
    "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr"
    "/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.CeoY6kKnh8B3/output/dev-clean"
)
DEFAULT_DEV_OTHER_DIR = (
    "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr"
    "/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.gWxOHrcGHjNZ/output/dev-other"
)
DEFAULT_LM_TEXT_PATH = (
    "/rwthfs/rz/cluster/hpcwork/bpd03090/sisyphus-work-dirs/exp2026_04_09_unsupervised_asr"
    "/i6_experiments/users/schmitt/text/normalize/NormalizeLBSLMDataJob.Y7vdO0nAu03n/output/corpus.norm.txt"
)


# Dummy version parameter to allow rescheduling jobs in Sisyphus
DUMMY_VERSION = 2


def py(dummy_version: int = DUMMY_VERSION):
    """
    Registers the SpeechT5 self-supervised evaluation jobs in Sisyphus.
    """
    eval_jobs = {}

    splits = [
        ("dev_clean", DEFAULT_DEV_CLEAN_DIR),
        ("dev_other", DEFAULT_DEV_OTHER_DIR),
    ]

    for split_name, split_dir in splits:
        job = SpeechT5BaseEvalJob(
            model_dir=DEFAULT_SPEECHT5_BASE_DIR,
            dataset_dir=split_dir,
            text_lm_path=DEFAULT_LM_TEXT_PATH,
            max_seqs=None,  # evaluate full split in Sisyphus
            max_cross_att_seqs=20,
            max_pca_seqs=100,
            name_prefix=f"speecht5_base_{split_name}",
            dummy_version=dummy_version,
        )
        tk.register_output(f"speecht5_base_eval/{split_name}/hypotheses.txt", job.out_hypotheses_txt)
        tk.register_output(f"speecht5_base_eval/{split_name}/asr_summary.json", job.out_asr_summary_json)
        tk.register_output(f"speecht5_base_eval/{split_name}/asr_report.txt", job.out_asr_report_txt)
        tk.register_output(f"speecht5_base_eval/{split_name}/cross_attention", job.out_cross_att_dir)
        tk.register_output(f"speecht5_base_eval/{split_name}/pca_analysis", job.out_pca_dir)

        eval_jobs[split_name] = job

    return eval_jobs


def main():
    return py()
