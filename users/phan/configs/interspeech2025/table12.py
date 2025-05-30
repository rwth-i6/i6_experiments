"""
Sisyphus config for table 1 and 2
"""

from __future__ import annotations

from typing import Optional
from sisyphus import tk



def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    # get the recognition results with no ELM and with only ELM
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_baseline import sis_run_with_prefix as ilm_baseline_exps
    ilm_baseline_exps()

    # get the recognition results with unigram ILM (renormalized prior)
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_unigram import sis_run_with_prefix as ilm_unigram_renorm_prior_exps
    ilm_unigram_renorm_prior_exps()

    # get the recognition results with transcription LM as ILM
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_transcription import sis_run_with_prefix as ilm_transcription_exps
    ilm_transcription_exps()

    # get the recognition results with KLDiv and smoothing
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_kldiv_and_smoothing import sis_run_with_prefix as ilm_kldiv_and_smoothing_exps
    ilm_kldiv_and_smoothing_exps()

    # get the recognition results with masking
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_kldiv_masking import sis_run_with_prefix as ilm_masking_exps
    ilm_masking_exps()

    # get the recognition results with sequence-level KD
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_kldiv_sequence_level import sis_run_with_prefix as ilm_seq_level_exps
    ilm_seq_level_exps()

py = sis_run_with_prefix  # if run directly via `sis m ...`

