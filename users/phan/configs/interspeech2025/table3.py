"""
Sisyphus config for table 3
"""

from __future__ import annotations

from typing import Optional
from sisyphus import tk



def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    # get the recognition results with only ELM
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_baseline import sis_run_with_prefix as ilm_baseline_exps
    ilm_baseline_exps()

    # get the recognition results with transcription LM (all contexts: 1, 6, 10, full) as ILM
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_transcription import sis_run_with_prefix as ilm_transcription_exps
    ilm_transcription_exps()

    # get the recognition results with smoothing for full-context
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_kldiv_and_smoothing import sis_run_with_prefix as ilm_kldiv_and_smoothing_exps
    ilm_kldiv_and_smoothing_exps()

    # get the recognition results with smoothing for limited context (1, 6, 10)
    from i6_experiments.users.phan.configs.interspeech2025.backup.conformer_ilm_smoothing_ffnn import sis_run_with_prefix as ilm_smoothing_ffnn_exps
    ilm_smoothing_ffnn_exps()

py = sis_run_with_prefix  # if run directly via `sis m ...`

