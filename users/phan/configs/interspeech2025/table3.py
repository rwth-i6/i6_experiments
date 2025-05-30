"""
Sisyphus config for table 3
"""

from __future__ import annotations

from typing import Optional
from sisyphus import tk



def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    # get the recognition results with transcription LM as ILM
    from i6_experiments.users.phan.configs.conformer_ilm_transcription import sis_run_with_prefix as ilm_transcription_exps
    ilm_transcription_exps()

    # get the recognition results with smoothing for full-context
    from i6_experiments.users.phan.configs.conformer_ilm_kldiv_v4_fixEos_noSpecAug import sis_run_with_prefix as ilm_kldiv_and_smoothing_exps
    ilm_kldiv_and_smoothing_exps()

    # get the recognition results with smoothing for limited context
    # context 1
    from i6_experiments.users.phan.configs.ilm_archs.ffnn_context1_layers2_hiddendim1000.conformer_ilm import sis_run_with_prefix as ilm_smoothing_context_1_exps
    ilm_smoothing_context_1_exps()
    # context 6
    from i6_experiments.users.phan.configs.ilm_archs.ffnn_context6_layers2_hiddendim1000.conformer_ilm import sis_run_with_prefix as ilm_smoothing_context_6_exps
    ilm_smoothing_context_6_exps()
    # context 10
    from i6_experiments.users.phan.configs.ilm_archs.ffnn_context10_layers2_hiddendim1000.conformer_ilm import sis_run_with_prefix as ilm_smoothing_context_10_exps
    ilm_smoothing_context_10_exps()

py = sis_run_with_prefix  # if run directly via `sis m ...`

