import os
from sisyphus import tk

from .data import get_tts_data_from_ctc_align
from .config import get_training_config

from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT, RETURNN_COMMON, RETURNN_DATA_ROOT


from ..ctc_aligner.experiments import get_baseline_ctc_alignment_v2

def get_ctc_based_tts():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """

    name = "experiments/alignment_analysis_tts/default_tts/ctc_based/baseline"

    alignment_hdf = get_baseline_ctc_alignment_v2()
    training_datasets = get_tts_data_from_ctc_align(alignment_hdf=alignment_hdf)

    training_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets
    )  # implicit reconstruction loss
