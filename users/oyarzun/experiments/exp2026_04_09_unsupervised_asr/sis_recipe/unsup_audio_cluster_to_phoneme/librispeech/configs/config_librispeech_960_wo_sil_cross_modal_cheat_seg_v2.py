"""
Config: LibriSpeech 960 (without silence) - v2 Monophone Cross-Modal Cheating Segmentation Evaluation

Evaluates v2 monophone cross-modal codes:
- clus_1P_40
- clus_2P_80
- clus_4P_160
- clus_200_200
Using oracle cheating segmentation and nearest-neighbor search against text phoneme embeddings.
"""

from typing import Optional
from .config_librispeech_960_wo_sil_cross_modal_cheat_seg import eval_v2


def py(cluster: Optional[str] = None):
    """
    Runs cheating segmentation nearest neighbor evaluation for v2 monophone codes.
    :param cluster: 'curium' for local cluster or 'rwth-hpc' / 'rwth-hcp' for RWTH cluster.
    """
    eval_v2(cluster=cluster)
