"""
Config: LibriSpeech 960 (without silence) - v3 Diphone & Triphone Cross-Modal Cheating Segmentation Evaluation

Evaluates v3 cross-modal codes:
- Diphone and Triphone codes
- Cluster sizes K in {200, 250, 500}
- Context windows in {3, 5, 10}
Using oracle cheating segmentation and nearest-neighbor search against text n-gram embeddings.
"""

from typing import Optional
from .config_librispeech_960_wo_sil_cross_modal_cheat_seg import eval_v3


def py(cluster: Optional[str] = None):
    """
    Runs cheating segmentation nearest neighbor evaluation for v3 diphone and triphone codes.
    :param cluster: 'curium' for local cluster or 'rwth-hpc' / 'rwth-hcp' for RWTH cluster.
    """
    eval_v3(cluster=cluster)
