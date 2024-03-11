from ..data.common import get_cv_bliss
from ..storage import asr_recognizer_systems
from ..pipeline import run_swer_evaluation


def run_evaluate_reference_swer():
    eval_systems = [
        "ls960eow_phon_ctc_50eps_fastsearch"
    ]
    for system_name in eval_systems:
        bliss = get_cv_bliss()
        run_swer_evaluation(
            prefix_name="experiments/jaist_project/evaluation/swer/" + system_name,
            synthetic_bliss=bliss,
            system=asr_recognizer_systems[system_name],
            with_confidence=True,
        )
