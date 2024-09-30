from ..data.common import get_cv_bliss
from ..storage import asr_recognizer_systems
from ..pipeline import run_swer_evaluation


def run_evaluate_reference_swer(prefix=None, bliss=None):
    """
    if None evaluates ground truth

    :param prefix:
    :param bliss:
    :return:
    """
    if prefix is None:
        prefix = "experiments/jaist_project/evaluation"
    eval_systems = [
        "ls960eow_phon_ctc_50eps_fastsearch"
    ]
    for system_name in eval_systems:
        if bliss is None:
            bliss = get_cv_bliss()
        run_swer_evaluation(
            prefix_name=prefix + "/swer/" + system_name,
            synthetic_bliss=bliss,
            system=asr_recognizer_systems[system_name],
            with_confidence=True,
        )


