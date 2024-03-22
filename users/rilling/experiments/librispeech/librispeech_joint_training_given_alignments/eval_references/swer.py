from i6_experiments.users.rossenbach.experiments.jaist_project.storage import asr_recognizer_systems

from ..data import get_cv_bliss, build_swer_test_dataset
from ..pipeline import search_single
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_PYTORCH_EXE

prefix = "experiments/librispeech/eval_references"

def get_swer_evaluation_reference():
    asr_system = "ls960eow_phon_ctc_50eps_fastsearch"

    bliss = get_cv_bliss()
    system = asr_recognizer_systems[asr_system]

    search_single(
        prefix_name=prefix + "/swer/" + asr_system,
        returnn_config=system.config,
        checkpoint=system.checkpoint,
        recognition_dataset=build_swer_test_dataset(
            synthetic_bliss=bliss,
            preemphasis=system.preemphasis,
            peak_normalization=system.peak_normalization,
        ),
        recognition_bliss_corpus=get_cv_bliss(),
        returnn_exe=RETURNN_PYTORCH_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        # with_confidence=with_confidence,
    )
