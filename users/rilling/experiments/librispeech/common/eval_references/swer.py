from i6_experiments.users.rossenbach.experiments.jaist_project.storage import asr_recognizer_systems

from ..data import get_bliss_corpus_dict, build_swer_test_dataset, get_tts_lexicon
from ..tts_eval import search_single
from ..default_tools import MINI_RETURNN_ROOT, RETURNN_PYTORCH_ASR_SEARCH_EXE

prefix = "experiments/librispeech/eval_references"

def get_swer_evaluation_reference():
    asr_system = "ls960eow_phon_ctc_50eps_fastsearch"

    from i6_core.corpus.filter import FilterCorpusRemoveUnknownWordSegmentsJob

    bliss = FilterCorpusRemoveUnknownWordSegmentsJob(
        bliss_corpus=get_bliss_corpus_dict()["test-clean"], bliss_lexicon=get_tts_lexicon(), all_unknown=False
    ).out_corpus
    system = asr_recognizer_systems[asr_system]

    search_single(
        prefix_name=prefix + "/swer/" + asr_system,
        returnn_config=system.config,
        checkpoint=system.checkpoint,
        recognition_dataset=build_swer_test_dataset(
            synthetic_bliss=bliss,
            returnn_exe=RETURNN_PYTORCH_ASR_SEARCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            preemphasis=system.preemphasis,
            peak_normalization=system.peak_normalization,
        ),
        recognition_bliss_corpus=bliss,
        returnn_exe=RETURNN_PYTORCH_ASR_SEARCH_EXE,
        returnn_root=MINI_RETURNN_ROOT,
        # with_confidence=with_confidence,
    )
