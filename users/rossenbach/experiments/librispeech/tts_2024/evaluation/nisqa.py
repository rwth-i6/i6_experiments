from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from ..data.tts_phon import get_librispeech_tts_segments, get_bliss_corpus_dict
from ..pipeline import evaluate_nisqa


def run_evaluate_reference_nisqa():
    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key="train-clean-100")
    train_clean_100_bliss = get_bliss_corpus_dict("ogg")["train-clean-100"]
    cv_corpus = FilterCorpusBySegmentsJob(
        bliss_corpus=train_clean_100_bliss,
        segment_file=cv_segments,
        delete_empty_recordings=True
    ).out_corpus
    evaluate_nisqa(prefix_name="experiments/jaist_project/evaluation/cv_mos", bliss_corpus=cv_corpus, with_bootstrap=True)
