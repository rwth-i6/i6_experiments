from ..pipeline import evaluate_nisqa
from ..data import get_librispeech_tts_segments, get_bliss_corpus_dict, FilterCorpusBySegmentsJob

prefix = "experiments/librispeech/eval_references"

def get_nisqa_reference():
    train_segments, cv_segments = get_librispeech_tts_segments(ls_corpus_key="train-clean-100")
    train_clean_100_bliss = get_bliss_corpus_dict("ogg")["train-clean-100"]
    cv_corpus = FilterCorpusBySegmentsJob(
        bliss_corpus=train_clean_100_bliss,
        segment_file=cv_segments,
        delete_empty_recordings=True
    ).out_corpus
    evaluate_nisqa(prefix_name=prefix + "/nisqa/", bliss_corpus=cv_corpus, vocoder="original")
