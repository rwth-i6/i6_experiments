"""Standalone SpeechLLM example; no OOV prototypes, fusion or benchmark suite."""

from sisyphus import tk

from ..model_pipelines.common.corpus import ScorableCorpus
from ..model_pipelines.qwen_asr.recog import DownloadQwenAsrJob, QwenAsrVocabJob, QwenAsrRecogJob
from ..tools import rasr_binary_path


def run(corpus: ScorableCorpus, *, cache_dir=None, beam=4):
    download = DownloadQwenAsrJob(cache_dir=cache_dir)
    download.add_alias("qwen-asr/model")
    vocab = QwenAsrVocabJob(download.out_repo)
    vocab.add_alias("qwen-asr/vocabulary")
    job = QwenAsrRecogJob(
        corpus.bliss_corpus_file,
        download.out_repo,
        vocab.out_vocab,
        vocab.out_info,
        rasr_binary_path,
        beam=beam,
        uppercase=True,
        strip_punctuation=True,
    )
    prefix = f"qwen-asr/{corpus.corpus_name}"
    job.add_alias(prefix)
    score = corpus.score_ctm(job.out_ctm)
    tk.register_output(prefix + "/hypotheses", job.out_hypotheses)
    tk.register_output(prefix + "/ctm", job.out_ctm)
    tk.register_output(prefix + "/scoring", score.out_report_dir)
    return job, score
