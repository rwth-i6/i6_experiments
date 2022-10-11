"""
Scoring function
"""

from sisyphus import tk
from i6_core.returnn.search import SearchWordsToCTMJob
from i6_core.recognition.scoring import Hub5ScoreJob
from . import get_bliss_xml_corpus
from ..task import DatasetConfig
from ..score_results import RecogOutput, ScoreResult
from ... import tools_paths

glm = tk.Path(
    "/work/asr2/oberdorfer/kaldi-stable/egs/swbd/s5/data/eval2000/glm",
    hash_overwrite="switchboard-kaldi-eval2000-glm",
    cached=True)

stms = {
    "hub5e_00": tk.Path(
        "/u/tuske/bin/switchboard/hub5e_00.2.stm",
        hash_overwrite="switchboard-tuske-hub5e_00.2.stm",
        cached=True),
    "hub5e_01": tk.Path(
        "/u/tuske/bin/switchboard/hub5e_01.2.stm",
        hash_overwrite="switchboard-tuske-hub5e_01.2.stm",
        cached=True),
    "rt03s": tk.Path(
        "/u/tuske/bin/switchboard/rt03s_ctsonly.stm",
        hash_overwrite="switchboard-tuske-rt03s_ctsonly.stm",
        cached=True),
}


def _score(*, hyp_words: tk.Path, corpus_name: str) -> ScoreResult:
    if corpus_name == "dev":  # name of corpus file
        corpus_name = "hub5e_00"
    assert corpus_name in stms
    ctm = SearchWordsToCTMJob(
        recog_words_file=hyp_words,
        bliss_corpus=get_bliss_xml_corpus(corpus_name))
    score_job = Hub5ScoreJob(
        glm=glm, hyp=ctm.out_ctm_file, ref=stms[corpus_name],
        sctk_binary_path=tools_paths.get_sctk_binary_path())
    return ScoreResult(
        dataset_name=corpus_name,
        main_measure_value=score_job.out_wer,
        report=score_job.out_report_dir)


def score(dataset: DatasetConfig, recog_output: RecogOutput) -> ScoreResult:
    """score"""
    return _score(hyp_words=recog_output.output, corpus_name=dataset.get_main_name())
