"""
Generic score function for speech recognition using sclite.
"""

from __future__ import annotations
from typing import Sequence, Callable

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.schmitt import tools_paths
from ..task import RecogOutput, ScoreResult


def generic_sclite_score_recog_out(
    dataset: DatasetConfig,
    recog_output: RecogOutput,
    *,
    post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]],
) -> ScoreResult:
    """
    score

    To use it for :class:`Task.score_recog_output_func`,
    you need to use functools.partial and set ``post_proc_funcs``.
    """
    from .serialize import ReturnnDatasetToTextDictJob

    corpus_name = dataset.get_main_name()

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )
    for f in post_proc_funcs:
        ref = f(ref)

    return sclite_score_recog_out_to_ref(recog_output, ref=ref, corpus_name=corpus_name)


def sclite_score_recog_out_to_ref(recog_output: RecogOutput, *, ref: RecogOutput, corpus_name: str) -> ScoreResult:
    """
    score

    To use it for :class:`Task.score_recog_output_func`,
    you need to use functools.partial and set ``post_proc_funcs``.
    """
    # sclite here. Could also use ReturnnComputeWERJob.
    from i6_core.returnn.search import SearchWordsDummyTimesToCTMJob
    from i6_core.text.convert import TextDictToStmJob
    from i6_core.recognition.scoring import ScliteJob

    hyp_words = recog_output.output
    corpus_text_dict = ref.output

    # Arbitrary seg length time. The jobs SearchWordsDummyTimesToCTMJob and TextDictToStmJob
    # serialize two points after decimal, so long seqs (>1h or so) might be problematic,
    # and no reason not to just use a high value here to avoid this problem whenever we get to it.
    seg_length_time = 1000.0
    search_ctm = SearchWordsDummyTimesToCTMJob(
        recog_words_file=hyp_words, seq_order_file=corpus_text_dict, seg_length_time=seg_length_time
    ).out_ctm_file
    stm_file = TextDictToStmJob(text_dict=corpus_text_dict, seg_length_time=seg_length_time).out_stm_path

    score_job = ScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )

    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)
