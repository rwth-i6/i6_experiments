"""
Generic score function for speech recognition using sclite.
"""

from __future__ import annotations
from typing import Sequence, Callable

from sisyphus import tk
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer import tools_paths
from ..task import RecogOutput, ScoreResult


def generic_sclite_score_recog_out(
    dataset: DatasetConfig,
    recog_output: RecogOutput,
    *,
    post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
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

    :param recog_output: in TextDict
    :param ref: in TextDict
    :param corpus_name: just for the resulting ScoreResult
    :return: score result
    """
    return sclite_score_hyps_to_ref(
        hyps_text_dict=recog_output.output, ref_text_dict=ref.output, corpus_name=corpus_name
    )


def sclite_score_hyps_to_ref(
    hyps_text_dict: tk.Path, *, ref_text_dict: tk.Path, corpus_name: str = "<undefined-corpus-name>"
) -> ScoreResult:
    """
    score

    :param hyps_text_dict: in TextDict
    :param ref_text_dict: in TextDict
    :param corpus_name: just for the resulting ScoreResult. you can also ignore this when it is not used
    :return: score result
    """
    # sclite here. Could also use ReturnnComputeWERJob.
    from i6_core.returnn.search import SearchWordsDummyTimesToCTMJob
    from i6_core.text.convert import TextDictToStmJob
    from i6_core.recognition.scoring import ScliteJob

    # Arbitrary seg length time. The jobs SearchWordsDummyTimesToCTMJob and TextDictToStmJob
    # serialize two points after decimal, so long seqs (>1h or so) might be problematic,
    # and no reason not to just use a high value here to avoid this problem whenever we get to it.
    seg_length_time = 1000.0
    search_ctm = SearchWordsDummyTimesToCTMJob(
        recog_words_file=hyps_text_dict, seq_order_file=ref_text_dict, seg_length_time=seg_length_time
    ).out_ctm_file
    stm_file = TextDictToStmJob(text_dict=ref_text_dict, seg_length_time=seg_length_time).out_stm_path

    score_job = ScliteJob(
        ref=stm_file, hyp=search_ctm, sctk_binary_path=tools_paths.get_sctk_binary_path(), precision_ndigit=2
    )

    return ScoreResult(dataset_name=corpus_name, main_measure_value=score_job.out_wer, report=score_job.out_report_dir)
