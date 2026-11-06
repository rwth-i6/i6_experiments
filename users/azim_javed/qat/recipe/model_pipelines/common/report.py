__all__ = ["register_memristor_report", "register_recog_report"]

import os
from typing import List, Optional

from i6_core.summary.wer import TableReport
from sisyphus import gs, tk

from .recog import MemristorRecogResult, RecogResult


def register_recog_report(recog_results: List[RecogResult], filename: str = "report.txt", live: bool = False):
    if live:
        # Capture the current dependencies, but render from the list again when
        # late (e.g. lazy-sweep) results are appended to it.
        return tk.register_report(
            os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, filename),
            lambda: _create_recog_report(recog_results)(),
            required=list(recog_results),
        )
    report_results = list(recog_results)
    return tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, filename),
        lambda: _create_recog_report(report_results)(),
        required=report_results,
    )


def register_memristor_report(
    memristor_results: List[MemristorRecogResult], filename: str = "memristor_report.txt", live: bool = False
):
    if live:
        return tk.register_report(
            os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, filename),
            lambda: _create_memristor_report(memristor_results)(),
            required=list(memristor_results),
        )
    return tk.register_report(
        os.path.join(gs.ALIAS_AND_OUTPUT_SUBDIR, filename), _create_memristor_report(memristor_results), required=True
    )


def _recog_result_to_dict(recog_result: RecogResult) -> dict[str, float]:
    def format_stats(stats: Optional[tk.Variable]) -> Optional[str]:
        if stats is None:
            return None
        if not stats.is_set():
            return "Not set"
        return "/".join(f"{val:.6f}" for val in stats.get().values())  # type: ignore

    wer = recog_result.wer
    num_errors = getattr(recog_result, "num_errors", None)
    ref_words = getattr(recog_result, "ref_words", None)
    if (
        num_errors is not None
        and ref_words is not None
        and num_errors.is_set()
        and ref_words.is_set()
        and ref_words.get() > 0
    ):
        wer = 100 * float(num_errors.get()) / float(ref_words.get())

    result = {}
    for name, value in [
        ("corpus", recog_result.corpus_name),
        ("WER", wer),
        ("Del", recog_result.deletion),
        ("Ins", recog_result.insertion),
        ("Sub", recog_result.substitution),
        ("Search Err", recog_result.search_error_rate),
        ("Model Err", recog_result.model_error_rate),
        ("Enc RTF", recog_result.enc_rtf),
        ("Search RTF", recog_result.search_rtf),
        ("Total RTF", recog_result.total_rtf),
        ("Unstable Latency", format_stats(recog_result.unstable_latency_stats)),
        ("Stable Latency", format_stats(recog_result.stable_latency_stats)),
        ("Step Hyps", format_stats(recog_result.step_hyps_stats)),
        ("Step Word End Hyps", format_stats(recog_result.step_word_end_hyps_stats)),
        ("Step Trees", format_stats(recog_result.step_trees_stats)),
    ]:
        if value is not None:
            result[name] = value

    return result


def _create_recog_report(recog_results: List[RecogResult]) -> TableReport:
    report = TableReport("Experiments", precision=6)

    recog_result_dicts = [
        (recog_result.descriptor, _recog_result_to_dict(recog_result)) for recog_result in recog_results
    ]

    column_indices = {}
    for _, recog_result in recog_result_dicts:
        for name in recog_result:
            if name not in column_indices:
                column_indices[name] = len(column_indices) + 2

    for row_idx, (descriptor, recog_result) in enumerate(recog_result_dicts, start=1):
        for name, val in recog_result.items():
            report.add_entry(col=f"{column_indices[name]} {name}", row=f"{row_idx}__{descriptor}", var=val)

    return report


def _memristor_result_to_dict(memristor_result: MemristorRecogResult) -> dict[str, Optional[tk.Variable]]:
    return {
        name: value
        for name, value in [
            ("corpus", memristor_result.corpus_name),
            ("Average WER", memristor_result.average_wer),
            ("Stddev WER", memristor_result.stddev_wer),
            ("Min WER", memristor_result.min_wer),
            ("Max WER", memristor_result.max_wer),
            ("Average Total RTF", memristor_result.average_total_rtf),
            ("Average Enc RTF", memristor_result.average_enc_rtf),
            ("Average Search RTF", memristor_result.average_search_rtf),
        ]
        if value is not None
    }


def _create_memristor_report(memristor_results: List[MemristorRecogResult]) -> TableReport:
    report = TableReport("Experiments", precision=5)

    memristor_result_dicts = [
        (memristor_result.descriptor, _memristor_result_to_dict(memristor_result))
        for memristor_result in memristor_results
    ]

    column_indices = {}
    for _, memristor_result in memristor_result_dicts:
        for name in memristor_result:
            if name not in column_indices:
                column_indices[name] = len(column_indices) + 2

    for row_idx, (descriptor, memristor_result) in enumerate(memristor_result_dicts, start=1):
        for name, val in memristor_result.items():
            report.add_entry(col=f"{column_indices[name]} {name}", row=f"{row_idx}__{descriptor}", var=val)

    return report
