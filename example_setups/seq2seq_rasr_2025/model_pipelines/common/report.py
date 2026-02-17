__all__ = ["register_report", "create_recog_report", "register_recog_report"]

from typing import List, Optional

from i6_core.summary.wer import TableReport
from sisyphus import gs, tk

from .recog import RecogResult


def register_report(report: TableReport, filename: str) -> None:
    tk.register_report(f"{gs.ALIAS_AND_OUTPUT_SUBDIR}/{filename}", report, required=True)


def _recog_result_to_dict(recog_result: RecogResult) -> dict[str, float]:
    def format_stats(stats: Optional[tk.Variable]) -> Optional[str]:
        if stats is None:
            return None
        if not stats.is_set():
            return "Not set"
        return "/".join(f"{val:.2f}" for val in stats.get().values())  # type: ignore

    result = {}
    for name, value in [
        ("corpus", recog_result.corpus_name),
        ("WER", recog_result.wer),
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


def create_recog_report(recog_results: List[RecogResult]) -> TableReport:
    report = TableReport("Experiments", precision=5)

    recog_result_dicts = [
        (recog_result.descriptor, _recog_result_to_dict(recog_result)) for recog_result in recog_results
    ]

    column_indices = {}
    for _, recog_result in recog_result_dicts:
        for name in recog_result:
            if name not in column_indices:
                column_indices[name] = len(column_indices) + 1

    for row_idx, (descriptor, recog_result) in enumerate(recog_result_dicts, start=1):
        for name, val in recog_result.items():
            report.add_entry(col=f"{column_indices[name]} {name}", row=f"{row_idx}_{descriptor}", var=val)

    return report


def register_recog_report(recog_results: List[RecogResult], filename: str = "report.txt") -> None:
    register_report(create_recog_report(recog_results), filename=filename)
