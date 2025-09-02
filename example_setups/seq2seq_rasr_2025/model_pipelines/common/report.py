from typing import List

from i6_core.summary.wer import TableReport

from .recog import RecogResult


def create_base_recog_report(recog_results: List[RecogResult]) -> TableReport:
    report = TableReport("Experiments", precision=5)

    for idx, recog_result in enumerate(recog_results, start=1):
        report.add_entry(col="1 corpus", row=f"{idx}_{recog_result.descriptor}", var=recog_result.corpus_name)
        report.add_entry(col="2 WER", row=f"{idx}_{recog_result.descriptor}", var=recog_result.wer)
        report.add_entry(col="3 Del", row=f"{idx}_{recog_result.descriptor}", var=recog_result.deletion)
        report.add_entry(col="4 Ins", row=f"{idx}_{recog_result.descriptor}", var=recog_result.insertion)
        report.add_entry(col="5 Sub", row=f"{idx}_{recog_result.descriptor}", var=recog_result.substitution)

    return report


def create_offline_recog_report(recog_results: List[RecogResult]) -> TableReport:
    report = TableReport("Experiments", precision=5)

    for idx, recog_result in enumerate(recog_results, start=1):
        report.add_entry(col="1 corpus", row=f"{idx}_{recog_result.descriptor}", var=recog_result.corpus_name)
        report.add_entry(col="2 WER", row=f"{idx}_{recog_result.descriptor}", var=recog_result.wer)
        report.add_entry(col="3 Del", row=f"{idx}_{recog_result.descriptor}", var=recog_result.deletion)
        report.add_entry(col="4 Ins", row=f"{idx}_{recog_result.descriptor}", var=recog_result.insertion)
        report.add_entry(col="5 Sub", row=f"{idx}_{recog_result.descriptor}", var=recog_result.substitution)
        report.add_entry(
            col="6 AM RTF", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "enc_rtf", 0)
        )
        report.add_entry(
            col="7 Search RTF", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "search_rtf", 0)
        )
        report.add_entry(
            col="8 Overall RTF", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "total_rtf", 0)
        )

    return report


def create_offline_recog_report_with_search_errors(
    recog_results: List[RecogResult], title: str = "Experiments"
) -> TableReport:
    report = TableReport(title, precision=5)

    for idx, recog_result in enumerate(recog_results, start=1):
        report.add_entry(col="1 corpus", row=f"{idx}_{recog_result.descriptor}", var=recog_result.corpus_name)
        report.add_entry(col="2 WER", row=f"{idx}_{recog_result.descriptor}", var=recog_result.wer)
        report.add_entry(col="3 Del", row=f"{idx}_{recog_result.descriptor}", var=recog_result.deletion)
        report.add_entry(col="4 Ins", row=f"{idx}_{recog_result.descriptor}", var=recog_result.insertion)
        report.add_entry(col="5 Sub", row=f"{idx}_{recog_result.descriptor}", var=recog_result.substitution)
        report.add_entry(
            col="6 Search Err",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "search_error_rate", 0),
        )
        report.add_entry(
            col="7 Model Err", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "model_error_rate", 0)
        )
        report.add_entry(
            col="8 Skipped", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "skipped_rate", 0)
        )
        report.add_entry(
            col="9 Correct", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "correct_rate", 0)
        )
        report.add_entry(
            col="10 AM RTF", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "enc_rtf", 0)
        )
        report.add_entry(
            col="11 Search RTF", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "search_rtf", 0)
        )
        report.add_entry(
            col="12 Overall RTF", row=f"{idx}_{recog_result.descriptor}", var=getattr(recog_result, "total_rtf", 0)
        )

    return report


def create_streaming_recog_report(recog_results: List[RecogResult]) -> TableReport:
    report = TableReport("Experiments", precision=5)

    for idx, recog_result in enumerate(recog_results, start=1):
        report.add_entry(col="1 corpus", row=f"{idx}_{recog_result.descriptor}", var=recog_result.corpus_name)
        report.add_entry(col="2 WER", row=f"{idx}_{recog_result.descriptor}", var=recog_result.wer)
        report.add_entry(col="3 Del", row=f"{idx}_{recog_result.descriptor}", var=recog_result.deletion)
        report.add_entry(col="4 Ins", row=f"{idx}_{recog_result.descriptor}", var=recog_result.insertion)
        report.add_entry(col="5 Sub", row=f"{idx}_{recog_result.descriptor}", var=recog_result.substitution)
        report.add_entry(
            col="6 Unstable Latency avg",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "unstable_latency_avg", 0),
        )
        report.add_entry(
            col="7 Unstable Latency p50",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "unstable_latency_p50", 0),
        )
        report.add_entry(
            col="8 Unstable Latency p90",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "unstable_latency_p90", 0),
        )
        report.add_entry(
            col="9 Unstable Latency p99",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "unstable_latency_p99", 0),
        )
        report.add_entry(
            col="10 Unstable Latency p100",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "unstable_latency_p100", 0),
        )
        report.add_entry(
            col="11 Stable Latency avg",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "stable_latency_avg", 0),
        )
        report.add_entry(
            col="12 Stable Latency p50",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "stable_latency_p50", 0),
        )
        report.add_entry(
            col="13 Stable Latency p90",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "stable_latency_p90", 0),
        )
        report.add_entry(
            col="14 Stable Latency p99",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "stable_latency_p99", 0),
        )
        report.add_entry(
            col="15 Stable Latency p100",
            row=f"{idx}_{recog_result.descriptor}",
            var=getattr(recog_result, "stable_latency_p100", 0),
        )

    return report
