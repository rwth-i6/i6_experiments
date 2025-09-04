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
        report.add_entry(col="6 Hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_avg)
        report.add_entry(col="7 Hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p50)
        report.add_entry(col="8 Hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p90)
        report.add_entry(col="9 Hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p99)
        report.add_entry(col="10 Hyps_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p100)
        report.add_entry(
            col="11 Word_end_hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_avg
        )
        report.add_entry(
            col="12 Word_end_hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p50
        )
        report.add_entry(
            col="13 Word_end_hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p90
        )
        report.add_entry(
            col="14 Word_end_hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p99
        )
        report.add_entry(
            col="15 Word_end_hyps_p100",
            row=f"{idx}_{recog_result.descriptor}",
            var=recog_result.step_word_end_hyps_p100,
        )
        report.add_entry(col="16 Trees_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_avg)
        report.add_entry(col="17 Trees_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p50)
        report.add_entry(col="18 Trees_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p90)
        report.add_entry(col="19 Trees_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p99)
        report.add_entry(col="20 Trees_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p100)

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
        report.add_entry(col="9 Hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_avg)
        report.add_entry(col="10 Hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p50)
        report.add_entry(col="11 Hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p90)
        report.add_entry(col="12 Hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p99)
        report.add_entry(col="13 Hyps_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p100)
        report.add_entry(
            col="14 Word_end_hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_avg
        )
        report.add_entry(
            col="15 Word_end_hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p50
        )
        report.add_entry(
            col="16 Word_end_hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p90
        )
        report.add_entry(
            col="17 Word_end_hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p99
        )
        report.add_entry(
            col="18 Word_end_hyps_p100",
            row=f"{idx}_{recog_result.descriptor}",
            var=recog_result.step_word_end_hyps_p100,
        )
        report.add_entry(col="19 Trees_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_avg)
        report.add_entry(col="20 Trees_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p50)
        report.add_entry(col="21 Trees_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p90)
        report.add_entry(col="22 Trees_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p99)
        report.add_entry(col="23 Trees_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p100)

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
        report.add_entry(col="13 Hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_avg)
        report.add_entry(col="14 Hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p50)
        report.add_entry(col="15 Hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p90)
        report.add_entry(col="16 Hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p99)
        report.add_entry(col="17 Hyps_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p100)
        report.add_entry(
            col="18 Word_end_hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_avg
        )
        report.add_entry(
            col="19 Word_end_hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p50
        )
        report.add_entry(
            col="20 Word_end_hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p90
        )
        report.add_entry(
            col="21 Word_end_hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p99
        )
        report.add_entry(
            col="22 Word_end_hyps_p100",
            row=f"{idx}_{recog_result.descriptor}",
            var=recog_result.step_word_end_hyps_p100,
        )
        report.add_entry(col="23 Trees_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_avg)
        report.add_entry(col="24 Trees_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p50)
        report.add_entry(col="25 Trees_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p90)
        report.add_entry(col="26 Trees_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p99)
        report.add_entry(col="27 Trees_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p100)

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
        report.add_entry(col="16 Hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_avg)
        report.add_entry(col="17 Hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p50)
        report.add_entry(col="18 Hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p90)
        report.add_entry(col="19 Hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p99)
        report.add_entry(col="20 Hyps_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_hyps_p100)
        report.add_entry(
            col="21 Word_end_hyps_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_avg
        )
        report.add_entry(
            col="22 Word_end_hyps_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p50
        )
        report.add_entry(
            col="23 Word_end_hyps_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p90
        )
        report.add_entry(
            col="24 Word_end_hyps_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_word_end_hyps_p99
        )
        report.add_entry(
            col="25 Word_end_hyps_p100",
            row=f"{idx}_{recog_result.descriptor}",
            var=recog_result.step_word_end_hyps_p100,
        )
        report.add_entry(col="26 Trees_avg", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_avg)
        report.add_entry(col="27 Trees_p50", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p50)
        report.add_entry(col="28 Trees_p90", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p90)
        report.add_entry(col="29 Trees_p99", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p99)
        report.add_entry(col="30 Trees_p100", row=f"{idx}_{recog_result.descriptor}", var=recog_result.step_trees_p100)

    return report
