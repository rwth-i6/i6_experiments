from typing import List

from i6_core.summary.wer import TableReport

from .decode_config import DecodeRecogResult


def create_report(recog_results: List[DecodeRecogResult], title: str = "Experiments") -> TableReport:
    report = TableReport(title, precision=5)

    for idx, recog_result in enumerate(recog_results, start=1):
        report.add_entry(col="1 corpus", row=f"{idx}_{recog_result.descriptor}", var=recog_result.corpus_name)
        report.add_entry(col="2 PER", row=f"{idx}_{recog_result.descriptor}", var=recog_result.per)
        report.add_entry(col="3 Del", row=f"{idx}_{recog_result.descriptor}", var=recog_result.deletion)
        report.add_entry(col="4 Ins", row=f"{idx}_{recog_result.descriptor}", var=recog_result.insertion)
        report.add_entry(col="5 Sub", row=f"{idx}_{recog_result.descriptor}", var=recog_result.substitution)

    return report
