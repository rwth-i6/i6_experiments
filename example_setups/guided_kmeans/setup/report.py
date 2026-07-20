from typing import List

from i6_core.summary.wer import TableReport

from .decode_config import DecodeRecogResult


def create_report(recog_results: List[DecodeRecogResult], title: str = "Experiments") -> TableReport:
    report = TableReport(title, precision=5)

    for idx, recog_result in enumerate(recog_results, start=1):
        row = f"{idx}_{recog_result.descriptor}"
        report.add_entry(col="1 corpus", row=row, var=recog_result.corpus_name)
        report.add_entry(col="2 PER", row=row, var=recog_result.per)
        report.add_entry(col="3 Del", row=row, var=recog_result.deletion)
        report.add_entry(col="4 Ins", row=row, var=recog_result.insertion)
        report.add_entry(col="5 Sub", row=row, var=recog_result.substitution)
        if recog_result.mean_cos_sim is not None:
            report.add_entry(col="6 MeanCosSim", row=row, var=recog_result.mean_cos_sim)
        if recog_result.l1_dist is not None:
            report.add_entry(col="7 L1Dist", row=row, var=recog_result.l1_dist)
        if recog_result.avg_total_score is not None:
            report.add_entry(col="8 AvgScore", row=row, var=recog_result.avg_total_score)

    return report
