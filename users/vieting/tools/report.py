__all__ = ["Report"]
import copy
from collections import defaultdict
from typing import Dict, Union, List
from sisyphus import *

_Report_Type = Dict[str, Union[tk.AbstractPath, str]]


class Report:
    def __init__(self, columns_start=None, columns_end=None):
        self._data = []
        self._columns_start = columns_start or []
        self._columns_end = columns_end or []

    @property
    def data(self):
        return self._data

    def add(self, report_args):
        self._data.append(report_args)

    def delete_column(self, column):
        for data in self._data:
            data.pop(column, None)

    def delete_redundant_columns(self, delete_columns_start=False, delete_columns_end=False, columns_skip=None):
        """
        Delete columns for which all entries have the same value.
        """
        if len(self._data) < 2:
            return

        columns_skip = columns_skip or []
        for col in self.get_columns():
            if (
                (col in columns_skip) or
                (not delete_columns_start and col in self._columns_start) or
                (not delete_columns_end and col in self._columns_end)
            ):
                continue
            values = [data.get(col, None) for data in self._data]
            if len(set(values)) == 1:
                self.delete_column(col)

    def delete_redundant_rows(self):
        rows_reduced = []
        for row in self._data:
            if row not in rows_reduced:
                rows_reduced.append(row)
        self._data = rows_reduced

    def get_columns(self):
        columns = set()
        for data in self._data:
            columns.update(data.keys())

        assert set(self._columns_start).issubset(columns), (
            f"not all start columns are in columns: {self._columns_start} vs. columns: {columns}")
        assert set(self._columns_end).issubset(columns), (
            f"not all end columns are in columns: {self._columns_end} vs. columns: {columns}")
        columns -= set(self._columns_start)
        columns -= set(self._columns_end)
        columns = self._columns_start + list(columns) + self._columns_end

        return columns

    def get_values(self):
        # fill values which are not given
        for col in self.get_columns():
            for idx in range(len(self._data)):
                if col not in self._data[idx]:
                    self._data[idx][col] = ""

        values_dict = {}
        for row_idx, data in enumerate(self._data):
            for col_idx, col in enumerate(self.get_columns()):
                values_dict["{}_{}".format(row_idx, col_idx)] = data[col]
        return values_dict

    def get_template(self):
        header = ",".join(self.get_columns())
        data = []
        for row in range(len(self._data)):
            data.append(",".join(["{{{}_{}}}".format(row, col) for col in range(len(self.get_columns()))]))
        return "\n".join([header] + data)

    def merge_report(self, other):
        self._data += other.data

    @classmethod
    def merge_reports(cls, report_list: List):
        if len(report_list) == 0:
            return Report()
        report = report_list.pop(0)
        for rprt in report_list:
            report.merge_report(rprt)
        return report

    def merge_eval_sets(self, eval_set_column: str):
        merged_dicts = defaultdict(lambda: defaultdict(dict))
        eval_sets = set()
        for d in self._data:
            # Create a key based on all items except eval_set_column and wer
            key = tuple((k, v) for k, v in d.items() if k not in [eval_set_column, "wer"])

            # Extract eval_set and "wer" values
            wer = d["wer"]
            eval_set = d[eval_set_column]
            eval_sets.add(eval_set)

            # Merge "wer" under the key specified by "eval_set"
            merged_dicts[key][eval_set] = wer

        # Convert the result back to the original list of dicts format
        result = []
        for base_key, merged_values in merged_dicts.items():
            # Recreate the base dictionary
            base_dict = {k: v for k, v in base_key}
            # Add the merged "wer" values
            base_dict.update(merged_values)
            result.append(base_dict)

        self._data = result
        self._columns_end = [col for col in self._columns_end if col != "wer"] + sorted(list(eval_sets))
