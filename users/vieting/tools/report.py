__all__ = ["Report"]
from sisyphus import *

from typing import Dict, Union, List

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
