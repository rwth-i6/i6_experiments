from __future__ import annotations
import copy
from sisyphus import tk
from typing import Any, Dict, List, Optional
from io import StringIO


class SummaryReport:
    """
    Report that summarizes results from multiple experiments.
    The results are organized by column. The columns can be fields like "name", "lm scale" or "WER". Optionally, the rows can be sorted by their value in one of the columns, e.g. WER.

    This report can be registered as a sisyphus target.
    """

    def __init__(
        self,
        col_names: Optional[List[str]] = None,
        col_sort_key: Optional[str] = None,
        sort_high_to_low: bool = False,
        precision: int = 2,
    ) -> None:
        """
        :param col_keys: Names of the columns in the table
        :param col_sort_key: If a key is provided, the rows are sorted by this key.
        :param sort_high_to_low: If True, the rows are sorted such that the highest value is on top. Otherwise the lowest value is on top. Only relevant is <col_sort_key> is not None.
        :param precision: Display precision of floating point values in the table.
        """

        self._col_names = col_names or []
        self._col_sort_key = col_sort_key
        self._sort_high_to_low = sort_high_to_low
        self._precision = precision

        self._collapsed_rows = []

        self._data: List[dict] = []

    def copy_structure(self, other: SummaryReport) -> None:
        """
        Copies the structure (column names, layout, sorting etc.) from another SummmaryReport.

        :param other: SummaryReport from which the structure is copied.
        """
        self._col_sort_key = other._col_sort_key
        self._sort_high_to_low = other._sort_high_to_low
        self._precision = other._precision
        self._col_names = other._col_names

    def add_column(
        self,
        name: str,
        position: Optional[int] = None,
        default_value: Optional[Any] = None,
    ) -> None:
        if position is None:
            position = len(self._col_names)
        self._col_names.insert(position, name)

        if default_value is None:
            return
        for row in self._data:
            row.setdefault(name, default_value)

    def merge_report(
        self,
        other: SummaryReport,
        update_structure: bool = False,
        collapse_rows: bool = False,
    ) -> None:
        """
        Adds all the data from another report into our own.

        :param other: SummaryReport from which we add the data.
        :param update_structure: If True, we take other the structure from <other> before merging.
        :param collapse_rows: If True, all the data from <other> is collapsed into a single row that contains only the best one.
        """

        if update_structure:
            self.copy_structure(other)

        prev_num_rows = len(self._data)
        for row in other._data:
            self.add_row(row)
        added_rows = len(self._data) - prev_num_rows

        if added_rows == 0:
            return

        if collapse_rows:
            self._collapsed_rows.append((prev_num_rows, prev_num_rows + added_rows))
        else:
            self._collapsed_rows.extend(
                [
                    (prev_num_rows + from_index, prev_num_rows + to_index)
                    for from_index, to_index in other._collapsed_rows
                ]
            )

    def add_row(self, row: Dict[str, Any]) -> None:
        # avoid duplication
        if not any([row == exist_row for exist_row in self._data]):
            self._data.append(row)

    def _dict_to_row(self, row: Dict[str, Any]) -> None:
        return [row.get(name, " ") for name in self._col_names]

    def _try_convert_to_float(self, val: Any) -> Any:
        try:
            return float(val)
        except ValueError:
            if not val.strip():
                return 0.0 if self._sort_high_to_low else float("inf")
            else:
                return val

    def _get_col_sort_index(self) -> int:
        col_sort_index = None
        if self._col_sort_key is not None and self._col_sort_key in self._col_names:
            col_sort_index = self._col_names.index(self._col_sort_key)
        else:
            col_sort_index = 0

        return col_sort_index

    def _get_string_list_rows(self) -> List[List[str]]:
        """Transformed data -> Get tk.Variables and round floats"""
        processed_data = []
        for row in self._data:
            row_list = self._dict_to_row(row)
            processed_row = []
            for val in row_list:
                if isinstance(val, tk.Variable):
                    if val.is_set():
                        v = val.get()
                    else:
                        v = " "
                else:
                    v = val
                if isinstance(v, float):
                    v = str(round(v, self._precision))
                else:
                    v = str(v)
                processed_row.append(v)
            # # Avoid duplicates
            # if not any([exist_row == processed_row for exist_row in processed_data]):
            processed_data.append(processed_row)

        return processed_data

    def _collapse_rows(self, data: List[List[str]]) -> List[List[str]]:
        """Collapse ranges given by self._collapsed_rows"""

        col_sort_index = self._get_col_sort_index()

        collapsed_data = []
        prev_to_index = 0
        for from_index, to_index in self._collapsed_rows:
            if prev_to_index < from_index:
                collapsed_data.extend(data[prev_to_index:from_index])
            prev_to_index = to_index

            # Data with no value at <col_sort_index> gets discarded
            remove_empty = [
                row for row in data[from_index:to_index] if row[col_sort_index].strip()
            ]
            # If nothing would be left, we have to take one of the valueless rows regardless
            if not remove_empty:
                collapsed_data.append(data[from_index])
                continue

            # Otherwise, pick find the row with the best value and append it
            opt_func = max if self._sort_high_to_low else min
            collapsed_data.append(
                opt_func(
                    remove_empty,
                    key=lambda row: self._try_convert_to_float(row[col_sort_index]),
                )
            )
        if prev_to_index < len(data):
            collapsed_data.extend(data[prev_to_index:])

        # Sort by value at <col_sort_index>
        collapsed_data.sort(
            key=lambda row: self._try_convert_to_float(row[col_sort_index]),
            reverse=self._sort_high_to_low,
        )

        return collapsed_data

    def __call__(self) -> str:
        processed_data = self._get_string_list_rows()
        collapsed_data = self._collapse_rows(processed_data)

        # Print the actual table
        f = StringIO()

        col_widths = [
            max(len(c_name), max(len(row[c]) for row in collapsed_data))
            for c, c_name in enumerate(self._col_names)
        ]

        def write_row_separator() -> None:
            """Writes a horizontal line with the right column separation"""
            for c_width in col_widths:
                f.write("+")
                f.write("-" * (c_width + 2))
            f.write("+\n")

        write_row_separator()
        for c_width, c_name in zip(col_widths, self._col_names):
            f.write(f"| {c_name:^{c_width}} ")
        f.write("|\n")
        write_row_separator()

        for row in collapsed_data:
            for value, c_width in zip(row, col_widths):
                f.write(f"| {value:<{c_width}} ")
            f.write("|\n")
        write_row_separator()

        return f.getvalue()
