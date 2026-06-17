"""Resolve a paper table's cell Variables to a plain ``data.json`` (numbers only).

This is the *data* half of the split table pipeline.
The Sis graph produces only the numbers, which genuinely depend on upstream metric jobs:
:class:`WriteTableDataJob` depends on the cell Variables and dumps ``{columns, rows}`` with raw values.
All presentation -- captions, headers, units, layout, col-align --
lives in authored spec files in the paper repo,
and is applied by the local ``scripts/render_tables.py``.
So a caption or header tweak is a one-second local re-render, never a manager restart;
the manager only runs when an actual number changes.
"""

import json
from typing import List, Dict, Any
from sisyphus import Job, Task


class WriteTableDataJob(Job):
    """Resolve a table's cell Variables to raw numbers and write ``data.json``.

    :param columns: ordered list of column keys.
    :param rows: ordered list of row dicts. Each is one of:
        a data row ``{"label", "cells": {key: cell}}``,
        where a ``cell`` is a literal (str/number),
        ``None`` (absent -> empty),
        or a resolve-spec ``{"var": Variable|Path, "key"?: str, "src"?: str}``
        (``var`` resolved at run time, ``key`` indexes the resolved dict,
        ``src`` is the registered output name the number comes from -> a provenance comment);
        or a layout marker ``{"hline": True}`` (also ``{"label": None, "cells": {}}``) or ``{"cline": True}``,
        passed through verbatim for the render layer.
    :param source: free-text origin of the table (builder file :: function), emitted as a header comment.

    Output ``data.json``: ``{"source", "columns": [keys], "rows": [...]}``.
    A data row's ``cells`` maps key -> number (raw, pre-unit-scaling),
    ``null`` (resolved but missing),
    or a literal string (a method name, ``"n/a"``, ...);
    its ``src`` maps key -> registered output name (only for cells that carry one).
    An absent cell is omitted from ``cells`` entirely (rendered empty),
    keeping it distinct from a resolved-but-missing ``null`` (rendered with the column's missing placeholder).
    """

    # v1 collapsed absent cells and missing values both to null;
    # v2 omits absent cells instead.
    __sis_version__ = 2

    def __init__(self, *, columns: List[str], rows: List[Dict[str, Any]], source: str = None):
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.source = source
        self.out_data = self.output_path("data.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _resolve(cell):
        # Raw value of one already-instanciated, non-None cell: number | None | literal str/number.
        if not isinstance(cell, dict):
            return cell  # literal passed through (method names, "n/a", ...)
        val = cell.get("var", cell.get("value"))
        if cell.get("key") is not None and isinstance(val, dict):
            val = val.get(cell["key"])
        if val is None:
            return None
        return float(val)

    def run(self):
        import os
        import sys
        import i6_experiments

        sys.path.insert(0, os.path.dirname(os.path.dirname(i6_experiments.__file__)))
        from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy

        rows_in = instanciate_delayed_copy(self.rows)
        out_rows = []
        for row in rows_in:
            if row.get("cline"):
                out_rows.append({"cline": True})
                continue
            if row.get("hline2"):
                out_rows.append({"hline2": True})
                continue
            if row.get("hline") or (row.get("label") is None and not row.get("cells")):
                out_rows.append({"hline": True})
                continue
            row_cells = row.get("cells", {})
            # Omit absent cells (input None) so the render layer leaves them empty;
            # a dict cell whose Variable resolves to None is kept as null,
            # so the render layer shows the column's missing placeholder instead.
            cells = {}
            srcs = {}
            for k in self.columns:
                cell = row_cells.get(k)
                if cell is None:
                    continue
                cells[k] = self._resolve(cell)
                if isinstance(cell, dict) and cell.get("src"):
                    srcs[k] = cell["src"]
            out_rows.append({"label": row.get("label", ""), "cells": cells, "src": srcs})

        data = {"source": self.source, "columns": list(self.columns), "rows": out_rows}
        with open(self.out_data.get_path(), "wt") as f:
            json.dump(data, f, indent=1)
        print(json.dumps(data, indent=1))
