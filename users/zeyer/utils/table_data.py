"""
Generic ``WriteTableDataJob``: resolve a table of cells to plain values and write JSON + TSV.

Some cells are Sisyphus values (a ``Variable`` resolved via ``.get()``,
or a ``Path``/``Variable`` pointing to a text or JSON file),
which only become concrete numbers once their upstream jobs finish.
This job depends on those cells and resolves them at run time,
so the table itself is a normal Sis output that updates when a number changes.

A lighter, self-contained cousin of
:class:`i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data.WriteTableDataJob`
(which additionally carries a paper-table cell-spec and live-preview mechanism);
this one is just "columns + rows of cells -> JSON and TSV".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sisyphus import Job, Task


class WriteTableDataJob(Job):
    """
    Resolve a table of cells to plain values and write ``table.json`` and ``table.tsv``.

    :param columns: ordered column keys (the column order and the TSV/JSON header).
    :param rows: list of row dicts mapping each column key to a cell.
        A cell is one of:
        a literal (str / int / float / bool / None),
        or a Sisyphus value resolved at run time --
        a ``Variable`` via ``.get()``,
        or a ``Path`` / ``Variable`` whose file is read and parsed as JSON, else float, else stripped text.
        A missing or ``None`` cell is written as an empty TSV field and ``null`` in JSON.
    :param sort_by: optional column keys to sort the (resolved) rows by before writing.
    :param float_fmt: printf format for float cells in the TSV (JSON keeps full precision).
    """

    def __init__(
        self,
        *,
        columns: Sequence[str],
        rows: List[Dict[str, Any]],
        sort_by: Optional[Sequence[str]] = None,
        float_fmt: str = "%.6g",
    ):
        super().__init__()
        self.columns = list(columns)
        self.rows = rows
        self.sort_by = list(sort_by) if sort_by else None
        self.float_fmt = float_fmt
        self.out_json = self.output_path("table.json")
        self.out_tsv = self.output_path("table.tsv")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _resolve(cell):
        import json

        if cell is None or isinstance(cell, (str, int, float, bool)):
            return cell
        # A Sisyphus Path/Variable backed by a file: read and parse it (JSON, else float, else text).
        get_path = getattr(cell, "get_path", None)
        if callable(get_path):
            with open(get_path()) as f:
                txt = f.read().strip()
            try:
                return json.loads(txt)
            except ValueError:
                pass
            try:
                return float(txt)
            except ValueError:
                return txt
        # A Variable / DelayedBase whose value comes via .get().
        get = getattr(cell, "get", None)
        if callable(get):
            return get()
        return cell

    def _fmt_tsv(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return self.float_fmt % value
        return str(value)

    def run(self):
        import json

        out = [{k: self._resolve(row.get(k)) for k in self.columns} for row in self.rows]
        if self.sort_by:
            out.sort(key=lambda d: tuple(d[k] for k in self.sort_by))

        with open(self.out_json.get_path(), "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")

        with open(self.out_tsv.get_path(), "w") as f:
            f.write("\t".join(self.columns) + "\n")
            for d in out:
                f.write("\t".join(self._fmt_tsv(d[k]) for k in self.columns) + "\n")
