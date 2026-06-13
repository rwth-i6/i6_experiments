"""Render LaTeX result tables from Sisyphus Variables, in the i6 ruled-table style.

Table generation is part of the Sis graph: a :class:`WriteLatexTableJob` depends on the
metric-output Variables and renders a ``table.tex`` (vertical rules + ``\\hline``, bold
bracketed-unit headers, optional bold-best) that the paper ``\\input``s. Auto-updates when the
upstream results do.
"""

from typing import Optional, List, Dict, Any
from sisyphus import Job, Task


def _latex_escape(s: str) -> str:
    # Escape the LaTeX specials that show up in our row labels (model names).
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


class WriteLatexTableJob(Job):
    """Render a LaTeX table (i6 ruled style) from result Variables.

    :param columns: ordered list of column dicts. Each: ``{"key", "header", "best"?}``.
        ``best`` in ``{"min", "max"}`` bolds the best numeric cell in that column. ``header`` is
        the column title WITHOUT bold/units markup -- it is wrapped in ``\\textbf{...}`` and should
        already carry its unit, e.g. ``"WBE [ms]"``.
    :param rows: ordered list of row dicts. Each: ``{"label", "cells": {col_key: cell}}``.
        A ``cell`` is a literal (str/number) OR a dict
        ``{"var": Variable|Path, "key"?: str, "mul"?: float, "fmt"?: str, "missing"?: str}``:
        ``var`` resolved at run time; ``key`` indexes the resolved dict; ``mul`` scales; ``fmt``
        formats (default ``"{:.1f}"``); ``missing`` is the placeholder (default ``"--"``).
        A row with ``label`` ``None`` and empty ``cells`` renders an ``\\hline`` separator.
        A row ``{"cline": True}`` renders a ``\\cline{2-N}`` (rule spanning all but the label
        column -- used to separate sub-rows within a grouped row whose label column should stay
        unbroken).
    :param label_header: header for the (first) row-label column, e.g. ``"Model"``.
    :param col_align: tabular column spec; default ``"|l|" + "r|"*len(columns)`` (ruled, numbers
        right-aligned -- with uniform decimals this aligns the decimal point).
    """

    def __init__(
        self,
        *,
        columns: List[Dict[str, Any]],
        rows: List[Dict[str, Any]],
        caption: Optional[str] = None,
        label: Optional[str] = None,
        label_header: str = "",
        col_align: Optional[str] = None,
        float_spec: str = "table",
        position: str = "tb",
    ):
        super().__init__()
        self.columns = columns
        self.rows = rows
        self.caption = caption
        self.label = label
        self.label_header = label_header
        self.col_align = col_align
        self.float_spec = float_spec
        self.position = position
        self.out_tex = self.output_path("table.tex")

    def tasks(self):
        yield Task("run", mini_task=True)

    @staticmethod
    def _cell_value(cell):
        """Return (numeric value or None, display string) for an already-resolved cell."""
        if cell is None:
            return None, ""
        if not isinstance(cell, dict):
            return None, str(cell)
        val = cell.get("var", cell.get("value"))
        if cell.get("key") is not None and isinstance(val, dict):
            val = val.get(cell["key"])
        if val is None:
            return None, cell.get("missing", "--")
        num = float(val)
        if cell.get("mul") is not None:
            num *= cell["mul"]
        return num, cell.get("fmt", "{:.1f}").format(num)

    def run(self):
        import os
        import sys
        import i6_experiments

        sys.path.insert(0, os.path.dirname(os.path.dirname(i6_experiments.__file__)))
        from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy

        rows = instanciate_delayed_copy(self.rows)
        cols = self.columns
        col_keys = [c["key"] for c in cols]

        grid = []
        for row in rows:
            if row.get("cline"):
                grid.append("CLINE")  # \cline{2-N} partial separator
                continue
            if row.get("label") is None and not row.get("cells"):
                grid.append(None)  # \hline separator
                continue
            cells = {k: self._cell_value(row.get("cells", {}).get(k)) for k in col_keys}
            grid.append((row.get("label", ""), cells))

        best_row = {}
        n_cols_total = 1 + len(cols)
        for c in cols:
            if c.get("best") in ("min", "max"):
                cand = [
                    (i, g[1][c["key"]][0])
                    for i, g in enumerate(grid)
                    if isinstance(g, tuple) and g[1][c["key"]][0] is not None
                ]
                if cand:
                    best_row[c["key"]] = (min if c["best"] == "min" else max)(cand, key=lambda iv: iv[1])[0]

        col_align = self.col_align or ("|l|" + "r|" * len(cols))
        out = [f"\\begin{{{self.float_spec}}}[{self.position}]", "  \\centering"]
        if self.caption:
            out.append(f"  \\caption{{{self.caption}}}")
        if self.label:
            out.append(f"  \\label{{{self.label}}}")
        out.append(f"  \\begin{{tabular}}{{{col_align}}}")
        out.append("    \\hline")
        hdr = ([f"\\textbf{{{self.label_header}}}"] if self.label_header else [""]) + [
            f"\\textbf{{{c['header']}}}" for c in cols
        ]
        out.append("    " + " & ".join(hdr) + r" \\ \hline\hline")
        for i, g in enumerate(grid):
            if g == "CLINE":
                out.append(f"    \\cline{{2-{n_cols_total}}}")
                continue
            if g is None:
                out.append("    \\hline")
                continue
            label, cells = g
            parts = [_latex_escape(str(label))]
            for c in cols:
                disp = cells[c["key"]][1]
                if best_row.get(c["key"]) == i and disp not in ("", "--"):
                    disp = f"\\textbf{{{disp}}}"
                parts.append(disp)
            out.append("    " + " & ".join(parts) + r" \\")
        out.append("    \\hline")
        out.append("  \\end{tabular}")
        out.append(f"\\end{{{self.float_spec}}}")
        out.append("")

        with open(self.out_tex.get_path(), "wt") as f:
            f.write("\n".join(out))
        print("\n".join(out))
