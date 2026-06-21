"""Resolve a paper table's cell Variables to a plain ``data.json`` (numbers only).

This is the *data* half of the split table pipeline.
The Sis graph produces only the numbers, which genuinely depend on upstream metric jobs:
:class:`WriteTableDataJob` depends on the cell Variables and dumps ``{columns, rows}`` with raw values.
All presentation -- captions, headers, units, layout, col-align --
lives in authored spec files in the paper repo,
and is applied by the local ``scripts/render_tables.py``.
So a caption or header tweak is a one-second local re-render, never a manager restart;
the manager only runs when an actual number changes.

This file is also runnable directly for the live PREVIEW refresh
(``python table_data.py --refresh-preview <dir>``); see the bottom of the file.
"""

import json
import os
import sys
from functools import reduce
from typing import List, Dict, Any

# Make this file runnable directly so the preview refresher (bottom) is a standalone CLI:
# add the recipe + sisyphus dirs to sys.path so the module's own ``from sisyphus import ...`` resolves.
# This mirrors the setup the ``sis_tools`` scripts use;
# a normal import already has ``__package__`` set and skips it,
# so it only kicks in when the file is started as a script.
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(6), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = f"{_setup_base_dir}/tools/sisyphus"


def _setup():
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        os.environ.setdefault("SIS_GLOBAL_SETTINGS_FILE", f"{_setup_base_dir}/settings.py")


_setup()

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

    def run(self):
        # Strict (ignore_exception=False): every input is available, so any .get() failure is a real bug.
        data = _write_table_data(
            columns=self.columns, rows=self.rows, source=self.source, out_path=self.out_data.get_path()
        )
        print(json.dumps(data, indent=1))


# Shared resolution helpers, used by WriteTableDataJob.run above and the preview refresher below.

_PENDING = "·"  # pending glyph for a not-yet-computed cell (preview only)
_PENDING_SENTINEL = object()  # marks a cell whose .get() was tolerated-as-unavailable


def _instanciate(rows, *, ignore_exception: bool):
    # Like instanciate_delayed_copy (calls .get() on every DelayedBase / Variable, not in-place),
    # but with ignore_exception=True,
    # a .get() that raises (a not-yet-computed cell) becomes a sentinel instead of propagating
    # -- that is the only difference between the real job and the preview.
    import tree
    from sisyphus.delayed_ops import DelayedBase

    def _get(x):
        if isinstance(x, DelayedBase):
            # Preview: a not-yet-produced output is pending. Check is_set() first (no worker guard):
            # an unfinished Variable with a backup would otherwise return the backup instead of raising.
            if ignore_exception and hasattr(x, "is_set") and not x.is_set():
                return _PENDING_SENTINEL
            try:
                return x.get()
            except Exception:
                if ignore_exception:
                    return _PENDING_SENTINEL
                raise
        return x

    return tree.map_structure(_get, rows)


def _resolve_cell(cell):
    # Raw value of one already-instanciated cell: number | None | literal str | pending glyph.
    if not isinstance(cell, dict):
        return cell  # literal passed through (method names, "n/a", ...)
    val = cell.get("var", cell.get("value"))
    if val is _PENDING_SENTINEL:
        return _PENDING
    if cell.get("key") is not None and isinstance(val, dict):
        val = val.get(cell["key"])
    if val is None:
        return None
    return val if isinstance(val, str) else float(val)


def _write_table_data(*, columns, rows, source, out_path, ignore_exception: bool = False):
    """Shared core of :meth:`WriteTableDataJob.run` and the preview refresher.

    Resolves every cell's Variable (``.get()``) and writes ``data.json`` to ``out_path``.
    With ``ignore_exception=True`` (the preview),
    a cell whose ``.get()`` raises -- a not-yet-computed metric -- becomes the pending glyph instead of propagating;
    the strict default (the real job) lets it raise,
    since the job only runs once every input is available.
    """
    rows_in = _instanciate(rows, ignore_exception=ignore_exception)
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
        # a dict cell whose Variable resolves to None is kept as null (the column's missing placeholder).
        cells = {}
        srcs = {}
        for k in columns:
            cell = row_cells.get(k)
            if cell is None:
                continue
            cells[k] = _resolve_cell(cell)
            if isinstance(cell, dict) and cell.get("src"):
                srcs[k] = cell["src"]
        out_rows.append({"label": row.get("label", ""), "cells": cells, "src": srcs})

    data = {"source": source, "columns": list(columns), "rows": out_rows}
    with open(out_path, "wt") as f:
        json.dump(data, f, indent=1)
    return data


# ----------------------------------------------------------------------------------------
# Live preview, DECOUPLED from the job graph.
#
# Sisyphus' ``update()`` hook can't drive a live partial preview:
# it only fires once a job's WHOLE current input set is available
# (the ``while self._sis_all_path_available() and ...`` gate in ``Job._sis_runnable``),
# so it never reacts to "one of N cells just landed".
# Instead the preview is rebuilt from disk on demand:
#
# - ``write_preview_manifest`` gzip-pickles, per table, ``(columns, rows, source)``
#   -- the rows already hold the real ``Variable`` objects,
#   so this captures everything with zero custom serialization
#   (it is the same content sisyphus pickles for the job in ``Job._sis_setup_directory``).
#   Written once per config load (by the recipe's ``build_preview_tables``),
#   so it exists long before the cells finish.
# - ``refresh_preview`` unpickles it and runs the SAME ``_write_table_data`` with ``ignore_exception=True``,
#   so unfinished cells become pending.
#   It needs no manager and no graph --
#   run it any time while jobs are still landing:
#   ``python table_data.py --refresh-preview output/tables-data-preview``.
# ----------------------------------------------------------------------------------------
def write_preview_manifest(name, columns, rows, source, out_dir):
    """Gzip-pickle a table's ``(columns, rows, source)`` to ``<name>.manifest.pkl`` for later refresh."""
    import gzip
    import pickle

    with gzip.open(os.path.join(out_dir, f"{name}.manifest.pkl"), "wb") as f:
        pickle.dump({"columns": columns, "rows": rows, "source": source}, f)


def _has_pending(data):
    """True if any resolved cell is still the pending glyph,
    i.e. the table is not yet fully finished."""
    for row in data["rows"]:
        for v in row.get("cells", {}).values():
            if v == _PENDING:
                return True
    return False


def _final_table_path(manifest_dir, name):
    """The real WriteTableDataJob output for ``name`` (``tables-data/<name>.data.json``),
    derived from the preview dir by dropping the ``-preview`` suffix
    (``tables-data-preview`` -> ``tables-data``).
    Returns None when the preview dir is not the standard ``*-preview`` sibling,
    or when the real file does not exist yet (the job has not run)."""
    md = manifest_dir.rstrip("/")
    if not md.endswith("-preview"):
        return None
    real = os.path.join(md[: -len("-preview")], f"{name}.data.json")
    return real if os.path.exists(real) else None


def _diff_table_data(preview, final):
    """Short, human-readable list of where a fully-finished preview disagrees with the real job output."""
    diffs = []
    if preview.get("source") != final.get("source"):
        diffs.append(f"source: preview={preview.get('source')!r} final={final.get('source')!r}")
    if preview.get("columns") != final.get("columns"):
        diffs.append(f"columns: preview={preview.get('columns')} final={final.get('columns')}")
    p_rows, f_rows = preview.get("rows", []), final.get("rows", [])
    if len(p_rows) != len(f_rows):
        diffs.append(f"row count: preview={len(p_rows)} final={len(f_rows)}")
    for i, (pr, fr) in enumerate(zip(p_rows, f_rows)):
        if pr == fr:
            continue
        label = pr.get("label", fr.get("label", ""))
        pc, fc = pr.get("cells", {}), fr.get("cells", {})
        for k in sorted(set(pc) | set(fc)):
            if pc.get(k) != fc.get(k):
                diffs.append(f"row {i} ({label!r}) col {k!r}: preview={pc.get(k)!r} final={fc.get(k)!r}")
    return diffs


def refresh_preview(manifest_dir):
    """Re-resolve every ``<name>.manifest.pkl`` in ``manifest_dir`` from current disk -> ``<name>.data.json``.

    Sanity check: once a table is fully finished (no pending cell),
    its refreshed preview MUST equal the real ``WriteTableDataJob`` output (``tables-data/<name>.data.json``).
    Both render paths share ``_write_table_data``,
    so a divergence is a bug (a stale final output, or the two paths drifting apart),
    never an expected state -- raise loudly rather than render a wrong table.
    """
    import glob
    import gzip
    import pickle

    n = 0
    mismatches = []
    for mpath in sorted(glob.glob(os.path.join(manifest_dir, "*.manifest.pkl"))):
        with gzip.open(mpath, "rb") as f:
            manifest = pickle.load(f)
        name = os.path.basename(mpath)[: -len(".manifest.pkl")]
        preview_path = os.path.join(manifest_dir, f"{name}.data.json")
        data = _write_table_data(
            columns=manifest["columns"],
            rows=manifest["rows"],
            source=manifest.get("source"),
            out_path=preview_path,
            ignore_exception=True,
        )
        n += 1
        # Only finished tables can be cross-checked; a pending cell means the final job has not run yet.
        if _has_pending(data):
            print(f"{name}: pending ❌")
            continue
        print(f"{name}: finished")
        final_path = _final_table_path(manifest_dir, name)
        if not final_path:
            continue
        with open(preview_path) as f:
            preview_data = json.load(f)
        with open(final_path) as f:
            final_data = json.load(f)
        if preview_data != final_data:
            mismatches.append((name, _diff_table_data(preview_data, final_data)))
    print(f"refreshed {n} preview table(s) in {manifest_dir}")
    if mismatches:
        msg = ["preview vs final table-data mismatch (the two render paths diverged -- a bug):"]
        for name, diffs in mismatches:
            msg.append(f"  {name}.data.json:")
            msg.extend(f"    {d}" for d in diffs[:20])
            if len(diffs) > 20:
                msg.append(f"    ... ({len(diffs) - 20} more)")
        raise AssertionError("\n".join(msg))
    return n


def _main():
    import argparse
    import sisyphus.toolkit
    from sisyphus import gs

    sys.stdout.reconfigure(line_buffering=True)

    # We are not a real worker, but we want to read finished Variable values directly:
    # satisfy Variable.get()'s check_is_worker assert, and make an unfinished Variable RAISE
    # (so ignore_exception turns it into the pending glyph instead of a placeholder string).
    sisyphus.toolkit._sis_running_in_worker = True
    gs.RAISE_VARIABLE_NOT_SET_EXCEPTION = True

    parser = argparse.ArgumentParser(description="refresh the preview data.json files from current disk state")
    parser.add_argument(
        "--refresh-preview",
        metavar="DIR",
        required=True,
        help="dir with <name>.manifest.pkl files; re-resolve each from disk and rewrite <name>.data.json",
    )
    args = parser.parse_args()
    refresh_preview(args.refresh_preview)


if __name__ == "__main__":
    _main()
