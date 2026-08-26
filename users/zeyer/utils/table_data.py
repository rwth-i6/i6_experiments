"""
Generic ``WriteTableDataJob``: resolve a table of cells to plain values and write JSON + TSV.

Some cells are Sisyphus values (a ``Variable`` resolved via ``.get()``,
or a ``Path``/``Variable`` pointing to a text or JSON file),
which only become concrete numbers once their upstream jobs finish.
This job depends on those cells and resolves them at run time,
so the table itself is a normal Sis output that updates when a number changes.

A lighter, self-contained cousin of
:class:`i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data.WriteTableDataJob`
(which additionally carries a paper-table cell-spec);
this one is "columns + rows of cells -> JSON and TSV",
plus the same live-preview mechanism for incomplete tables (see the bottom of the file).
"""

from __future__ import annotations

import os
import sys
from functools import reduce
from typing import Any, Dict, List, Optional, Sequence

# Make this file runnable directly for the live preview refresh
# (``python table_data.py --refresh-preview <dir>``, see the bottom of the file):
# add the recipe + sisyphus dirs to sys.path so ``from sisyphus import ...`` resolves.
# A normal import has ``__package__`` set and skips this.
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)


def _setup():
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.utils"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        _sis_dir = f"{_setup_base_dir}/tools/sisyphus"
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        os.environ.setdefault("SIS_GLOBAL_SETTINGS_FILE", f"{_setup_base_dir}/settings.py")


_setup()

from sisyphus import Job, Task  # noqa: E402  (after the standalone-CLI sys.path setup)


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


# ---- Live preview (incomplete tables) ----
#
# Mirrors :mod:`i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.table_data`:
# the Sis job above only finishes once every cell's upstream job finished, so for a live view
# of an INCOMPLETE table:
# - ``write_preview_manifest`` gzip-pickles, per table, (columns, rows) at config-load time;
# - ``refresh_preview`` (CLI: ``python table_data.py --refresh-preview <dir>``) re-resolves each
#   manifest from current disk state -- a pending cell becomes the ``·`` glyph -- and writes
#   ``<name>.data.json`` (the same flat list-of-rows JSON as the job's ``table.json``);
#   when the real job's final table exists (``<dir>/../tables-data/<name>.data.json``),
#   that is copied instead, so preview and final never diverge.

_PENDING = "·"  # pending glyph for a not-yet-computed cell (preview only)


def write_preview_manifest(name: str, columns: Sequence[str], rows, out_dir: str):
    """Gzip-pickle a table's (columns, rows) to ``<name>.manifest.pkl`` for later refresh."""
    import gzip
    import pickle

    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, f"{name}.manifest.pkl"), "wb") as f:
        pickle.dump({"columns": list(columns), "rows": rows}, f)


def _resolve_tolerant(cell):
    from sisyphus.delayed_ops import DelayedBase

    # is_set() first (no worker guard): an unfinished Variable with a backup
    # would otherwise return the backup instead of raising.
    if isinstance(cell, DelayedBase) and hasattr(cell, "is_set") and not cell.is_set():
        return _PENDING
    try:
        return WriteTableDataJob._resolve(cell)
    except Exception:
        return _PENDING


def refresh_preview(manifest_dir: str):
    """Re-resolve every ``<name>.manifest.pkl`` in ``manifest_dir`` from current disk state
    -> ``<name>.data.json`` (pending cells = the ``·`` glyph);
    a finished job's final table is copied instead."""
    import glob
    import gzip
    import json
    import pickle
    import shutil

    n = 0
    md = manifest_dir.rstrip("/")
    final_dir = os.path.join(os.path.dirname(md), "tables-data")
    for mpath in sorted(glob.glob(os.path.join(manifest_dir, "*.manifest.pkl"))):
        with gzip.open(mpath, "rb") as f:
            manifest = pickle.load(f)
        name = os.path.basename(mpath)[: -len(".manifest.pkl")]
        preview_path = os.path.join(manifest_dir, f"{name}.data.json")
        final_path = os.path.join(final_dir, f"{name}.data.json")
        if os.path.exists(final_path):  # dangling output symlink -> False, i.e. job not finished
            shutil.copyfile(final_path, preview_path)
            print(f"{name}: final")
        else:
            rows = [{k: _resolve_tolerant(row.get(k)) for k in manifest["columns"]} for row in manifest["rows"]]
            with open(preview_path, "w") as f:
                json.dump(rows, f, indent=2)
                f.write("\n")
            n_pending = sum(1 for r in rows for v in r.values() if v == _PENDING)
            print(f"{name}: preview ({n_pending} pending cell(s))")
        n += 1
    print(f"refreshed {n} preview table(s) in {manifest_dir}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument(
        "--refresh-preview",
        metavar="DIR",
        required=True,
        help="dir with <name>.manifest.pkl files; re-resolve each from disk and rewrite <name>.data.json",
    )
    _args = _ap.parse_args()
    refresh_preview(_args.refresh_preview)
