"""Ported from speech-llm c49559ce src/speech_llm/sae/json_io.py.

Strict-JSON writing for the SAE analysis jobs.

Many statistics in these reports are legitimately undefined -- a selector with no live group, a
correlation over an empty cell, a mean of nothing. ``json.dump`` spells those ``NaN``/``Infinity``,
which Python reads back happily and every strict parser (jq, JS, Rust, most dashboards) rejects, so a
report file silently becomes unloadable outside Python. Write ``null`` instead and let ``allow_nan``
turn anything the converter cannot reach into an error at write time rather than at read time.
"""

from __future__ import annotations

import json
import math

__all__ = ["dump_json", "json_safe"]


def json_safe(o):
    """Recursively replace non-finite floats with ``None``; other values pass through unchanged."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    return o


def dump_json(obj, path: str, **kw) -> None:
    """``json.dump`` to ``path`` with undefined values as ``null`` and stray non-finites fatal."""
    with open(path, "w") as fh:
        json.dump(json_safe(obj), fh, allow_nan=False, **kw)
