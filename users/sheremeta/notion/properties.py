"""
Conversion of plain values into the values of Notion database columns.
"""

import math
import numbers
from typing import Any, Dict, List, Optional, Union

#: the column kinds a row can write, each the Notion property type of the same name
COLUMN_KINDS = ("title", "rich_text", "number", "select", "multi_select", "checkbox", "url", "date")

# Notion takes at most this many characters per rich text item and this many items per value
_TEXT_ITEM_CHARS = 2000
_MAX_TEXT_ITEMS = 100


def property_value(kind: str, value: Any) -> Dict[str, Any]:
    """
    Builds the value of one column the way the Notion API takes it, None clears the column.

    :param kind: one of :data:`COLUMN_KINDS`
    :param value: a string for text, select and url columns, an ISO 8601 date or date and time for date columns, a
        finite real number for number columns, a bool for checkbox columns, a sequence of strings for multi_select
    :return: the value under the key of its kind, for example ``{"number": 5.08}``
    """
    if kind in ("title", "rich_text"):
        return {kind: _text_items(value)}
    if kind == "number":
        return {"number": _number(value)}
    if kind == "select":
        return {"select": None if value is None else {"name": _option_name(value)}}
    if kind == "multi_select":
        return {"multi_select": [{"name": _option_name(name)} for name in value or ()]}
    if kind == "checkbox":
        if not isinstance(value, bool):
            raise ValueError(f"checkbox column needs a bool, got {value!r}")
        return {"checkbox": value}
    if kind == "url":
        return {"url": None if value is None else str(value)}
    if kind == "date":
        return {"date": None if value is None else {"start": str(value)}}
    raise ValueError(f"unknown column kind {kind!r}, expected one of {COLUMN_KINDS}")


def _text_items(value: Any) -> List[Dict[str, Any]]:
    text = "" if value is None else str(value)
    items = [text[start : start + _TEXT_ITEM_CHARS] for start in range(0, len(text), _TEXT_ITEM_CHARS)]
    if len(items) > _MAX_TEXT_ITEMS:
        raise ValueError(f"a text of {len(text)} characters needs more than {_MAX_TEXT_ITEMS} rich text items")
    return [{"type": "text", "text": {"content": item}} for item in items]


def _number(value: Any) -> Optional[Union[int, float]]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"number column needs a finite number, got {value!r}")
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _option_name(value: Any) -> str:
    name = str(value)
    if "," in name:
        raise ValueError(f"Notion option names cannot contain commas, got {name!r}")
    return name
