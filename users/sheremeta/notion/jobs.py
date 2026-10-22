"""
The sisyphus job that writes one row to a Notion database.
"""

import datetime
import json
import os
import socket
import stat
from typing import Any, Dict, Optional, Tuple

from sisyphus import Job, Task, gs
from sisyphus.delayed_ops import DelayedBase

from . import api
from .properties import COLUMN_KINDS, property_value

#: where the integration token is read from when the sisyphus settings set no NOTION_TOKEN_FILE
DEFAULT_TOKEN_FILE = "~/.config/notion/token"

#: the kind of a column and its value, the value a plain value or anything sisyphus resolves at run time
Column = Tuple[str, Any]


class PushNotionRowJob(Job):
    """
    Writes one row to a Notion database. The row whose key column holds the key already is updated, otherwise a
    row is created, so a second push of the same experiment replaces its values. Only the given columns are
    written, the other columns of the row keep what is in them.
    """

    def __init__(
        self,
        *,
        data_source_id: str,
        key: str,
        columns: Dict[str, Column],
        timestamp_column: Optional[str] = None,
        host_column: Optional[str] = None,
    ):
        """
        :param data_source_id: the data source of the database, the id in its ``collection://`` URL
        :param key: the column that identifies the row, a title or rich_text column of ``columns``
        :param columns: per column name its kind, see :data:`COLUMN_KINDS`, and its value. A value can be a
            ``tk.Variable``, ``var["key"]``, ``var.function(f)`` or any other delayed object sisyphus resolves
        :param timestamp_column: a date column set to the time of the push, None for none
        :param host_column: a rich_text column set to the machine the push ran on, None for none
        """
        assert key in columns and columns[key][0] in ("title", "rich_text"), f"key {key!r} must be a text column"
        for name, (kind, _value) in columns.items():
            assert kind in COLUMN_KINDS, f"column {name!r} has the unknown kind {kind!r}"
        self.data_source_id = data_source_id
        self.key = key
        self.columns = dict(columns)
        self.timestamp_column = timestamp_column
        self.host_column = host_column
        self.out_page = self.output_var("page")
        self.out_properties = self.output_path("properties.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        values = {name: (kind, _resolved(value)) for name, (kind, value) in self.columns.items()}
        if self.timestamp_column is not None:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            values[self.timestamp_column] = ("date", now)
        if self.host_column is not None:
            values[self.host_column] = ("rich_text", socket.getfqdn())
        properties = {name: property_value(kind, value) for name, (kind, value) in values.items()}
        with open(self.out_properties.get_path(), "w") as f:
            json.dump(properties, f, indent=1, sort_keys=True)

        key_kind, key_value = values[self.key]
        if not isinstance(key_value, str) or not key_value:
            raise ValueError(f"key column {self.key!r} needs a non-empty string, got {key_value!r}")
        notion = api.NotionApi(_read_token())
        matches = notion.query_equals(self.data_source_id, self.key, key_kind, key_value)
        if len(matches) > 1:
            raise ValueError(f"several rows hold {self.key} {key_value!r}, expected at most one")
        if matches:
            page = notion.update_page(matches[0]["id"], properties)
        else:
            page = notion.create_page(self.data_source_id, properties)
        self.out_page.set({"id": page["id"], "url": page.get("url"), "created": not matches})


def _read_token() -> str:
    path = os.path.expanduser(getattr(gs, "NOTION_TOKEN_FILE", DEFAULT_TOKEN_FILE))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no Notion token at {path}, put the integration token there with mode 600")
    if os.stat(path).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(f"{path} is open to group or others, run chmod 600 {path}")
    with open(path) as f:
        token = f.read().strip()
    if not token:
        raise ValueError(f"{path} is empty")
    return token


def _resolved(value: Any) -> Any:
    if isinstance(value, DelayedBase):
        return _resolved(value.get())
    if isinstance(value, (list, tuple)):
        return [_resolved(item) for item in value]
    return value
