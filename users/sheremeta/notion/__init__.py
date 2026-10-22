"""
Writes experiment results to a Notion database, one row per :class:`PushNotionRowJob`.

A job updates the row whose key column holds its key, or creates one, so pushing the same experiment again replaces
its values. A column value can be a plain value or anything sisyphus resolves at run time, a ``tk.Variable``,
``var["key"]`` or ``var.function(f)``, so a row can be put together from the outputs of other jobs while the graph
is built.

Setup, once per user:

1. Create an internal integration in Notion's integration settings with the capabilities to read, update and insert
   content, and add it to the database under the database's connections.
2. Store its token in ``~/.config/notion/token`` with mode 600 on every machine a manager runs on, or set
   ``NOTION_TOKEN_FILE`` in the sisyphus settings to another file. The token itself does not belong in the settings.
3. Create the database with the columns the jobs write. The data source id is the id in the database's
   ``collection://`` URL.

The job runs as a mini task, so on the machine of the manager, and that machine has to reach https://api.notion.com.
"""

from .jobs import PushNotionRowJob
from .properties import COLUMN_KINDS

__all__ = ["PushNotionRowJob", "COLUMN_KINDS"]
