"""Publish generated reports safely during Sisyphus's periodic updates."""

from functools import partial
from pathlib import Path

from sisyphus import tk


def read_job_report(path, *, title):
    """A report dependency need not exist when OutputReport refreshes its page.

    Catch missing files only: permission/decoding errors should remain visible.
    Do not cache this result, so the next refresh publishes the completed file.
    """
    source = Path(path.get_path())
    try:
        return source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return (f"# {title}\n\nReport not available yet. Check the report job and its dependencies "
                f"in Sisyphus for queued, running or failed tasks.\n\nSource: `{source}`\n")


def register_job_report(name, path, *, title, update_frequency=300):
    # required still declares/schedules the dependency. It does not guard the
    # periodic callback in this Sisyphus version, hence the safe reader above.
    return tk.register_report(name, partial(read_job_report, path, title=title),
                              required=[path], update_frequency=update_frequency)
