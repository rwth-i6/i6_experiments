"""
Check whether we are in manager
"""

import threading
from sisyphus.manager import Manager


def check_in_manager() -> bool:
    """
    Check whether we are in manager.

    This is not exactly the opposite of the check whether we are in worker
    (see :func:`tk.running_in_worker`):
    E.g. there are other ways how the Sis config could have been loaded,
    via some external scripts, or Sis console, or so.
    We really only return True here when we detect the manager.

    The manager is detected currently by checking the threads.
    """
    for thread in threading.enumerate():
        if isinstance(thread, Manager):
            return True
    return False
