"""
Check whether we are in manager
"""

import sys
import sisyphus.manager


def check_in_manager() -> bool:
    """
    Check whether we are in manager.

    This is not exactly the opposite of the check whether we are in worker
    (see :func:`tk.running_in_worker`):
    E.g. there are other ways how the Sis config could have been loaded,
    via some external scripts, or Sis console, or so.
    We really only return True here when we detect the manager.

    The manager is detected currently by checking whether :func:`sisyphus.manager.manager`
    is in the call stack.
    """
    manager_func = sisyphus.manager.manager
    frame = sys._getframe()
    assert frame
    while frame:
        if frame.f_code is manager_func.__code__:
            return True
        frame = frame.f_back
    return False
