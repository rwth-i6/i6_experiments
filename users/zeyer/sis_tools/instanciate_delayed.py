"""
:func:`i6_core.util.instanciate_delayed` but not operating inplace,
which can potentially be dangerous and lead to bugs which are hard to track down
and which are not immediately present.
"""

from typing import Any
import sys
from sisyphus.delayed_ops import DelayedBase
import tree
from i6_core.util import instanciate_delayed as _instanciate_delayed_old
from sisyphus.hash import short_hash


use_buggy_old_instanciate_delayed = False
enable_check_behavior_change_new_to_old = True  # for now...


def instanciate_delayed(o: Any) -> Any:
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    In contrast to :func:`i6_core.util.instanciate_delayed` this function does not operate inplace.

    :param o: nested structure that may contain DelayedBase objects
    :return:
    """
    if use_buggy_old_instanciate_delayed:
        return _instanciate_delayed_old(o)
    o_ = tree.map_structure(_instanciate_delayed_obj, o)
    if enable_check_behavior_change_new_to_old:
        h = short_hash(o)
        o_copy = tree.map_structure(lambda x: x, o)  # copy, so that we don't modify the original
        assert h == short_hash(o_copy)  # sanity check
        _instanciate_delayed_old(o_copy)
        assert h == short_hash(o)  # sanity check
        if h != short_hash(o_copy):
            print(
                "WARNING: The original behavior of instanciate_delayed modified the input inplace, "
                "and thus the hash changed.\n"
                f"  Object: {o}\n"
                "Call stack (most recent call last):"
            )
            # noinspection PyUnresolvedReferences,PyProtectedMember
            f = sys._getframe()
            while f is not None:
                print(f"  {f.f_code.co_name} in {f.f_code.co_filename}:{f.f_lineno}")
                f = f.f_back
            print(
                "Now the code uses a new instanciate_delayed which does not modify the input inplace. "
                "If you observe that many of your jobs do get a different hash now, "
                "it means you were affected by this change. "
                "Specifically, imagine you have code like this:\n"
                "  train_exp('exp1', task, model_def, config=config)\n"
                "  train_exp('exp2', task, model_def, config=config)  # exactly same as exp1\n"
                "  train_exp('exp3', task, model_def, config=config)  # exactly same as exp1\n"
                "  train_exp('exp4', task, model_def, config=config2)  # different\n"
                "(similar for train_exp, search_dataset, recog_model, recog_training_exp, ...) "
                "I.e. you use exactly the same args: Because instanciate_delayed modified some of the data inplace, "
                "the first call will get a different hash than exp2 and exp3. "
                "Now, with the new behavior, all of exp1, exp2 and exp3 will get the same hash. "
                "But also exp4 might get a new hash than before"
                " (if the instanciate_delayed modification happened on the task itself).\n"
                "If you want to keep the old behavior, call this in the beginning of your Sisyphus config:\n"
                "  set_use_buggy_old_instanciate_delayed(True).\n"
                "(If you want to have a context manager for this, let me know.)\n"
                "If you want to disable this warning, call this in the beginning of your Sisyphus config:\n"
                "  set_enable_check_behavior_change_new_to_old(False)."
            )
    return o_


def _instanciate_delayed_obj(o: Any) -> Any:
    if isinstance(o, DelayedBase):
        return o.get()
    return o


def set_use_buggy_old_instanciate_delayed(b: bool) -> None:
    """
    Set whether to use the old buggy version of instanciate_delayed,
    i.e. :func:`i6_core.util.instanciate_delayed`.

    :param b: True to use the old version, False to use the new version
    """
    global use_buggy_old_instanciate_delayed
    use_buggy_old_instanciate_delayed = b


def set_enable_check_behavior_change_new_to_old(b: bool) -> None:
    """
    Set whether to enable the check for behavior change from the new to the old version of instanciate_delayed.

    :param b: True to enable the check, False to disable it
    """
    global enable_check_behavior_change_new_to_old
    enable_check_behavior_change_new_to_old = b
