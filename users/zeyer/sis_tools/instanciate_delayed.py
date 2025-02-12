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


# Keep this as the default to not break hashes...
use_buggy_old_instanciate_delayed = True


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
    return tree.map_structure(_instanciate_delayed_obj, o)


def print_instanciate_delayed_warning_on_obj(o: Any):
    """
    Earlier, the instanciate_delayed function here did such check itself.
    But that check might be misleading, because every change does change the hash,
    however, the problem is whether it changed some other original object.
    Only the calling code can really test this.
    So we expect the calling code checks for the consistency, and if it detects a problem,
    then it can call this function to print a warning.
    """
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
        "Specifically, imagine you have code like this:\n"
        "  train_exp('exp1', task, model_def, config=config)\n"
        "  train_exp('exp2', task, model_def, config=config)  # exactly same as exp1\n"
        "  train_exp('exp3', task, model_def, config=config)  # exactly same as exp1\n"
        "  train_exp('exp4', task, model_def, config=config2)  # different\n"
        "(similar for train_exp, search_dataset, recog_model, recog_training_exp, ...) "
        "I.e. you use exactly the same args: Because instanciate_delayed modified some of the data inplace, "
        "the first call will get a different hash than exp2 and exp3. "
        " (if the instanciate_delayed modification happened on the task itself).\n"
        "If you want to use the new fixed behavior, call this in the beginning of your Sisyphus config:\n"
        "  set_use_buggy_old_instanciate_delayed(False).\n"
        "(If you want to have a context manager for this, let me know.)\n"
        "Now, with the new fixed behavior, all of exp1, exp2 and exp3 will get the same hash. "
        "However, note that also exp4 might get a new hash than before."
    )


def _instanciate_delayed_obj(o: Any) -> Any:
    if isinstance(o, DelayedBase):
        return o.get()
    return o


def set_use_buggy_old_instanciate_delayed(b: bool) -> None:
    """
    Set whether to use the old buggy version of instanciate_delayed,
    i.e. :func:`i6_core.util.instanciate_delayed`,
    or not.

    Note that this can change the hash of your experiments.
    The hashes should be more correct with the new version.

    :param b: True to use the old version, False to use the new version
    """
    global use_buggy_old_instanciate_delayed
    use_buggy_old_instanciate_delayed = b
