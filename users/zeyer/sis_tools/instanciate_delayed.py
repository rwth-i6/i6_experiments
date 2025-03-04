"""
:func:`i6_core.util.instanciate_delayed` but not operating inplace,
which can potentially be dangerous and lead to bugs which are hard to track down
and which are not immediately present.
"""

from typing import Any, Callable
import sys
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import short_hash
from i6_core.util import instanciate_delayed as instanciate_delayed_inplace


def instanciate_delayed_copy(o: Any) -> Any:
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    In contrast to :func:`i6_core.util.instanciate_delayed` this function does not operate inplace.

    :param o: nested structure that may contain DelayedBase objects
    :return: o with all DelayedBase objects replaced by their .get() value
    """
    import tree

    def _instanciate_delayed_obj(o: Any) -> Any:
        if isinstance(o, DelayedBase):
            return o.get()
        return o

    return tree.map_structure(_instanciate_delayed_obj, o)


def instanciate_delayed_inplace_with_warning(f: Callable[[], Any]) -> Any:
    """
    Recursively traverses a structure and calls .get() on all
    existing Delayed Operations, especially Variables in the structure

    This uses :func:`i6_core.util.instanciate_delayed`, i.e. it operates inplace!

    Additionally, afterwards, we check whether the hash of the object changed when calling f() again.
    If so, it means that the inplace operation changed something inside the source object (e.g. DatasetConfigStatic).

    :param f: function which returns nested structure that may contain DelayedBase objects
    :return: f() with all DelayedBase objects replaced by their .get() value
    """
    o = f()
    h = short_hash(o)
    o_ = instanciate_delayed_inplace(o)
    if h != short_hash(f()):
        print_instanciate_delayed_warning(obj=o, func=f)
    return o_


_not_specified = object()


def print_instanciate_delayed_warning(*, obj: Any = _not_specified, func: Any = None):
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
        "and thus the hash changed."
    )
    if func:
        print(f"Function: {func}")
    if obj is not _not_specified:
        print(f"Object: {obj}")
    print("Call stack (most recent call last):")
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
        "I.e. you use exactly the same args: "
        "Because instanciate_delayed modified some of the data inplace, "
        "the first call will get a different hash than exp2 and exp3 "
        "(if the instanciate_delayed modification happened on the task itself).\n"
        f"If you want to fix this, make sure that the function {func} returns a deep copy. "
        "Or use instanciate_delayed_copy instead of the inplace instanciate_delayed.\n"
        "With fixed behavior, all of exp1, exp2 and exp3 will get the same hash. "
        "However, note that also exp4 might get a new hash than before."
    )
