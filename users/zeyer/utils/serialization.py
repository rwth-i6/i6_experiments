"""
Serialization helpers
"""

from __future__ import annotations
import os
import sisyphus
from sisyphus import gs


def get_import_py_code() -> str:
    """
    :return: python code necessary to setup the import paths

    If we use :class:`i6_experiments.common.setups.returnn_common.serialization.Collection`,
    then this is done automatically and not needed.
    If we use :class:`i6_experiments.common.setups.serialization.Collection`,
    then we need to add this manually.
    It usually makes sense to put this into :class:`NonhashedCode`.
    """
    content = ["import os\nimport sys\n"]
    root_path = os.path.join(gs.BASE_DIR, gs.RECIPE_PREFIX)
    content.append(f"sys.path.insert(0, {root_path!r})\n")
    content.append(f"sys.path.insert(1, {os.path.dirname(sisyphus.__path__[0])!r})\n")
    return "".join(content)
