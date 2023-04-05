from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Union
import os
import pathlib
import shutil
import string
import textwrap

from sisyphus import gs, tk
from sisyphus.hash import sis_hash_helper
from sisyphus.delayed_ops import DelayedBase

from ..serialization import SerializerObject


class PyTorchModel(SerializerObject):
    """
    Serializes a `get_network` function into the config, which calls
    a defined network construction function and defines the parameters to it.
    This is for returnn_common networks.

    Note that the network constructor function always needs "epoch" as first defined parameter,
    and should return an `nn.Module` object.
    """

    TEMPLATE = textwrap.dedent(
        """\

    model_kwargs = ${MODEL_KWARGS}

    def get_model(**kwargs):
        return ${MODEL_CLASS}(**model_kwargs)

    """
    )

    def __init__(
        self,
        model_class_name: str,
        model_kwargs: Dict[str, Any],
    ):
        """
        :param model_class_name:
        :param model_kwargs:
        """

        super().__init__()
        self.model_class_name = model_class_name
        self.model_kwargs = model_kwargs

    def get(self):
        """get"""
        return string.Template(self.TEMPLATE).substitute(
            {
                "MODEL_KWARGS": str(self.model_kwargs),
                "MODEL_CLASS": self.model_class_name,
            }
        )

    def _sis_hash(self):
        h = {
            "model_kwargs": self.model_kwargs,
        }
        return sis_hash_helper(h)


class Collection(DelayedBase):
    """
    A helper class to serialize a RETURNN config with returnn_common elements.
    Should be passed to either `returnn_prolog` or `returnn_epilog` in a `ReturnnConfig`

    The tasks are:
     - managing the returnn_common version (via sys.path)
     - managing returnn_common net definitions (via importing a nn.Module class definition, using :class:`Import`)
     - managing returnn_common net construction which returns the final net, using `ReturnnCommonImport`
       (via importing a (epoch, **kwargs) -> nn.Module function)
     - managing nn.Dim/Data and the extern_data entry
       via :class:`ExternData`, :class:`DimInitArgs` and :class:`DataInitArgs`
     - managing the package imports from which all imports can be found
     - optionally make a local copy of all imported code instead if importing it directly out of the recipes

    """

    def __init__(
        self,
        serializer_objects: List[SerializerObject],
        *,
        packages: Optional[Set[Union[str, tk.Path]]] = None,
        make_local_package_copy: bool = False,
    ):
        """

        :param serializer_objects: all serializer objects which are serialized into a string in order
        :param packages: Path to packages to import, if None, tries to extract them from serializer_objects
        :param make_local_package_copy: whether to make a local copy of imported code into the Job directory
        """
        super().__init__(None)
        self.serializer_objects = serializer_objects
        self.packages = packages
        self.make_local_package_copy = make_local_package_copy

        self.root_path = os.path.join(gs.BASE_DIR, gs.RECIPE_PREFIX)

        assert (not self.make_local_package_copy) or self.packages, (
            "Please specify which packages to copy if you are using "
            "`make_local_package_copy=True` in combination with `Import` objects"
        )

    def get(self) -> str:
        """get"""
        content = ["import os\nimport sys\n"]

        # have sys.path setup first
        if self.make_local_package_copy:
            out_dir = os.path.join(os.getcwd(), "../output")
            for package in self.packages:
                if isinstance(package, tk.Path):
                    package_path = package.get_path()
                elif isinstance(package, str):
                    package_path = package.replace(".", "/")
                else:
                    assert False, "invalid type for packages"
                target_package_path = os.path.join(out_dir, package_path)
                pathlib.Path(os.path.dirname(target_package_path)).mkdir(parents=True, exist_ok=True)
                shutil.copytree(os.path.join(self.root_path, package_path), target_package_path)
                content.append(f"sys.path.insert(0, os.path.dirname(__file__))\n")
        else:
            content.append(f"sys.path.insert(0, {self.root_path!r})\n")

        content += [obj.get() for obj in self.serializer_objects]
        return "".join(content)

    def _sis_hash(self) -> bytes:
        h = {
            "delayed_objects": [obj for obj in self.serializer_objects if obj.use_for_hash],
        }
        return sis_hash_helper(h)
