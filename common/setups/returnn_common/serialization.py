"""
Contains code that is necessary/helpful to generate RETURNN configs that make use of returnn_common

Usage Example::

    from i6_experiments.common.setups.returnn_common import serialization

    rc_some_unhashed_code = serialization.NonhashedCode(code=some_code)
    rc_extern_data = serialization.ExternData(extern_data=extern_data)
    rc_model = serialization.Import(
        "i6_experiments.users.some_user.common_modules.some_module.SomeNnModuleModel")
    rc_construction_code = serialization.Import(
        "i6_experiments.users.some_user.common_modules.constructor_module.some_constructor_function")

    rc_network = serialization.Network(
        net_func_name=rc_construction_code.object_name,
        net_func_map={
          "net_module": rc_model.object_name,
          "audio_data": "audio_features",
          "label_data": "bpe_labels",
          "audio_feature_dim": "audio_features_feature",
          "audio_time_dim": "audio_features_time",
          "label_time_dim": "bpe_labels_time",
          "label_dim": "bpe_labels_indices"
        },
        net_kwargs={... some additional constructor function args ...}
    )

    serializer = serialization.Collection(
        serializer_objects=[
          rc_recursionlimit,
          rc_extern_data,
          rc_model,
          rc_construction_code,
          rc_network],
        returnn_common_root=returnn_common_root,
    )
    returnn_config = ReturnnConfig(
        config=config,
        post_config=post_config,
        python_epilog=[serializer],
    )
"""

from dataclasses import dataclass, asdict
from typing import Any, List, Union, Optional, Dict, Set
from types import FunctionType
import os
import sys
import pathlib
import shutil
import string
import textwrap

import sisyphus
from sisyphus import tk, gs
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper

from i6_core.util import instanciate_delayed, uopen
from i6_core.returnn.config import CodeWrapper


class SerializerObject(DelayedBase):
    """
    Base class for objects that can be passed to :class:`Collection`
    """

    use_for_hash = True

    def __init__(self):
        # suppress init warning
        super().__init__(None)


@dataclass(frozen=True)
class DimInitArgs:
    """
    A helper class to store input args for a nn.Dim object

    :param name: name of the dim
    :param dim: dimension size (feature axis size or label index count for sparse_dim),
                None for dynamic (eg. spatial) axes
    :param is_feature: If the dim is a feature dim and not a spatial dim.
    """

    name: str
    dim: Optional[Union[int, tk.Variable]]
    is_feature: bool = False


@dataclass(frozen=True)
class DataInitArgs:
    """
    A helper class to store needed input args for a nn.Data object without `returnn_common` dependency

    :param name: name of the data (equivalent to the extern_data entry)
    :param available_for_inference: if this data is available during decoding/forward pass etc...
    :param dim_tags: list of dim tags representing an axis of the data, without batch or a hidden sparse dim
    :param sparse_dim: provide this dim to make the data sparse and define the index size:
    :param dtype: dtype of the data, usually float32
    """

    name: str
    available_for_inference: bool
    dim_tags: List[DimInitArgs]
    sparse_dim: Optional[DimInitArgs]
    dtype: str = "float32"

    def _sis_hash(self) -> bytes:
        # INFO: asdict is recursive, so DimInitArgs will be converted as well
        return sis_hash_helper(asdict(self))


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
        returnn_common_root: Optional[tk.Path] = None,
        packages: Optional[Set[Union[str, tk.Path]]] = None,
        make_local_package_copy: bool = False,
    ):
        """

        :param serializer_objects: all serializer objects which are serialized into a string in order
        :param returnn_common_root: Path to returnn_common, if None, assumes direct import is fine
        :param packages: Path to packages to import, if None, tries to extract them from serializer_objects
        :param make_local_package_copy: whether to make a local copy of imported code into the Job directory
        """
        super().__init__(None)
        self.serializer_objects = serializer_objects
        self.returnn_common_root = returnn_common_root
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
                pathlib.Path(os.path.dirname(target_package_path)).mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copytree(
                    os.path.join(self.root_path, package_path), target_package_path
                )
                content.append(f"sys.path.insert(0, os.path.dirname(__file__))\n")
        else:
            content.append(f"sys.path.insert(0, {self.root_path!r})\n")

        # Make sure Sisyphus can be imported, as many recipes usually import it.
        content.append(
            f"sys.path.insert(1, {os.path.dirname(sisyphus.__path__[0])!r})\n"
        )

        if self.returnn_common_root is None:
            # Note that this here depends on a proper sys.path setup.
            content.append("from returnn_common import nn\n\n")
        else:
            if self.make_local_package_copy:
                assert f"/{gs.RECIPE_PREFIX}" not in self.returnn_common_root.get(), (
                    "please do not use returnn_common from your recipe folder "
                    "when using `make_local_package_copy=True`, as then the local copy will not be used"
                )
                # TODO: maybe find a workaround for this problem?  Somehow python ignores the sys.path priority
                # order here and always chooses the package from recipe/ first...
            content.append(
                f'sys.path.insert(0, "{self.returnn_common_root.get()}/..")\n'
                "from returnn_common import nn\n\n"
            )

        content += [obj.get() for obj in self.serializer_objects]
        return "".join(content)

    def _sis_hash(self) -> bytes:
        h = {
            "delayed_objects": [
                obj for obj in self.serializer_objects if obj.use_for_hash
            ],
        }
        if self.returnn_common_root:
            h["returnn_common_root"] = self.returnn_common_root
        return sis_hash_helper(h)


class ExternData(SerializerObject):
    """
    Write nn.Dim, nn.Data and extern_data definitions as string into a config, using DataInitArgs and DimInitArgs
    as definition helpers.
    """

    def __init__(self, extern_data: List[DataInitArgs]):
        """
        :param extern_data: A DataInitArgs object for each extern_data entry that should be available in the config
        """
        super().__init__()
        self.extern_data = extern_data

    @staticmethod
    def _serialize_data(data: DataInitArgs) -> List[str]:
        """
        Serialize a single DataInitArgs object

        :param data: init args for serialization
        """
        content = [
            f"{data.name} = nn.Data(\n",
            f'    name="{data.name}",',
            f"    dim_tags=[nn.batch_dim, {', '.join(dim.name for dim in data.dim_tags)}],",
        ]
        if data.sparse_dim is not None:
            content.append(f"    sparse_dim={data.sparse_dim.name},")
        content.append(f"    available_for_inference={data.available_for_inference},")
        if (
            data.dtype != "float32"
        ):  # RETURNN default is float32 so we only append it otherwise
            content.append(f'    dtype="{data.dtype}",')
        content.append(")\n")
        return content

    def get(self) -> str:
        """get"""
        content = []

        # collect dims into a set to only write each Dim once if shared
        dims = {}  # set but deterministic insertion order
        for constructor_data in self.extern_data:
            for dim in constructor_data.dim_tags:
                dims[dim] = None
            if constructor_data.sparse_dim:
                dims[constructor_data.sparse_dim] = None

        for dim in dims.keys():
            content.append(
                f"{dim.name} = nn.{'FeatureDim' if dim.is_feature else 'SpatialDim'}({dim.name!r}, "
                f"{instanciate_delayed(dim.dim)})\n"
            )

        for constructor_data in self.extern_data:
            content += self._serialize_data(constructor_data)

        # RETURNN does not allow for "name" in the args, as this is set via the dict key
        # thus, we need to explicitly remove it for now
        for constructor_data in self.extern_data:
            content.append(
                f"{constructor_data.name}_args = {constructor_data.name}.get_kwargs()\n"
            )
            content.append(f'{constructor_data.name}_args.pop("name")\n')

        content.append("\nextern_data={\n")
        for constructor_data in self.extern_data:
            content.append(
                f'    "{constructor_data.name}": {constructor_data.name}_args,\n'
            )
        content.append("}\n")
        return "".join(content)


class Import(SerializerObject):
    """
    A class to indicate a module or function that should be imported within the returnn config

    When passed to the ReturnnCommonSerializer it will automatically detect the local package in case of
    `make_local_package_copy=True`, unless specific package paths are given.
    """

    def __init__(
        self,
        code_object_path: Union[str, FunctionType, Any],
        import_as: Optional[str] = None,
        *,
        use_for_hash: bool = True,
        ignore_import_as_for_hash: bool = False,
    ):
        """
        :param code_object_path: e.g. `i6_experiments.users.username.my_rc_files.SomeNiceASRModel`.
            This can be the object itself, e.g. a function or a class. Then it will use __qualname__ and __module__.
        :param import_as: if given, the code object will be imported as this name
        :param use_for_hash:
        """
        super().__init__()
        if not isinstance(code_object_path, str):
            assert getattr(code_object_path, "__qualname__", None) and getattr(
                code_object_path, "__module__", None
            )
            mod_name = code_object_path.__module__
            qual_name = code_object_path.__qualname__
            assert "." not in qual_name
            assert getattr(sys.modules[mod_name], qual_name) is code_object_path
            code_object_path = f"{mod_name}.{qual_name}"
        self.code_object = code_object_path

        self.object_name = self.code_object.split(".")[-1]
        self.module = ".".join(self.code_object.split(".")[:-1])
        self.package = ".".join(self.code_object.split(".")[:-2])
        self.import_as = import_as
        self.use_for_hash = use_for_hash
        self.ignore_import_as_for_hash = ignore_import_as_for_hash

    def get(self) -> str:
        """get. this code is run in the task"""
        if self.import_as:
            return f"from {self.module} import {self.object_name} as {self.import_as}\n"
        return f"from {self.module} import {self.object_name}\n"

    def _sis_hash(self):
        if self.import_as and not self.ignore_import_as_for_hash:
            return sis_hash_helper(
                {"code_object": self.code_object, "import_as": self.import_as}
            )
        return sis_hash_helper(self.code_object)


class Network(SerializerObject):
    """
    Serializes a `get_network` function into the config, which calls
    a defined network construction function and defines the parameters to it.

    Note that the network constructor function always needs "epoch" as first defined parameter,
    and should return an `nn.Module` object.
    """

    TEMPLATE = textwrap.dedent(
        """\
        
        network_kwargs = ${NETWORK_KWARGS}
        
        def get_network(epoch, **kwargs):
            nn.reset_default_root_name_ctx()
            net = ${FUNCTION_NAME}(epoch=epoch, **network_kwargs)
            return nn.get_returnn_config().get_net_dict_raw_dict(net)
        
        """
    )

    def __init__(
        self,
        net_func_name: str,
        net_func_map: Dict[str, str],
        net_kwargs: Dict[str, Any],
    ):
        """

        :param net_func_name: name of the network construction function to be imported
            This should as default be set to `ReturnnCommonImport("my_package.my_function").object_name`
            Use the `epoch` parameter for dynamic network definition.
        :param net_func_map: A mapping to define which config objects should be linked to which function parameters
            This can for example be for a network module definition:
            `rc_module = ReturnnCommonImport("my_package.my_net_module")`
            `'net_module': rc_module.object_name`
            Or for a known nn.Data object named `bpe_labels` and a constructor parameter named `label_data` just:
            `label_data`: `bpe_labels`
            The `value` objects will be written via `CodeWrapper` to directly refer to the serialized objects within
            the returnn config file.
        :param net_kwargs: A dict containing any additional (hyper)-parameter kwargs that should be passed to the
            constructor function
        """
        super().__init__()
        self.net_func_name = net_func_name
        self.net_kwargs = net_kwargs
        self.net_kwargs.update({k: CodeWrapper(v) for k, v in net_func_map.items()})

    def get(self):
        """get"""
        return string.Template(self.TEMPLATE).substitute(
            {
                "NETWORK_KWARGS": str(self.net_kwargs),
                "FUNCTION_NAME": self.net_func_name,
            }
        )

    def _sis_hash(self):
        h = {
            "net_kwargs": self.net_kwargs,
        }
        return sis_hash_helper(h)


class _NonhashedSerializerObject(SerializerObject):
    """
    Any serializer object which is not used for the hash.
    """

    use_for_hash = False

    def _sis_hash(self):
        raise Exception(f"({self.__class__.__name__}) must not be hashed")


class NonhashedCode(_NonhashedSerializerObject):
    """
    Insert code from raw string which is not hashed.
    """

    def __init__(self, code: Union[str, tk.Path]):
        super().__init__()
        self.code = code

    def get(self):
        """get"""
        return self.code


class NonhashedCodeFromFile(_NonhashedSerializerObject):
    """
    Insert code from file content which is not hashed (neither the file name nor the content).
    """

    def __init__(self, filename: tk.Path):
        super().__init__()
        self.filename = filename

    def get(self):
        """get"""
        with uopen(self.filename, "rt") as f:
            return f.read()


class CodeFromFile(SerializerObject):
    """
    Insert code from a file hashed by file path/name or full content
    """

    def __init__(self, filename: tk.Path, hash_full_content: bool = False):
        """
        :param filename:
        :param hash_full_content: False -> hash filename, True -> hash content (but not filename)
        """
        super().__init__()
        self.filename = filename
        self.hash_full_content = hash_full_content

    def get(self):
        """get"""
        with uopen(self.filename, "rt") as f:
            return f.read()

    def _sis_hash(self):
        if self.hash_full_content:
            with uopen(self.filename, "rt") as f:
                return sis_hash_helper(f.read())
        else:
            return sis_hash_helper(self.filename)


class ExplicitHash(SerializerObject):
    """
    Inserts nothing, but uses the given object for hashing
    """

    # noinspection PyShadowingBuiltins
    def __init__(self, hash: Any):
        super().__init__()
        self.hash = hash

    def get(self) -> str:
        """get"""
        return ""

    def _sis_hash(self):
        return sis_hash_helper(self.hash)


PythonEnlargeStackWorkaroundNonhashedCode = NonhashedCode(
    textwrap.dedent(
        """\
        # https://github.com/rwth-i6/returnn/issues/957
        # https://stackoverflow.com/a/16248113/133374
        import resource
        import sys
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
        except Exception as exc:
            print(f"resource.setrlimit {type(exc).__name__}: {exc}")
        sys.setrecursionlimit(10 ** 6)
        """
    )
)


PythonCacheManagerFunctionNonhashedCode = NonhashedCode(
    textwrap.dedent(
        """\
        _cf_cache = {}
        
        def cf(filename):
            "Cache manager"
            from subprocess import check_output, CalledProcessError
            if filename in _cf_cache:
                return _cf_cache[filename]
            if int(os.environ.get("RETURNN_DEBUG", "0")):
                print("use local file: %s" % filename)
                return filename  # for debugging
            try:
                cached_fn = check_output(["cf", filename]).strip().decode("utf8")
            except CalledProcessError:
                print("Cache manager: Error occurred, using local file")
                return filename
            assert os.path.exists(cached_fn)
            _cf_cache[filename] = cached_fn
            return cached_fn
        """
    )
)


# Modelines should be at the beginning or end of a file.
# Many editors (e.g. VSCode) read those information.
PythonModelineNonhashedCode = NonhashedCode("# -*- mode: python; tab-width: 4 -*-\n")
