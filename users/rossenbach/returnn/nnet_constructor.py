"""
Define all code that is necessary/helpful to generate Returnn Configs that make use of returnn_common
"""
from dataclasses import dataclass, asdict
from typing import Any, List, Union, Optional, Dict
import os
import pathlib
import shutil
import string
import textwrap

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper

from i6_core.util import instanciate_delayed, uopen
from i6_core.returnn.config import CodeWrapper


class SerializerObject(DelayedBase):
    """
    Any class that can be passed to ReturnnCommonSerializer
    """
    def __init__(self):
        # suppress init warning
        super().__init__(None)


@dataclass(frozen=True)
class DimInitArgs:
    """
    A helper class to store input args for a nn.Dim object

    :param name: name of the dim
    :param dim: dimension size (feature axis size or label index count for sparse_dim), None for spatial axis
    """
    name: str
    dim: Optional[Union[int, tk.Variable]]
    is_feature: bool = False


@dataclass(frozen=True)
class DataInitArgs():
    """
    A helper class to store needed input args for a nn.Data object

    :param name: name of the data (equivalent to the extern_data entry)
    :param available_for_inference: if this data is available during decoding/forward pass etc...
    :param dim_tags: list of dim tags representing an axis of the data, without batch or a hidden sparse dim
    :param sparse_dim: provide this dim to make the data sparse and define
    :param has_batch_dim: set to False do create data without a batch dim (probably never needed)
    """
    name: str
    available_for_inference: bool
    dim_tags: List[DimInitArgs]
    sparse_dim: Optional[DimInitArgs]
    has_batch_dim: bool = True

    def __post_init__(self):
        if self.sparse_dim is not None:
            assert self.sparse_dim.dim is not None, "A sparse dim can not have 'None' as dimension"
            assert self.sparse_dim.is_feature is True, "A sparse dim should be a feature dim"

    def _sis_hash(self) -> bytes:
        # INFO: asdict is recursive, so DimInitArgs will be converted as well
        return sis_hash_helper(asdict(self))


class ReturnnCommonSerializer(DelayedBase):
    """
    A helper class to serialize a Returnn config with returnn_common elements.
    Should be passed to either `returnn_prolog` or `returnn_epilog` in a `ReturnnConfig`

    The tasks are:
     - managing the returnn_common version (via sys.path)
     - managing returnn_common net definitions (via importing a nn.Module class definition, using `ReturnnCommonImport`)
     - managing returnn_common net construction which returns the final net, using `ReturnnCommonImport`
       (via importing a (epoch, **kwargs) -> nn.Module function)
     - managing nn.Dim/Data and the extern_data entry via `ReturnnCommonExternData`, `DimInitArgs` and `DataInitArgs`
     - managing the package imports from which all imports can be found
     - optionally make a local copy of all imported code instead if importing it directly out of the recipes

    """

    def __init__(self,
                 serializer_objects: List[SerializerObject],
                 returnn_common_root: Optional[tk.Path] = None,
                 packages: Optional[List[Union[str, tk.Path]]]=None,
                 make_local_package_copy=False,
                 ):
        """

        :param serializer_objects: all serializer objects which are serialized into a string  in order
        :param returnn_common_root:
        :param packages:
        :param make_local_package_copy:
        """
        super().__init__(None)
        self.serializer_objects = serializer_objects
        self.returnn_common_root = returnn_common_root
        self.packages = packages
        self.make_local_package_copy = make_local_package_copy

        self.root_path = os.path.join(os.getcwd(), "recipe")

    def get(self) -> str:
        if self.packages == None:
            # try to collect packages from objects
            self.packages = set()
            for object in self.serializer_objects:
                if isinstance(object, ReturnnCommonImport):
                    self.packages.add(object._package)

        content = "import os\nimport sys\n"

        if self.returnn_common_root is None:
            content += "from returnn_common import nn\n\n"
        else:
            if self.make_local_package_copy:
                assert "/recipe" not in self.returnn_common_root.get(), (
                    "please do not use returnn_common from your recipe folder "
                    "when using `make_local_package_copy=True`, as then the local copy will not be used"
                )
                # TODO: maybe find a workaround for this problem?  Somehow python ignores the sys.path priority
                # order here and always chooses the package from recipe/ first...
            content += (f"sys.path.insert(0, \"{self.returnn_common_root.get()}/..\")\n"
                        "from returnn_common import nn\n\n"
                        )

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
                content += f"sys.path.insert(0, os.path.dirname(__file__))\n"
        else:
            content += f"sys.path.insert(0, \"{self.root_path}\")\n"

        return content + "\n".join([obj.get() for obj in self.serializer_objects])

    def _sis_hash(self) -> bytes:
        h = {
            "delayed_objects": [obj for obj in self.serializer_objects if not isinstance(obj, NonhashedCode)],
            "returnn_common_root": self.returnn_common_root
        }
        return sis_hash_helper(h)


class ReturnnCommonExternData(SerializerObject):
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

    def serialize_data(self, constructor_data: DataInitArgs) -> str:
        """
        Serialize a single DataInitArgs object

        :param constructor_data: init args for serialization
        """
        content = ""
        wrapped_dim_names = []
        for c_dim in constructor_data.dim_tags:
            wrapped_dim_names.append(CodeWrapper(c_dim.name))

        if constructor_data.has_batch_dim:
            wrapped_dim_names = [CodeWrapper("nn.batch_dim")] + wrapped_dim_names

        if constructor_data.sparse_dim is not None:
            sparse_dim = constructor_data.sparse_dim
            sparse_dim_name = sparse_dim.name
        else:
            sparse_dim_name = None

        content += textwrap.dedent(f"""\
        {constructor_data.name} = nn.Data(
            name="{constructor_data.name}",
            available_for_inference={constructor_data.available_for_inference},
            dim_tags={wrapped_dim_names},
            sparse_dim={sparse_dim_name},
        )
        """)

        return content

    def get(self) -> str:
        content = ""

        # collect dims into a set to only write each Dim once if shared
        dims = set()
        for constructor_data in self.extern_data:
            for dim in constructor_data.dim_tags:
                dims.add(dim)
            if constructor_data.sparse_dim:
                dims.add(constructor_data.sparse_dim)

        for dim in dims:
            content += f"{dim.name} = nn.{'FeatureDim' if dim.is_feature else 'SpatialDim'}(\"{dim.name}\", "\
                       f"{instanciate_delayed(dim.dim)})\n"

        for constructor_data in self.extern_data:
            content += self.serialize_data(constructor_data)

        # RETURNN does not allow for "name" in the args, as this is set via the dict key
        # thus, we need to explicitely remove it for now
        for constructor_data in self.extern_data:
            content += f"{constructor_data.name}_args = {constructor_data.name}.get_kwargs()\n"
            content += f"{constructor_data.name}_args.pop(\"name\")\n"

        content += "\nextern_data={\n"
        for constructor_data in self.extern_data:
            content += f"    \"{constructor_data.name}\": {constructor_data.name}_args,\n"
        content += "}\n"
        return content


class ReturnnCommonImport(SerializerObject):
    """
    A class to indicate a module or function that should be imported within the returnn config

    When passed to the ReturnnCommonSerializer it will automatically detect the local package in case of
    `make_local_package_copy=True`, unless specific package paths are given.
    """
    def __init__(self,
                 code_object: str,
                ):
        """

        :param code_object: path to a python object, e.g. `i6_experiments.users.username.my_rc_files.SomeNiceASRModel`
        """
        super().__init__()
        self.code_object = code_object

        self._object_name = self.code_object.split(".")[-1]
        self._module = ".".join(self.code_object.split(".")[:-1])
        self._package = ".".join(self.code_object.split(".")[:-2])

    def get(self):
        # this is run in the task!
        return f"from {self._module} import {self._object_name}\n"

    def get_name(self):
        return self._object_name

    def _sis_hash(self):
        return sis_hash_helper(self.code_object)


class ReturnnCommonDynamicNetwork(SerializerObject):
    """
    Serializes a `get_network` function into the config, which calls
    a defined network construction function and defines the parameters to it.

    Note that the network constructor function always needs "epoch" as first defined parameter,
    and should return an `nn.Module` object.
    """
    TEMPLATE = textwrap.dedent("""\
        
        network_kwargs = ${NETWORK_KWARGS}
        
        def get_network(epoch, **kwargs):
            nn.reset_default_root_name_ctx()
            net = ${FUNCTION_NAME}(epoch, **network_kwargs)
            return nn.get_returnn_config().get_net_dict_raw_dict(net)
        
        """)

    def __init__(self,
                 net_func_name: str,
                 net_func_map: Dict[str, str],
                 net_kwargs: Dict[str, Any],
                 ):
        """

        :param net_func_name: name of the network construction function to be imported
            This should as default be set to `ReturnnCommonImport("my_package.my_function").get_name()`
        :param net_func_map: A mapping to define which config objects should be linked to which function parameters
            This can for example be for a network module definition:
            `rc_module = ReturnnCommonImport("my_package.my_net_module")`
            `'net_module': rc_module.get_name()`
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
        self.net_kwargs.update(
            {k: CodeWrapper(v) for k,v in net_func_map.items()}
        )

    def get(self):
        return string.Template(self.TEMPLATE).substitute(
            {
                'NETWORK_KWARGS': str(self.net_kwargs),
                'FUNCTION_NAME': self.net_func_name,

            }
        )

    def _sis_hash(self):
        h = {
            "net_kwargs": self.net_kwargs,
        }
        return sis_hash_helper(h)


class NonhashedCode(SerializerObject):
    """
    Insert code as string or from file without any hashing
    """

    def __init__(self, code: Union[str, tk.Path]):
        super().__init__()
        self.code = code

    def get(self):
        if isinstance(self.code, tk.Path):
            with uopen(self.code, "rt") as f:
                return f.read()
        else:
            return self.code


class CodeFromFile(SerializerObject):
    """
    Insert code from a file hashed by file path/name or full content
    """

    def __init__(self, code: tk.Path, hash_full_content=False):
        super().__init__()
        self.code = code
        self.hash_full_content = hash_full_content

    def get(self):
        with uopen(self.code, "rt") as f:
            return f.read()

    def _sis_hash(self):
        if self.hash_full_content:
            with uopen(self.code, "rt") as f:
                return sis_hash_helper(f.read())
        else:
            return sis_hash_helper(self.code)



