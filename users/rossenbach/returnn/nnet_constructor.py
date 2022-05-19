"""
Experimental approach to `returnn_common.nn` network construction as explicit job with tk.Path/tk.Variable resolving
"""
from dataclasses import dataclass, asdict
from typing import Any, List, Union, Optional, Dict, Callable
import inspect
import os
import pathlib
import shutil
import string
import pprint
import textwrap


from sisyphus import tk, Job, Task
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper
import sys

from i6_core.util import instanciate_delayed, uopen
from i6_core.returnn.config import CodeWrapper

@dataclass(frozen=True)
class DimInitArgs:
    """
    :param name: name of the dim
    :param dim: dimension size (feature axis size or label index count for sparse_dim), None for spatial axis
    """
    name: str
    dim: Optional[Union[int, tk.Variable]]
    is_feature: bool = False

    def _sis_hash(self):
        return asdict(self)


@dataclass(frozen=True)
class DataInitArgs():
    """

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

    def _sis_hash(self):
        return sis_hash_helper(asdict(self))


class ReturnnCommonSerializer(DelayedBase):

    def __init__(self,
                 delayed_objects: List[DelayedBase],
                 returnn_common_root: Optional[tk.Path] = None,
                 packages: Optional[List[Union[str, tk.Path]]]=None,
                 make_local_package_copy=False,
                 ):
        self.delayed_objects = delayed_objects
        self.returnn_common_root = returnn_common_root
        self.packages = packages
        self.make_local_package_copy = make_local_package_copy

        self.root_path = os.path.join(os.getcwd(), "recipe")

    def get(self):
        if self.packages == None:
            # try to collect packages from objects
            self.packages = set()
            for object in self.delayed_objects:
                if isinstance(object, ReturnnCommonImport):
                    self.packages.add(object.package)

        content = "import os\nimport sys\n"
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
                pathlib.Path(target_package_path).mkdir(parents=True, exist_ok=True)
                shutil.copytree(os.path.join(self.root_path, package_path), target_package_path)
                content += f"sys.path.insert(0, os.path.dirname(__file__))\n"
        else:
            content += f"sys.path.insert(0, \"{self.root_path}\")\n"

        if self.returnn_common_root is None:
            content += "from returnn_common import nn\n\n"
        else:
            content += (f"sys.path.insert(0, \"{self.returnn_common_root.get()}/..\")\n"
                        "from returnn_common import nn\n\n"
                       )
        return content + "\n".join([obj.get() for obj in self.delayed_objects])

    def _sis_hash(self):
        h = {
            "delayed_objects": [obj for obj in self.delayed_objects if not isinstance(obj, NonhashedCode)],
            "returnn_common_root": self.returnn_common_root
        }
        return sis_hash_helper(h)


class ReturnnCommonExternData(DelayedBase):

    def __init__(self, extern_data: List[DataInitArgs]):
        self.extern_data = extern_data

    def create_data(self, constructor_data: DataInitArgs):
        """
        convert NetConstructorData into an actual

        :param constructor_data:
        :return:
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

    def get(self):
        content = ""

        # collect dims
        dims = set()
        for constructor_data in self.extern_data:
            for dim in constructor_data.dim_tags:
                dims.add(dim)
            if constructor_data.sparse_dim:
                dims.add(constructor_data.sparse_dim)
        for dim in dims:
            content += f"{dim.name} = nn.{'FeatureDim' if dim.is_feature else 'SpatialDim'}(\"{dim.name}\", {instanciate_delayed(dim.dim)})\n"

        for constructor_data in self.extern_data:
            content += self.create_data(constructor_data)

        for constructor_data in self.extern_data:
            content += f"{constructor_data.name}_args = {constructor_data.name}.get_kwargs()\n"
            content += f"{constructor_data.name}_args.pop(\"name\")\n"

        content += "\nextern_data={\n"
        for constructor_data in self.extern_data:
            content += f"    \"{constructor_data.name}\": {constructor_data.name}_args,\n"
        content += "}\n"
        return content


class ReturnnCommonImport(DelayedBase):

    def __init__(self,
                 code_object: str,
                ):
        """

        :param module:
        :param packages: None uses
        """
        super().__init__(None)
        self.code_object = code_object

        self.model_name = self.code_object.split(".")[-1]
        self.module = ".".join(self.code_object.split(".")[:-1])
        self.package = ".".join(self.code_object.split(".")[:-2])

    def get(self):
        # this is run in the task!
        return f"from {self.module} import {self.model_name}\n"

    def _sis_hash(self):
        return sis_hash_helper(self.code_object)



class ReturnnCommonDynamicNetwork(DelayedBase):
    """

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

        :param returnn_common_root:
        :param network_file:
        :param parameter_dict:
        """
        super().__init__(None)
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


class NonhashedCode(DelayedBase):
    """
    Insert code as string or from file without any hashing
    """

    def __init__(self, code: Union[str, tk.Path]):
        self.code = code

    def get(self):
        if isinstance(self.code, tk.Path):
            with uopen(self.code, "rt") as f:
                return f.read()
        else:
            return self.code


class CodeFromFile(DelayedBase):
    """
    Insert code from a file hashed by file path/name or full content
    """

    def __init__(self, code: tk.Path, hash_full_content=False):
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



