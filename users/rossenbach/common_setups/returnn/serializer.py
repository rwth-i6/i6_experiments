"""
Serializer Prototypes
"""
import string
import sys
import textwrap
from typing import Any, Dict, Optional, Union

from sisyphus.hash import sis_hash_helper

from i6_core.util import instanciate_delayed

from i6_experiments.common.setups.serialization import SerializerObject, FunctionType



class Import(SerializerObject):
    """
    A class to indicate a module or function that should be imported within the returnn config

    When passed to the ReturnnCommonSerializer it will automatically detect the local package in case of
    `make_local_package_copy=True`, unless specific package paths are given.

    For imports from external libraries, e.g. git repositories use "ExternalImport".
    """

    def __init__(
        self,
        *,
        code_object_path: Union[str, FunctionType, Any],
        unhashed_package_root: str,
        import_as: Optional[str] = None,
        use_for_hash: bool = True,
        ignore_import_as_for_hash: bool = False,
    ):
        """
        :param code_object_path: e.g. `i6_experiments.users.username.my_rc_files.SomeNiceASRModel`.
            This can be the object itself, e.g. a function or a class. Then it will use __qualname__ and __module__.
        :param unhashed_package_root: The root path to a package, from where relatives paths will be hashed.
            Recommended is to use the root folder of an experiment module.
        :param import_as: if given, the code object will be imported as this name
        :param use_for_hash: if false, this module is not hashed when passed to a Collection/Serializer
        :Param ignore_import_as_for_hash: do not hash `import_as` if set
        """
        super().__init__()
        if not isinstance(code_object_path, str):
            assert getattr(code_object_path, "__qualname__", None) and getattr(code_object_path, "__module__", None)
            mod_name = code_object_path.__module__
            qual_name = code_object_path.__qualname__
            assert "." not in qual_name
            assert getattr(sys.modules[mod_name], qual_name) is code_object_path
            code_object_path = f"{mod_name}.{qual_name}"
        self.code_object = code_object_path

        self.object_name = self.code_object.split(".")[-1]
        self.module = ".".join(self.code_object.split(".")[:-1])

        if len(unhashed_package_root) > 0:
            if not self.code_object.startswith(unhashed_package_root):
                raise ValueError(
                    f"unhashed_package_root: {unhashed_package_root} is not a prefix of {self.code_object}"
                )
            self.code_object = self.code_object[len(unhashed_package_root):]

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
            return sis_hash_helper({"code_object": self.code_object, "import_as": self.import_as})
        return sis_hash_helper(self.code_object)


class PartialImport(Import):
    """
    Like Import, but for partial callables where certain parameters are given fixed and are hashed.
    """
    
    TEMPLATE = textwrap.dedent(
    """\
        import functools
        kwargs = ${KWARGS}
        
        from ${IMPORT_PATH} import ${IMPORT_NAME} as _${IMPORT_NAME}
        ${OBJECT_NAME} = functools.partial(_${IMPORT_NAME}, **kwargs)
        
    """
    )

    def __init__(
        self,
        *,
        code_object_path: Union[str, FunctionType, Any],
        unhashed_package_root: str,
        hashed_arguments: Dict[str, Any],
        unhashed_arguments: Dict[str, Any],
        import_as: Optional[str] = None,
        use_for_hash: bool = True,
        ignore_import_as_for_hash: bool = False,
    ):
        """
        :param code_object_path: e.g. `i6_experiments.users.username.my_rc_files.SomeNiceASRModel`.
            This can be the object itself, e.g. a function or a class. Then it will use __qualname__ and __module__.
        :param unhashed_package_root: The root path to a package, from where relatives paths will be hashed.
            Recommended is to use the root folder of an experiment module.
        :param hashed_arguments: argument dictionary for addition partial arguments to set to the callable.
            Will be serialized as dict into the config, so make sure to use only serializable/parseable content
        :param unhashed_arguments: same as above, but does not influence the hash
        :param import_as: if given, the code object will be imported as this name
        :param use_for_hash: if false, this module is not hashed when passed to a Collection/Serializer
        :param ignore_import_as_for_hash: do not hash `import_as` if set
        """

        super().__init__(
            code_object_path=code_object_path,
            unhashed_package_root=unhashed_package_root,
            import_as=import_as,
            use_for_hash=use_for_hash,
            ignore_import_as_for_hash=ignore_import_as_for_hash
        )
        self.hashed_arguments = hashed_arguments
        self.unhashed_arguments = unhashed_arguments

    def get(self) -> str:
        arguments = {**self.unhashed_arguments, **self.hashed_arguments}
        return string.Template(self.TEMPLATE).substitute(
            {
                "KWARGS": str(instanciate_delayed(arguments)),
                "IMPORT_PATH": self.module,
                "IMPORT_NAME": self.object_name,
                "OBJECT_NAME": self.import_as if self.import_as is not None else self.object_name
            }
        )

    def _sis_hash(self):
        super_hash = super()._sis_hash()
        return sis_hash_helper({"import": super_hash, "hashed_arguments": self.hashed_arguments})