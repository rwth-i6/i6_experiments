"""
Helper code for serializing any data, e.g. for ReturnnConfig.
"""

from __future__ import annotations
from typing import Any, Union, Optional, List
from types import FunctionType
import sys
import textwrap

from sisyphus import tk
from sisyphus.hash import sis_hash_helper, short_hash
from sisyphus.delayed_ops import DelayedBase

from i6_core.util import uopen


class SerializerObject(DelayedBase):
    """
    Base class for objects that can be passed to :class:`Collection` or :class:`returnn_common.Collection`.
    """

    use_for_hash = True

    def __init__(self):
        # suppress init warning
        super().__init__(None)

    def get(self) -> str:
        """get"""
        raise NotImplementedError


class Collection(DelayedBase):
    """
    Collection of a list of :class:`SerializerObject`
    """

    def __init__(
        self,
        serializer_objects: List[SerializerObject],
    ):
        """
        :param serializer_objects: all serializer objects which are serialized into a string in order.
            For the hash, it will ignore those with use_for_hash=False.
        """
        super().__init__(None)
        self.serializer_objects = serializer_objects

    def get(self) -> str:
        """get"""
        content = [obj.get() for obj in self.serializer_objects]
        return "".join(content)

    def _sis_hash(self) -> bytes:
        h = {
            "delayed_objects": [obj for obj in self.serializer_objects if obj.use_for_hash],
        }
        return sis_hash_helper(h)


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
            assert getattr(code_object_path, "__qualname__", None) and getattr(code_object_path, "__module__", None)
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
            return sis_hash_helper({"code_object": self.code_object, "import_as": self.import_as})
        return sis_hash_helper(self.code_object)


class CodeFromFunction(SerializerObject):
    """
    Insert code from function.
    """

    def __init__(self, name: str, func: FunctionType, *, hash_full_python_code: bool = False):
        """
        :param name: name of the function as exposed in the config
        :param func:
        :param hash_full_python_code: if True, the full python code of the function is hashed,
            otherwise only the module name and function qualname are hashed.
        """
        super().__init__()
        self.name = name
        self.func = func
        self.hash_full_python_code = hash_full_python_code

        # Similar as ReturnnConfig.
        import inspect

        self._func_code = inspect.getsource(self.func)
        code_hash = short_hash(self._func_code)
        if self.func.__name__ == self.name:
            self._code = self._func_code
        else:
            # Wrap the code inside a function to be sure we do not conflict with other names.
            self._code = "".join(
                [
                    f"def _{self.name}_{code_hash}():\n",
                    textwrap.indent(self._func_code, "    "),
                    "\n",
                    f"    return {self.func.__name__}\n",
                    f"{self.name} = _{self.name}_{code_hash}()\n",
                ]
            )

    def get(self):
        """get"""
        return self._code

    def _sis_hash(self):
        if self.hash_full_python_code:
            return sis_hash_helper((self.name, self._func_code))
        else:
            return sis_hash_helper((self.name, f"{self.func.__module__}.{self.func.__qualname__}"))


# noinspection PyAbstractClass
class _NonhashedSerializerObject(SerializerObject):
    """
    Any serializer object which is not used for the hash.
    """

    use_for_hash = False

    def _sis_hash(self):
        raise Exception(f"{self.__class__.__name__} must not be hashed")


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
