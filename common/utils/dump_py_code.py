"""
Dump to Python code utils
"""

import re
import sys
from operator import attrgetter
from typing import Any, Optional, TextIO
from sisyphus import gs, tk
from sisyphus.hash import sis_hash_helper
import i6_core.util
import i6_core.rasr as rasr
from .python import is_valid_python_identifier_name


_valid_primitive_types = (type(None), int, float, str, bool, tk.Path, type)


class PythonCodeDumper:
    """
    Serialize an object to Python code.
    """

    _other_reserved_names = {"gs", "tk", "rasr", "i6core", "i6experiments"}

    def __init__(self, *, file: Optional[TextIO] = None):
        self.file = file
        # We use id(obj) for identification. To keep id unique, keep all objects alive.
        self._reserved_names = set()
        self._id_to_obj_name = {}  # id -> (obj, name). like pickle memo
        self._imports = {}  # module name -> module

    def dump(self, obj: Any, *, lhs: str):
        """
        Serializes the object.
        Objects which are referenced multiple times
        will get an own temporary variable name (prefixed with "_").
        """
        assert not lhs.startswith("_")  # reserved for us
        assert lhs not in self._other_reserved_names
        assert is_valid_python_identifier_name(lhs)
        # Clear any previous memo. Any mutable objects could have been changed in the meantime.
        self._id_to_obj_name.clear()
        self._reserved_names.clear()
        self._dump(obj, lhs=lhs)

    def _dump(self, obj: Any, *, lhs: str, check_memo: bool = True):
        if check_memo and id(obj) in self._id_to_obj_name:
            print(f"{lhs} = {self._id_to_obj_name[id(obj)][1]}", file=self.file)
            return
        if isinstance(obj, rasr.CommonRasrParameters):
            self._dump_crp(crp=obj, lhs=lhs)
        elif isinstance(obj, rasr.RasrConfig):
            self._dump_rasr_config(config=obj, parent_is_config=False, lhs=lhs)
        elif isinstance(obj, dict):
            self._dump_dict(obj, lhs=lhs)
        elif isinstance(obj, (list, tuple, set)):
            print(f"{lhs} = {self._py_repr(obj)}", file=self.file)
            self._register_obj(obj, name=lhs)
        elif isinstance(obj, i6_core.util.MultiPath):
            self._dump_multi_path(obj, lhs=lhs)
        elif isinstance(obj, _valid_primitive_types):
            print(f"{lhs} = {self._py_repr(obj)}", file=self.file)
        else:
            # We follow a similar logic as pickle does (but simplified).
            # See pickle._Pickler.save() for reference.
            # It tries to use obj.__reduce_ex__ or obj.__reduce__.
            # Also see copyref._reduce_ex, __newobj__ etc.
            # However, we simplify this further to what you would get for new-style class objects.
            # See the logic of pickle NEWOBJ and others.
            cls = type(obj)
            assert issubclass(cls, object)  # not implemented otherwise
            print(f"{lhs} = object.__new__({self._py_repr(cls)})", file=self.file)
            self._register_obj(obj, name=lhs)
            if hasattr(obj, "__getstate__"):
                state = obj.__getstate__()
            else:
                state = obj.__dict__
            if state:
                if hasattr(obj, "__setstate__"):
                    print(f"{lhs}.__setstate__({self._py_repr(state)})", file=self.file)
                else:
                    slotstate = None
                    if isinstance(state, tuple) and len(state) == 2:
                        state, slotstate = state
                    if state:
                        for k, v in state.items():
                            self._dump(v, lhs=f"{lhs}.{k}")
                    if slotstate:
                        for k, v in slotstate.items():
                            self._dump(v, lhs=f"{lhs}.{k}")

    def _dump_dict(self, obj: dict, *, lhs: str):
        lines = [f"{lhs} = {{"]
        for k, v in obj.items():
            lines.append(f"    {self._py_repr(k)}: {self._py_repr(v)},")
        lines.append("}")
        for line in lines:
            print(line, file=self.file)
        self._register_obj(obj, name=lhs)

    def _dump_crp(self, crp: rasr.CommonRasrParameters, *, lhs=None):
        if lhs is None:
            lhs = "crp"
        self._import_reserved("rasr")
        print(f"{lhs} = rasr.CommonRasrParameters()", file=self.file)
        self._register_obj(crp, name=lhs)
        for k, v in vars(crp).items():
            if isinstance(v, dict):
                self._dump_crp_dict(lhs=f"{lhs}.{k}", d=v)
            else:
                self._dump(v, lhs=f"{lhs}.{k}")

    def _dump_crp_dict(self, d: dict, *, lhs: str):
        for k, v in d.items():
            self._dump(v, lhs=f"{lhs}.{k}")

    def _dump_rasr_config(
        self, config: rasr.RasrConfig, *, lhs: str, parent_is_config: bool
    ):
        kwargs = {}
        for k in ["prolog", "epilog"]:
            v = getattr(config, f"_{k}")
            h = getattr(config, f"_{k}_hash")
            if v:
                kwargs[k] = v
                if h != v:
                    kwargs[f"{k}_hash"] = h
            else:
                assert not h
        if kwargs or not parent_is_config:
            assert config._value is None  # noqa
            self._import_reserved("rasr")
            print(
                f"{lhs} = rasr.RasrConfig({', '.join(f'{k}={v!r}' for (k, v) in kwargs.items())})",
                file=self.file,
            )
        else:
            if config._value is not None:  # noqa
                print(f"{lhs} = {config._value!r}", file=self.file)  # noqa
        self._register_obj(config, name=lhs)
        for k in config:
            v = config[k]
            py_attr = k.replace("-", "_")
            if is_valid_python_identifier_name(py_attr):
                sub_lhs = f"{lhs}.{py_attr}"
            else:
                sub_lhs = f"{lhs}[{k!r}]"
            if isinstance(v, rasr.RasrConfig):
                self._dump_rasr_config(lhs=sub_lhs, config=v, parent_is_config=True)
            else:
                self._dump(v, lhs=sub_lhs)

    def _dump_multi_path(self, p: i6_core.util.MultiPath, *, lhs: str):
        lines = [
            f"{lhs} = i6_core.util.MultiPath(",
            f"    {self._py_repr(p.path_template)},",
            f"    {self._py_repr(p.hidden_paths)},",
            f"    {self._py_repr(p.cached)},",
            f"    {self._py_repr(p.path_root)},",
        ]
        if p.hash_overwrite:
            lines.append(f"    hash_overwrite={self._py_repr(p.hash_overwrite)},")
        lines.append(")")
        self._import_user_mod("i6_core.util")
        for line in lines:
            print(line, file=self.file)
        self._register_obj(p, name=lhs)

    def _py_repr(self, obj: Any) -> str:
        """
        Unfortunately some __repr__ implementations are messed up, so need to use some custom here.
        """
        if id(obj) in self._id_to_obj_name:
            return self._id_to_obj_name[id(obj)][1]
        if isinstance(obj, tk.Path):
            self._import_reserved("tk")
            return f"tk.Path({obj.path!r})"
        if isinstance(obj, dict):
            return self._name_for_obj(obj)
        if isinstance(obj, list):
            return f"[{', '.join(f'{self._py_repr(v)}' for v in obj)}]"
        if isinstance(obj, tuple):
            return f"({''.join(f'{self._py_repr(v)}, ' for v in obj)})"
        if isinstance(obj, set):
            if not obj:
                return "set()"
            return f"{{{', '.join(sorted([self._py_repr(v) for v in obj], key=sis_hash_helper))}}}"
        if isinstance(obj, type):
            self._import_user_mod(obj.__module__)
            assert attrgetter(obj.__qualname__)(sys.modules[obj.__module__]) is obj
            return f"{obj.__module__}.{obj.__qualname__}"
        if isinstance(obj, str):
            if obj is gs.BASE_DIR:
                self._import_reserved("gs")
                return "gs.BASE_DIR"
            return repr(obj)
        if isinstance(obj, _valid_primitive_types):
            return repr(obj)
        return self._name_for_obj(obj)

    def _name_for_obj(self, obj: Any) -> str:
        if id(obj) in self._id_to_obj_name:
            return self._id_to_obj_name[id(obj)][1]
        name = self._new_name_for_obj(obj)
        self._dump(obj, lhs=name, check_memo=False)
        return name

    def _register_obj(self, obj: Any, *, name: str):
        if id(obj) in self._id_to_obj_name:
            return
        self._id_to_obj_name[id(obj)] = (obj, name)
        self._reserved_names.add(name)

    def _new_name_for_obj(self, obj: Any) -> str:
        name = type(obj).__name__
        name = re.sub(
            r"(?<!^)(?=[A-Z])", "_", name
        ).lower()  # https://stackoverflow.com/a/1176023/133374
        if not name.startswith("_"):
            name = "_" + name
        i = 0
        while True:
            name_ = f"{name}{i}" if i > 0 else name
            if name_ not in self._reserved_names:
                self._reserved_names.add(name_)
                return name_
            i += 1

    def _import_user_mod(self, name: str):
        if name in self._imports:
            return
        print(f"import {name}", file=self.file)
        self._imports[name] = sys.modules[name]

    def _import_reserved(self, name: str):
        if name in self._imports:
            return
        code = {
            "gs": "from sisyphus import gs",
            "tk": "from sisyphus import tk",
            "rasr": "import i6_core.rasr as rasr",
        }
        print(code[name], file=self.file)
        self._imports[name] = globals()[name]
