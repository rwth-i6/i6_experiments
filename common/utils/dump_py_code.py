"""
Dump to Python code utils
"""


from typing import Any, Optional, TextIO
from sisyphus import gs, tk
import i6_core.util
import i6_core.rasr as rasr
from i6_core.returnn.config import CodeWrapper
from .python import is_valid_python_attrib_name


_valid_primitive_types = (type(None), int, float, str, bool, i6_core.util.MultiPath)


class PythonCodeDumper:
    """
    Serialize an object to Python code.
    """

    _other_reserved_names = {"gs", "tk", "rasr", "i6core"}

    def __init__(self, *, file: Optional[TextIO] = None):
        self.file = file
        # We use id(obj) for identification. To keep id unique, keep all objects alive.
        self._name_to_id = {}  # name -> id
        self._id_to_obj_name = {}  # id -> (obj, name)
        self._imports = {}  # module name -> module

    def dump(self, obj: Any, *, lhs: str):
        """
        Serializes the object.
        Objects which are referenced multiple times
        will get an own temporary variable name (prefixed with "_").
        """
        assert not lhs.startswith("_")  # reserved for us
        assert lhs not in self._other_reserved_names

        if isinstance(obj, rasr.CommonRasrParameters):
            self._dump_crp(crp=obj, lhs=lhs)
        elif isinstance(obj, rasr.RasrConfig):
            self._dump_rasr_config(config=obj, parent_is_config=False, lhs=lhs)
        else:
            print(f"{lhs} = {self._py_repr(obj)}", file=self.file)

    def _dump_crp(self, crp: rasr.CommonRasrParameters, *, lhs=None):
        """
        Dump rasr.CommonRasrParameters as Python code
        """
        if lhs is None:
            lhs = "crp"
        self._import_reserved("rasr")
        print(f"{lhs} = rasr.CommonRasrParameters()", file=self.file)
        for k, v in vars(crp).items():
            if isinstance(v, rasr.RasrConfig):
                self._dump_rasr_config(f"{lhs}.{k}", v, parent_is_config=False)
            elif isinstance(v, rasr.CommonRasrParameters):
                self._dump_crp(v, lhs=f"{lhs}.{k}")
            elif isinstance(v, dict):
                self._dump_crp_dict(f"{lhs}.{k}", v)
            elif isinstance(v, _valid_primitive_types):
                print(f"{lhs}.{k} = {self._py_repr(v)}", file=self.file)
            else:
                raise TypeError(f"{lhs}.{k} is type {type(v)}")

    def _dump_crp_dict(self, lhs: str, d: dict):
        for k, v in d.items():
            if isinstance(v, rasr.RasrConfig):
                self._dump_rasr_config(f"{lhs}.{k}", v, parent_is_config=False)
            elif isinstance(v, _valid_primitive_types):
                print(f"{lhs}.{k} = {self._py_repr(v)}", file=self.file)
            else:
                raise TypeError(f"{lhs}.{k} is type {type(v)}")

    def _dump_rasr_config(
        self, lhs: str, config: rasr.RasrConfig, *, parent_is_config: bool
    ):
        """
        Dump rasr.RasrConfig as Python code
        """
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
        for k in config:
            v = config[k]
            py_attr = k.replace("-", "_")
            if is_valid_python_attrib_name(py_attr):
                sub_lhs = f"{lhs}.{py_attr}"
            else:
                sub_lhs = f"{lhs}[{k!r}]"
            if isinstance(v, rasr.RasrConfig):
                self._dump_rasr_config(sub_lhs, v, parent_is_config=True)
            else:
                print(f"{sub_lhs} = {self._py_repr(v)}", file=self.file)

    def _py_repr(self, obj: Any) -> str:
        """
        Unfortunately some __repr__ implementations are messed up, so need to use some custom here.
        """
        if isinstance(obj, tk.Path):
            self._import_reserved("tk")
            return f"tk.Path({obj.path!r})"
        if isinstance(obj, i6_core.util.MultiPath):
            return self._py_multi_path_repr(obj)
        if isinstance(obj, dict):
            return f"{{{', '.join(f'{self._py_repr(k)}: {self._py_repr(v)}' for (k, v) in obj.items())}}}"
        if isinstance(obj, list):
            return f"[{', '.join(f'{self._py_repr(v)}' for v in obj)}]"
        if isinstance(obj, tuple):
            return f"({''.join(f'{self._py_repr(v)}, ' for v in obj)})"
        return repr(obj)

    def _py_multi_path_repr(self, p: i6_core.util.MultiPath) -> str:
        """
        repr of MultiPath
        """
        args = [p.path_template, p.hidden_paths, p.cached]
        if p.path_root == gs.BASE_DIR:
            self._import_reserved("gs")
            args.append(CodeWrapper("gs.BASE_DIR"))
        else:
            args.append(p.path_root)
        kwargs = {}
        if p.hash_overwrite:
            kwargs["hash_overwrite"] = p.hash_overwrite
        self._import_reserved("i6_core.util")
        return (
            f"i6core.util.MultiPath("
            f"{', '.join(f'{self._py_repr(v)}' for v in args)}"
            f"{''.join(f', {k}={self._py_repr(v)}' for (k, v) in kwargs.items())})"
        )

    def _register_obj(self, obj: Any):
        if id(obj) in self._id_to_obj_name:
            return
        name = self._new_name_for_obj(obj)
        self._id_to_obj_name[id(obj)] = (obj, name)

    def _new_name_for_obj(self, obj: Any) -> str:
        name = "_" + type(obj).__name__.lower()
        if name not in self._name_to_id:
            return name
        i = 1
        while f"{name}{i}" in self._name_to_id:
            i += 1
        return f"{name}{i}"

    def _import_reserved(self, name: str):
        if name in self._imports:
            return
        code = {
            "gs": "from sisyphus import gs",
            "tk": "from sisyphus import tk",
            "i6_core.util": "import i6_core.util",
            "rasr": "import i6_core.rasr as rasr",
        }
        print(code[name], file=self.file)
        self._imports[name] = globals()[name]
