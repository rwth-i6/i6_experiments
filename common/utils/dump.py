"""
Dump to Python code utils
"""


import i6_core.util
import i6_core.rasr as rasr
from .python import is_valid_python_attrib_name
from .repr import py_repr


_valid_primitive_types = (type(None), int, float, str, bool, i6_core.util.MultiPath)


def dump_crp(crp: rasr.CommonRasrParameters, *, _lhs=None, file=None):
    """
    Dump rasr.CommonRasrParameters as Python code
    """
    if _lhs is None:
        _lhs = "crp"
    print(f"{_lhs} = rasr.CommonRasrParameters()", file=file)
    for k, v in vars(crp).items():
        if isinstance(v, rasr.RasrConfig):
            dump_rasr_config(f"{_lhs}.{k}", v, parent_is_config=False, file=file)
        elif isinstance(v, rasr.CommonRasrParameters):
            dump_crp(v, _lhs=f"{_lhs}.{k}", file=file)
        elif isinstance(v, dict):
            _dump_crp_dict(f"{_lhs}.{k}", v, file=file)
        elif isinstance(v, _valid_primitive_types):
            print(f"{_lhs}.{k} = {py_repr(v)}", file=file)
        else:
            raise TypeError(f"{_lhs}.{k} is type {type(v)}")


def _dump_crp_dict(lhs: str, d: dict, *, file=None):
    for k, v in d.items():
        if isinstance(v, rasr.RasrConfig):
            dump_rasr_config(f"{lhs}.{k}", v, parent_is_config=False, file=file)
        elif isinstance(v, _valid_primitive_types):
            print(f"{lhs}.{k} = {py_repr(v)}", file=file)
        else:
            raise TypeError(f"{lhs}.{k} is type {type(v)}")


def dump_rasr_config(lhs: str, config: rasr.RasrConfig, *, parent_is_config: bool, file=None):
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
        print(f"{lhs} = rasr.RasrConfig({', '.join(f'{k}={v!r}' for (k, v) in kwargs.items())})", file=file)
    else:
        if config._value is not None:  # noqa
            print(f"{lhs} = {config._value!r}", file=file)  # noqa
    for k in config:
        v = config[k]
        py_attr = k.replace("-", "_")
        if is_valid_python_attrib_name(py_attr):
            sub_lhs = f"{lhs}.{py_attr}"
        else:
            sub_lhs = f"{lhs}[{k!r}]"
        if isinstance(v, rasr.RasrConfig):
            dump_rasr_config(sub_lhs, v, parent_is_config=True, file=file)
        else:
            print(f"{sub_lhs} = {py_repr(v)}", file=file)
