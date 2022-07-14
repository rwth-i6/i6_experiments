
"""
repr
"""

from sisyphus import gs, tk
import i6_core.util
from i6_core.returnn.config import CodeWrapper


def py_repr(obj):
    """
    Unfortunately some of the repr implementations are messed up, so need to use some custom here.
    """
    if isinstance(obj, tk.Path):
        return f"tk.Path({obj.path!r})"
    if isinstance(obj, i6_core.util.MultiPath):
        return py_multi_path_repr(obj)
    if isinstance(obj, dict):
        return f"{{{', '.join(f'{py_repr(k)}: {py_repr(v)}' for (k, v) in obj.items())}}}"
    if isinstance(obj, list):
        return f"[{', '.join(f'{py_repr(v)}' for v in obj)}]"
    if isinstance(obj, tuple):
        return f"({''.join(f'{py_repr(v)}, ' for v in obj)})"
    return repr(obj)


def py_multi_path_repr(p: i6_core.util.MultiPath):
    """
    repr of MultiPath
    """
    args = [p.path_template, p.hidden_paths, p.cached]
    if p.path_root == gs.BASE_DIR:
        args.append(CodeWrapper("gs.BASE_DIR"))
    else:
        args.append(p.path_root)
    kwargs = {}
    if p.hash_overwrite:
        kwargs["hash_overwrite"] = p.hash_overwrite
    return (
        f"MultiPath("
        f"{', '.join(f'{py_repr(v)}' for v in args)}"
        f"{''.join(f', {k}={py_repr(v)}' for (k, v) in kwargs.items())})")
