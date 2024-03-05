"""
A generic interface to get the paths to the tools.

The default fallback behavior:

These are intended to be used for setups where you do not want the hash to change
when you update the corresponding tool (e.g. RASR, RETURNN, etc).
The assumption is that you would use the latest version of the tool,
and the behavior would not change.

Thus, we define a fixed hash_overwrite here, which is supposed to never change anymore,
and reflect such global versions.

Currently, we get the binaries via the Sisyphus global settings,
but we might extend that mechanism.

However, if needed, we can extend this interface that it is possible
to overwrite the tools.
This could be done via a context scope,
or just overwrite it globally.

Also see i6_experiments/common/baselines/librispeech/default_tools.py.
"""

from __future__ import annotations
from typing import Optional, Callable
import sys
import shutil
from sisyphus import tk, gs


def monkey_patch_i6_core():
    """
    Monkey patch i6_core.util get_executable_path and maybe related methods
    to avoid such warnings::

        use of gs is deprecated, please provide a Path object for gs.RETURNN_PYTHON_EXE
        use of gs is deprecated, please provide a Path object for gs.RETURNN_ROOT
        Creating absolute path inside current work directory: /u/zeyer/setups/combined/2021-05-31/tools/returnn
            (disable with WARNING_ABSPATH=False)
        ...

    Those warnings happen because of the logic of :func:`i6_core.util.get_executable_path`.
    """
    from i6_core import util

    global _orig_i6_core_util_get_executable_path
    if _orig_i6_core_util_get_executable_path is None:
        _orig_i6_core_util_get_executable_path = util.get_executable_path

    util.get_executable_path = _i6_core_util_get_executable_path
    util.get_returnn_python_exe = _i6_core_util_get_returnn_python_exe
    util.get_returnn_root = _i6_core_util_get_returnn_root


def get_rasr_binary_path() -> tk.Path:
    """
    RASR binary path

    RASR_ROOT example (set via gs): "/work/tools/asr/rasr/20220603_github_default/"
    RASR binary path example: '{RASR_ROOT}/arch/{RASR_ARCH}'
    """
    assert getattr(gs, "RASR_ROOT", None), "RASR_ROOT not set"
    rasr_root = getattr(gs, "RASR_ROOT")
    rasr_arch = get_rasr_arch()
    return tk.Path(f"{rasr_root}/arch/{rasr_arch}", hash_overwrite="DEFAULT_RASR_BINARY_PATH")


def get_rasr_arch() -> str:
    """RASR arch"""
    return getattr(gs, "RASR_ARCH", None) or "linux-x86_64-standard"


def get_rasr_exe(name: str) -> tk.Path:
    """
    Like i6_core.rasr.command.RasrCommand.select_exe, or i6_core.rasr.crp.CommonRasrParameters.set_executables.

    :param name: e.g. "nn-trainer"
    :return: path to RASR binary executable
    """
    rasr_binary_path = get_rasr_binary_path()
    rasr_arch = get_rasr_arch()
    return rasr_binary_path.join_right(f"{name}.{rasr_arch}")


def get_sctk_binary_path() -> tk.Path:
    """SCTK binary path"""
    hash_overwrite = "DEFAULT_SCTK_BINARY_PATH"
    sctk_path = getattr(gs, "SCTK_PATH", None)

    if sctk_path is None:
        from i6_experiments.common.tools.sctk import compile_sctk

        path = compile_sctk(branch="v2.4.12")  # use last published version
        path.hash_overwrite = hash_overwrite
        return path

    return tk.Path(sctk_path, hash_overwrite=hash_overwrite)


def get_returnn_python_exe() -> tk.Path:
    """
    RETURNN Python executable
    """
    hash_overwrite = "DEFAULT_RETURNN_PYTHON_EXE"
    path = getattr(gs, "RETURNN_PYTHON_EXE", None)
    if path is None:
        path = sys.executable
    return tk.Path(path, hash_overwrite=hash_overwrite)


def get_returnn_root() -> tk.Path:
    """
    RETURNN root
    """
    assert getattr(gs, "RETURNN_ROOT", None), "RETURNN_ROOT not set"
    return tk.Path(getattr(gs, "RETURNN_ROOT"), hash_overwrite="DEFAULT_RETURNN_ROOT")


_orig_i6_core_util_get_executable_path: Optional[Callable[..., tk.Path]] = None


def _i6_core_util_get_executable_path(
    path: Optional[tk.Path],
    gs_member_name: Optional[str],
    default_exec_path: Optional[tk.Path] = None,
) -> tk.Path:
    """
    Helper function that allows to select a specific version of software while
    maintaining compatibility to different methods that were used in the past to select
    software versions.
    It will return a Path object for the first path found in

    :param path: Directly specify the path to be used
    :param gs_member_name: get path from sisyphus.global_settings.<gs_member_name>
    :param default_exec_path: general fallback if no specific version is given
    """
    if path is not None:
        return _orig_i6_core_util_get_executable_path(path, gs_member_name, default_exec_path)
    if getattr(gs, gs_member_name, None) is not None:
        # Custom hash_overwrite, to avoid the warning, and also to have the hash independent of the path.
        return tk.Path(getattr(gs, gs_member_name), hash_overwrite=f"gs.{gs_member_name}")
    if default_exec_path is not None:
        return default_exec_path
    assert False, f"could not find executable for {gs_member_name}"


def _i6_core_util_get_returnn_python_exe(returnn_python_exe: Optional[tk.Path]) -> tk.Path:
    if returnn_python_exe is None:
        return get_returnn_python_exe()
    system_python = tk.Path(shutil.which(gs.SIS_COMMAND[0]))
    return _orig_i6_core_util_get_executable_path(returnn_python_exe, "RETURNN_PYTHON_EXE", system_python)


def _i6_core_util_get_returnn_root(returnn_root: Optional[tk.Path]) -> tk.Path:
    if returnn_root is None:
        return get_returnn_root()
    return _orig_i6_core_util_get_executable_path(returnn_root, "RETURNN_ROOT")
