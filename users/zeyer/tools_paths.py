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

from sisyphus import tk, gs


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
    # If it is common to have sclite in the PATH env, we could also check for that here...
    assert getattr(gs, "SCTK_PATH", None), "SCTK_PATH not set"
    return tk.Path(getattr(gs, "SCTK_PATH"), hash_overwrite="DEFAULT_SCTK_BINARY_PATH")


def get_returnn_python_exe() -> tk.Path:
    """
    RETURNN Python executable
    """
    assert getattr(gs, "RETURNN_PYTHON_EXE", None), "RETURNN_PYTHON_EXE not set"
    return tk.Path(getattr(gs, "RETURNN_PYTHON_EXE"), hash_overwrite="DEFAULT_RETURNN_PYTHON_EXE")


def get_returnn_root() -> tk.Path:
    """
    RETURNN root
    """
    assert getattr(gs, "RETURNN_ROOT", None), "RETURNN_ROOT not set"
    return tk.Path(getattr(gs, "RETURNN_ROOT"), hash_overwrite="DEFAULT_RETURNN_ROOT")
