"""
Defines the external software to be used for the Experiments
"""

from typing import Optional
import shutil

from sisyphus import tk, gs

from i6_experiments.common.tools.sctk import compile_sctk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.download import DownloadJob


def get_ffmpeg_binary() -> tk.Path:
    """
    FFMPEG binary
    """
    path = getattr(gs, "FFMPEG_BINARY", None)
    if path is None:
        path = shutil.which("ffmpeg")
        assert path, "ffmpeg not found"
    return tk.Path(path, hash_overwrite="DEFAULT_FFMPEG_BINARY")


def get_returnn_exe() -> tk.Path:
    path = getattr(gs, "RETURNN_EXE", "/usr/bin/python3")
    return tk.Path(path, hash_overwrite="GENERIC_RETURNN_LAUNCHER")


def get_fasttext_python_exe() -> tk.Path:
    path = getattr(gs, "FASTTEXT_PYTHON_EXE", "/usr/bin/python3")
    return tk.Path(path, hash_overwrite="FASTTEXT_PYTHON_EXE")


def get_returnn_root() -> tk.Path:
    path = getattr(gs, "RETURNN_ROOT", "returnn")
    return tk.Path(path, hash_overwrite="DEFAULT_RETURNN_ROOT")


def get_fairseq_root(
    commit: Optional[str] = "ecbf110e1eb43861214b05fa001eff584954f65a",
):
    """
    :param python_env: path to the python environment where fairseq will be installed
    """

    fairseq_root = CloneGitRepositoryJob(
        "https://github.com/facebookresearch/fairseq", checkout_folder_name="fairseq", commit=commit
    ).out_repository

    return fairseq_root


def get_lid_model():
    return DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file


RETURNN_EXE = get_returnn_exe()

RETURNN_ROOT = get_returnn_root()

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"
