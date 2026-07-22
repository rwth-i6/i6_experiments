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


def get_returnn_onnx_export_exe() -> tk.Path:
    path = getattr(gs, "RETURNN_ONNX_EXPORT_EXE", "/usr/bin/python3")
    return tk.Path(path, hash_overwrite="GENERIC_RETURNN_ONNX_EXPORT_LAUNCHER")


def get_fasttext_python_exe() -> tk.Path:
    path = getattr(gs, "FASTTEXT_PYTHON_EXE", "/usr/bin/python3")
    return tk.Path(path, hash_overwrite="FASTTEXT_PYTHON_EXE")


def get_returnn_root() -> tk.Path:
    path = getattr(gs, "RETURNN_ROOT", "returnn")
    return tk.Path(path, hash_overwrite="DEFAULT_RETURNN_ROOT")


def get_returnn_onnx_export_root() -> tk.Path:
    returnn_root = CloneGitRepositoryJob(
        "https://github.com/rwth-i6/returnn",
        branch="robin-support-onnx-export",
        checkout_folder_name="returnn",
    ).out_repository
    returnn_root.hash_overwrite = "DEFAULT_RETURNN_ONNX_EXPORT_ROOT"
    return returnn_root


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


def get_rvad_root():
    """
    Clone the rVADfast repository and return its path.
    """
    out_repository = CloneGitRepositoryJob(
        url="https://github.com/zhenghuatan/rVADfast.git",
        checkout_folder_name="rVADfast",
        commit="0ed4c1246ad5fdb1cead801153f455b9cf6d569b",
    ).out_repository.copy()
    return out_repository


def get_wav2letter_root():
    return CloneGitRepositoryJob(
        "https://github.com/flashlight/wav2letter",
        commit="e5a4b62d87f15fde6a963d9ac174c8db8eb67fbc",
        checkout_folder_name="wav2letter",
    ).out_repository


RETURNN_EXE = get_returnn_exe()
RETURNN_ONNX_EXE = get_returnn_onnx_export_exe()

RETURNN_ROOT = get_returnn_root()
RETURNN_ONNX_ROOT = get_returnn_onnx_export_root()

SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12").copy()  # use last published version
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"
