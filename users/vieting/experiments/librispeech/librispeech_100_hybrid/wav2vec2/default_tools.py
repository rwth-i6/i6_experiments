from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode


PACKAGE = __package__

#RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")
RASR_BINARY_PATH = compile_rasr_binaries_i6mode(branch="bene_unpushed_assert")
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"

RASR_BINARY_PATH_U22 = tk.Path("/work/asr4/rossenbach/neon_test/rasr_versions/rasr_no_tf/arch/linux-x86_64-standard/")
RASR_BINARY_PATH_U22.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"

RASR_BINARY_PATH_APPTEK_APPTAINER = tk.Path("/work/asr4/rossenbach/rescale/pytorch_mixed_precision/onnx_extended_rasr/arch/linux-x86_64-standard")
RASR_BINARY_PATH_APPTEK_APPTAINER.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"

RETURNN_EXE = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
RETURNN_DATA_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                          commit="37ba06ab2697e7af4de96037565fdf4f78acdb80").out_repository

RETURNN_RC_ROOT = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn", commit="ac52150e2309e00d55af70e879c176c6400cc536").out_repository
RETURNN_RC_ROOT.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_RC_ROOT"

RETURNN_COMMON = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="e3083fac1899bb764710ca46ff9257247e4e6b14", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"
