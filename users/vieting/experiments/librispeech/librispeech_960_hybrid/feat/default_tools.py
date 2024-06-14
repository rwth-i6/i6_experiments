"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from sisyphus import tk
from i6_core.tools.git import CloneGitRepositoryJob
from i6_experiments.common.tools.audio import compile_ffmpeg_binary
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk
from i6_experiments.users.vieting.tools.conda import InstallMinicondaJob, CreateCondaEnvJob

# RASR_BINARY_PATH = None
# RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")  # use tested RASR
# RASR_BINARY_PATH = compile_rasr_binaries_i6mode(
#     commit="d506533105e636bf3df37dffbe41d84c248d0a5a"
# )  #  use most recent RASR
RASR_BINARY_PATH = tk.Path("/work/asr4/vieting/programs/rasr/20230707/rasr/arch/linux-x86_64-standard")
assert (
    RASR_BINARY_PATH
), "Please set a specific RASR_BINARY_PATH before running the pipeline"
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"


SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"

# conda = InstallMinicondaJob()
# packages = {
#     "numpy": "==1.18.5",
#     "tensorflow-gpu": "==2.4.1",
#     "pysoundfile": "==0.10.2",
#     "h5py": "==2.10.0",
#     "typing": "==3.7.4.3",
#     "black": "==22.3.0",
#     "flask": "==1.1.1",
#     "ipdb": "==0.13.3",
#     "tqdm": "==4.61.2",
# }
# conda_env_job = CreateCondaEnvJob(
#     conda.out_conda_exe, python_version="3.8", channels=["conda-forge"], packages=packages,
# )
# conda_env_job.add_alias("tools/conda_envs/returnn_training")
# RETURNN_EXE = conda_env_job.out_env_exe
RETURNN_EXE = tk.Path(
    "/usr/bin/python3",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)
RETURNN_CPU_EXE = tk.Path(
    "/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_generic_launcher.sh",
    hash_overwrite="GENERIC_RETURNN_LAUNCHER",
)

RETURNN_ROOT = CloneGitRepositoryJob(
    "https://github.com/rwth-i6/returnn",
    #commit="6d2945a85cc95df5349a59541d84f172dd55cc20",
).out_repository
RETURNN_ROOT.hash_overwrite = "SWITCHBOARD_DEFAULT_RETURNN_ROOT"

RETURNN_COMMON = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn_common", commit="e3083fac1899bb764710ca46ff9257247e4e6b14", checkout_folder_name="returnn_common").out_repository
RETURNN_COMMON.hash_overwrite = "LIBRISPEECH_DEFAULT_RETURNN_COMMON"