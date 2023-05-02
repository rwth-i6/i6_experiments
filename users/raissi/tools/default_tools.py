"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from sisyphus import tk
from i6_experiments.common.tools.audio import compile_ffmpeg_binary
from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_experiments.common.tools.sctk import compile_sctk

# RASR_BINARY_PATH = None
# RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")  # use tested RASR
<<<<<<< HEAD
#RASR_BINARY_PATH = compile_rasr_binaries_i6mode()  #  use most recent RASR
rasr_path_tf1 = tk.Path("/u/raissi/dev/rasr_github/rasr_tf1_conformer", hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF1")
rasr_path_tf2 = tk.Path("/u/raissi/dev/rasr_github/rasr_tf2", hash_overwrite="CONFORMER_DEFAULT_RASR_BINARY_PATH_TF2")
RASR_BINARY_PATHS = {'TF1': rasr_path_tf1, 'TF2': rasr_path_tf2}


returnn_launcher_tf1 = tk.Path("/u/raissi/bin/returnn/returnn_tf1.15_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER_TF1")
returnn_launcher_tf2 = tk.Path("/u/raissi/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER_TF1")
RETURNN_LAUNCHERS = {'TF1': returnn_launcher_tf1, 'TF2': returnn_launcher_tf2}

=======
RASR_BINARY_PATH = compile_rasr_binaries_i6mode()  #  use most recent RASR
assert RASR_BINARY_PATH, "Please set a specific RASR_BINARY_PATH before running the pipeline"
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"
>>>>>>> a102a41b (wip tolls set)


SCTK_BINARY_PATH = compile_sctk(branch="v2.4.12")  # use last published version
# SCTK_BINARY_PATH = compile_sctk()  # use most recent SCTK
<<<<<<< HEAD
SCTK_BINARY_PATH.hash_overwrite = "DEFAULT_SCTK_BINARY_PATH"
=======
SCTK_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_SCTK_BINARY_PATH"
>>>>>>> a102a41b (wip tolls set)
