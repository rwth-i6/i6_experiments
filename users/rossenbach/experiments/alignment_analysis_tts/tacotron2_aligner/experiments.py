from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob

from .pipeline import build_training_dataset


def run_tacotron2_aligner_training():

    returnn_exe = tk.Path("/u/rossenbach/bin/returnn/returnn_tf2.3.4_mkl_launcher.sh", hash_overwrite="GENERIC_RETURNN_LAUNCHER")
    returnn_root = CloneGitRepositoryJob("https://github.com/rwth-i6/returnn",
                                         commit="37ba06ab2697e7af4de96037565fdf4f78acdb80").out_repository

    datasets = build_training_dataset(returnn_root=returnn_root, returnn_cpu_exe=returnn_exe,)