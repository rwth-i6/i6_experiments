"""
List of default tools and software to be defined as default independent from hashing
by setting one explicit hash.

In order to use different software paths without hash changes, just use the same explicit hash string as given here.

If you want a stronger guarantee that you get the intended results, please consider using the explicit software
version listed here. Nevertheless, the most recent "head" should be safe to be used as well

"""
from typing import Optional

from sisyphus import tk
# from i6_experiments.common.tools.audio import compile_ffmpeg_binary
# from i6_experiments.common.tools.rasr import compile_rasr_binaries_i6mode
from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.compile import MakeJob


def compile_rasr_binaries_i6mode(
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        rasr_git_repository: str = "https://github.com/rwth-i6/rasr",
) -> tk.Path:
    """
    Compile RASR for i6-internal usage

    :param branch: specify a specific branch
    :param commit: specify a specific commit
    :param rasr_git_repository: where to clone RASR from, usually does not need to be altered
    :return: path to the binary folder
    """
    rasr_repo = CloneGitRepositoryJob(
        rasr_git_repository, branch=branch, commit=commit
    ).out_repository
    make_job = MakeJob(
        folder=rasr_repo,
        make_sequence=["build", "install"],
        configure_opts=["--i6"],
        num_processes=8,
        link_outputs={"binaries": "arch/linux-x86_64-standard/"},
    )
    make_job.rqmt["mem"] = 8
    return make_job.out_links["binaries"]


RASR_BINARY_PATH = None
RASR_BINARY_PATH = compile_rasr_binaries_i6mode(commit="907eec4f4e36c11153f6ab6b5dd7675116f909f6")  # use tested RASR
# RASR_BINARY_PATH = compile_rasr_binaries_i6mode()  #  use most recent RASR
assert RASR_BINARY_PATH, "Please set a specific RASR_BINARY_PATH before running the pipeline"
RASR_BINARY_PATH.hash_overwrite = "LIBRISPEECH_DEFAULT_RASR_BINARY_PATH"

