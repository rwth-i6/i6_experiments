from typing import Optional

from sisyphus import tk

from i6_core.tools.compile import MakeJob
from i6_core.tools.git import CloneGitRepositoryJob


def compile_rasr_binaries_i6mode(
    branch: Optional[str] = None,
    commit: Optional[str] = None,
    rasr_git_repository: str = "https://github.com/rwth-i6/rasr",
    rasr_arch: str = "linux-x86_64-standard",
) -> tk.Path:
    """
    Compile RASR for i6-internal usage

    :param branch: specify a specific branch
    :param commit: specify a specific commit
    :param rasr_git_repository: where to clone RASR from, usually does not need to be altered
    :param rasr_arch: RASR compile architecture string
    :return: path to the binary folder
    """
    rasr_repo = CloneGitRepositoryJob(rasr_git_repository, branch=branch, commit=commit).out_repository
    make_job = MakeJob(
        folder=rasr_repo,
        make_sequence=["build", "install"],
        configure_opts=["--i6"],
        num_processes=8,
        link_outputs={"binaries": f"arch/{rasr_arch}/"},
    )
    make_job.rqmt["mem"] = 8
    return make_job.out_links["binaries"]
