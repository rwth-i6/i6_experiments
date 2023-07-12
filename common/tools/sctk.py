from typing import Optional

from sisyphus import tk

from i6_core.tools.compile import MakeJob
from i6_core.tools.git import CloneGitRepositoryJob


def compile_sctk(
    branch: Optional[str] = None,
    commit: Optional[str] = None,
    sctk_git_repository: str = "https://github.com/usnistgov/SCTK.git",
) -> tk.Path:
    """
    :param branch: specify a specific branch
    :param commit: specify a specific commit
    :param sctk_git_repository: where to clone SCTK from, usually does not need to be altered
    :return: SCTK binary folder
    """
    sctk_repo = CloneGitRepositoryJob(url=sctk_git_repository, branch=branch, commit=commit).out_repository
    sctk_make = MakeJob(
        folder=sctk_repo,
        make_sequence=["config", "all", "check", "install", "doc"],
        link_outputs={"bin": "bin/"},
    )
    # This is needed for the compilation to work in the i6 environment, otherwise still untested
    sctk_make._sis_environment.set("CPPFLAGS", "-std=c++11")
    return sctk_make.out_links["bin"]
