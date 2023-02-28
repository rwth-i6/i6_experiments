from typing import Optional

from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.compile import MakeJob


def compile_ffmpeg_binary(branch: Optional[str] = None, commit: Optional[str] = None) -> tk.Path:
    """
    Compiles FFmpeg using the official Github master.

    For requirements see:
    https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu

    The current configure options are set in a way that the resulting FFmpeg binary lacks many features.
    This is done because we only need limited codec capabilities.

    :param branch: specify a specific branch or tag, e.g. "n4.1.4"
    :param commit: specify a specific commit
    :return: Path object pointing to the "ffmpeg" binary executable
    """
    ffmpeg_repo = CloneGitRepositoryJob(
        "https://github.com/FFmpeg/FFmpeg",
        branch=branch,
        commit=commit,
    )

    ffmpeg_make_binary = MakeJob(
        folder=ffmpeg_repo.out_repository,
        make_sequence=["all"],
        num_processes=4,
        configure_opts=["--disable-x86asm", "--enable-libvorbis"],
        link_outputs={"ffmpeg": "ffmpeg"},
        output_folder_name=None,
    )
    # optionally add additional configuration options without changing the hash:
    # ffmpeg_make_binary.configure_opts += ["libmp3lame-dev"]

    return ffmpeg_make_binary.out_links["ffmpeg"]
