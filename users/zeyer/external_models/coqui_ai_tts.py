"""
Coqui-ai TTS: https://github.com/coqui-ai/TTS
own fork: https://github.com/albertz/coqui-ai-tts
"""

from typing import Optional
from sisyphus import Job, Task, Path, tk


def py():
    """
    demo to run this directly with Sisyphus
    """
    path = download_model("tts_models/multilingual/multi-dataset/your_tts")
    tk.register_output("external_models/coqui_ai_tts/your_tts", path)


def download_model(model_name: str, *, tts_repo_dir: Optional[Path] = None):
    """
    :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
    :param tts_repo_dir: if not specified, uses :func:`get_default_tts_repo_dir`
    """
    if tts_repo_dir is None:
        tts_repo_dir = get_default_tts_repo_dir()
    download_job = DownloadModel(model_name=model_name, tts_repo_dir=tts_repo_dir)
    return download_job.out_tts_data_dir


class DownloadModel(Job):
    """
    Downloads a model via the Coqui-ai TTS api.
    See: https://github.com/coqui-ai/TTS?tab=readme-ov-file#-python-api

    Thus, it requires the TTS repo (with TTS source code; e.g. via :func:`get_default_tts_repo_dir`),
    and then will use that Python API.
    Thus, it requires the TTS dependencies: https://github.com/coqui-ai/TTS/blob/dev/requirements.txt
    Specifically, that should be:
    - torch
    - coqpit
    - trainer (might need --ignore-requires-python)
    - tqdm
    - pysbd
    - mutagen
    - pandas
    - anyascii
    - inflect
    - bangla
    - bnnumerizer
    - bnunicodenormalizer
    - gruut
    - jamo
    - jieba
    - pypinyin
    """

    def __init__(self, *, model_name: str, tts_repo_dir: Path):
        """
        :param model_name: for example "tts_models/multilingual/multi-dataset/your_tts"
        :param tts_repo_dir:
        """
        super().__init__()
        self.model_name = model_name
        self.tts_repo_dir = tts_repo_dir
        self.out_tts_data_dir = self.output_path("tts_home", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import sys
        import os
        import tempfile
        import shutil
        import stat
        from sisyphus import gs

        sys.path.insert(0, self.tts_repo_dir.get_path())

        with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
            print("using temp-dir:", tmp_dir)

            os.environ["TTS_HOME"] = tmp_dir
            os.environ["COQUI_TOS_AGREED"] = "1"

            from TTS.api import TTS

            tts = TTS(model_name=self.model_name, progress_bar=False)
            assert str(tts.manager.output_prefix) == tmp_dir + "/tts"
            assert os.path.isdir(tts.manager.output_prefix)
            dir_content = os.listdir(tts.manager.output_prefix)
            print(".../tts dir content:", dir_content)
            assert dir_content  # non-empty
            shutil.copytree(tts.manager.output_prefix, self.out_tts_data_dir.get_path() + "/tts")

        # Make read-only, to ensure we don't modify it further.
        for root, dirs, files in os.walk(self.out_tts_data_dir.get_path()):
            for fn in dirs + files:
                fn_ = os.path.join(root, fn)
                os.chmod(fn_, os.stat(fn_).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def get_default_tts_repo_dir() -> Path:
    """
    :return: upstream, via :func:`get_upstream_tts_git_repo`
    """
    return get_upstream_tts_git_repo()


def get_upstream_tts_git_repo() -> Path:
    """
    :return: upstream, via :class:`CloneGitRepositoryJob`, from https://github.com/coqui-ai/TTS.git
    """
    from i6_core.tools.git import CloneGitRepositoryJob

    clone_job = CloneGitRepositoryJob(
        "https://github.com/coqui-ai/TTS.git", commit="dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e"
    )
    return clone_job.out_repository
