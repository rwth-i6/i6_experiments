"""
Coqui-ai TTS: https://github.com/coqui-ai/TTS
own fork: https://github.com/albertz/coqui-ai-tts
"""

from __future__ import annotations
import os
import sys
import functools
from typing import Optional, TypeVar


_my_dir = os.path.dirname(__file__)
_base_dir = functools.reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.external_models"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


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

    To use it later, set the ``TTS_HOME`` env var to ``out_tts_data_dir.get_path()``.
    Also, set the env var ``COQUI_TOS_AGREED=1``.

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


def _demo():
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model-name", default="tts_models/multilingual/multi-dataset/your_tts")
    arg_parser.add_argument("--model-dir", default="output/external_models/coqui_ai_tts/your_tts")
    arg_parser.add_argument("--tts-repo-dir")
    arg_parser.add_argument("--device", default="cuda")
    args = arg_parser.parse_args()

    print(f"{os.path.basename(__file__)} demo, {args=}")

    import torch

    dev = torch.device(args.device)

    model_name = args.model_name
    model_dir = args.model_dir
    assert os.path.exists(model_dir), f"model dir does not exist: {model_dir}"
    assert os.path.exists(model_dir + "/tts"), f"model dir does not exist: {model_dir}/tts"

    tts_repo_dir = args.tts_repo_dir
    if not tts_repo_dir:
        tts_repo_dir = get_default_tts_repo_dir().get_path()
        print("Using default TTS repo dir:", tts_repo_dir)

    sys.path.insert(0, tts_repo_dir)
    os.environ["TTS_HOME"] = model_dir
    os.environ["COQUI_TOS_AGREED"] = "1"

    print("Importing TTS...")

    from TTS.api import TTS
    from TTS.utils.manage import ModelManager

    def _disallowed_create_dir_and_download_model(*_args, **_kwargs):
        raise RuntimeError(
            f"Disallowed create_dir_and_download_model({_args}, {_kwargs}),"
            f" model {model_name} not found, model dir {model_dir} not valid? "
        )

    # patch to avoid any accidental download
    assert hasattr(ModelManager, "create_dir_and_download_model")
    ModelManager.create_dir_and_download_model = _disallowed_create_dir_and_download_model

    print("Loading TTS model...")
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.to(dev)

    # See tts.tts() func to how it generates audio.
    # This is a high-level wrapper, but we want to call it more directly to be able to use it in batch mode.
    print(f"{type(tts.synthesizer.tts_model)=}")
    print(f"{hasattr(tts.synthesizer.tts_model, "synthesize")=}")

    ...  # TODO...


if __name__ == "__main__":
    _demo()
