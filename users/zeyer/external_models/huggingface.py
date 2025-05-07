"""
Generic HuggingFace model downloader
"""

from typing import Union
import os
from sisyphus import tk, Job, Task


class DownloadHuggingFaceRepoJob(Job):
    """
    Download a model from HuggingFace.
    This basically does ``huggingface-cli download <model_id>``.

    Requires the huggingface_hub library to be installed.
    """

    def __init__(self, *, model_id: str):
        """
        :param model_id: e.g. "CohereLabs/aya-expanse-32b" or so

        Note for token auth:
        It will use the standard HF methods to determine the token.
        E.g. it will look for the HF_TOKEN env var,
        or it will look into the HF home dir (set via HF_HOME env, or as default ~/.cache/huggingface).
        Do ``python -m huggingface_hub.commands.huggingface_cli login``.
        See HF :func:`get_token`.
        """
        super().__init__()
        self.model_id = model_id
        self.rqmt = {"time": 4, "cpu": 2, "mem": 8}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        from argparse import ArgumentParser

        # Keep HF_HOME untouched such that it might use the token from there.
        hub_cache_dir = self.out_hub_cache_dir.get_path()
        os.environ["HF_HUB_CACHE"] = hub_cache_dir
        print("HF_HUB_CACHE:", hub_cache_dir)

        print("Import huggingface_hub CLI...")

        from huggingface_hub.commands.download import DownloadCommand

        parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = parser.add_subparsers(help="huggingface-cli command helpers")
        DownloadCommand.register_subcommand(commands_parser)
        args = parser.parse_args(["download", self.model_id])
        service = args.func(args)
        print(service)
        service.run()


class DownloadHuggingFaceRepoJobV2(Job):
    """
    Download a repo from HuggingFace.
    This basically does ``huggingface-cli download <model_id>``.

    Requires the huggingface_hub library to be installed.
    """

    def __init__(self, *, repo_id: str, repo_type: str):
        """
        :param repo_id: e.g. "CohereLabs/aya-expanse-32b" or so
        :param repo_type: `model`, `dataset` or `space`

        Note for token auth:
        It will use the standard HF methods to determine the token.
        E.g. it will look for the HF_TOKEN env var,
        or it will look into the HF home dir (set via HF_HOME env, or as default ~/.cache/huggingface).
        Do ``python -m huggingface_hub.commands.huggingface_cli login``.
        See HF :func:`get_token`.
        """
        super().__init__()
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.rqmt = {"time": 4, "cpu": 2, "mem": 8}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        from argparse import ArgumentParser

        # Keep HF_HOME untouched such that it might use the token from there.
        hub_cache_dir = self.out_hub_cache_dir.get_path()
        os.environ["HF_HUB_CACHE"] = hub_cache_dir
        print("HF_HUB_CACHE:", hub_cache_dir)

        print("Import huggingface_hub CLI...")

        from huggingface_hub.commands.download import DownloadCommand

        parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
        commands_parser = parser.add_subparsers(help="huggingface-cli command helpers")
        DownloadCommand.register_subcommand(commands_parser)
        args = parser.parse_args(["download", self.repo_id, "--repo-type", self.repo_type])
        service = args.func(args)
        print(service)
        service.run()


def get_model_dir_from_hub_cache_dir(hub_cache_dir: Union[tk.Path, str]):
    """
    Use this inside your job.
    """
    if isinstance(hub_cache_dir, tk.Path):
        hub_cache_dir = hub_cache_dir.get_path()
    assert isinstance(hub_cache_dir, str) and os.path.isdir(hub_cache_dir)
    cache_content = [fn for fn in os.listdir(hub_cache_dir) if not fn.startswith(".")]
    assert cache_content, f"empty cache dir {hub_cache_dir}"
    assert len(cache_content) == 1, f"cache dir {hub_cache_dir} has multiple entries: {cache_content}"
    model_dir = hub_cache_dir + "/" + cache_content[0]
    refs_content = os.listdir(model_dir + "/refs")
    assert len(refs_content) == 1, f"refs dir {model_dir}/refs has not a single entry but {refs_content}"
    ref = open(model_dir + "/refs/" + refs_content[0]).read().strip()
    assert ref
    snapshot_dir = model_dir + "/snapshots/" + ref
    assert os.path.isdir(snapshot_dir)
    return snapshot_dir
