"""
Generic HuggingFace model downloader
"""

from typing import Union, Optional, List
import os
import shutil
import copy

import huggingface_hub

from sisyphus import tk, Job, Task

from i6_experiments.users.zeyer.external_models.huggingface import get_content_dir_from_hub_cache_dir


class DownloadHuggingFaceRepoJob(Job):
    """
    Download a model from HuggingFace.
    This basically does ``huggingface-cli download <model_id>``.

    Requires the huggingface_hub library to be installed.
    """

    def __init__(self, *, model_id: str, file_list: Optional[List[str]] = None, require_login: bool = False):
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
        self.file_list = file_list
        self.require_login = require_login

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

        if self.require_login:
            from huggingface_hub import login

            login()

        import huggingface_hub

        if hasattr(huggingface_hub, "commands"):
            from huggingface_hub.commands.download import DownloadCommand

            parser = ArgumentParser("huggingface-cli", usage="huggingface-cli <command> [<args>]")
            commands_parser = parser.add_subparsers(help="huggingface-cli command helpers")
            DownloadCommand.register_subcommand(commands_parser)
            args = ["download", self.model_id]
            if self.file_list:
                assert isinstance(self.file_list, list)
                args += self.file_list

            args = parser.parse_args(args)
            service = args.func(args)
            print(service)
            service.run()
        else:
            assert hasattr(huggingface_hub, "snapshot_download")
            from huggingface_hub import snapshot_download

            kwargs = dict(
                repo_id=self.model_id,
                # repo_type=repo_type,
                # revision=revision,
                # local_dir=output_dir,
                token=True if self.require_login else None,
            )

            # Newer huggingface_hub versions removed this arg.
            # Older ones used it to avoid symlinks in local_dir.
            import inspect

            if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
                kwargs["local_dir_use_symlinks"] = False

            if self.file_list:
                assert isinstance(self.file_list, list)
                kwargs["allow_patterns"] = self.file_list

            path = snapshot_download(**kwargs)

    @classmethod
    def hash(cls, kwargs):
        d = copy.deepcopy(kwargs)
        if not d.get("require_login", False):
            d.pop("require_login", None)

        return super().hash(d)


class DownloadHuggingFaceRepoJobV2(DownloadHuggingFaceRepoJob):
    """
    Additionally to V1, this adds a symlink to the actual content directory of the cache.
    """

    def __init__(self, *, model_id: str, file_list: Optional[List[str]] = None):
        """
        :param model_id: e.g. "CohereLabs/aya-expanse-32b" or so

        Note for token auth:
        It will use the standard HF methods to determine the token.
        E.g. it will look for the HF_TOKEN env var,
        or it will look into the HF home dir (set via HF_HOME env, or as default ~/.cache/huggingface).
        Do ``python -m huggingface_hub.commands.huggingface_cli login``.
        See HF :func:`get_token`.
        """
        super().__init__(model_id=model_id, file_list=file_list)
        self.out_content_dir = self.output_path("content")

        if file_list and len(file_list) == 1:
            self.out_single_file = self.output_path("file")
        self.file_list = file_list

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)
        yield Task("symlink", mini_task=True)

    def symlink(self):
        content_dir = get_content_dir_from_hub_cache_dir(self.out_hub_cache_dir)
        os.symlink(content_dir, self.out_content_dir)

        if hasattr(self, "out_single_file"):
            os.symlink(os.path.join(content_dir, self.file_list[0]), self.out_single_file)


class GetContentDirFromHfCacheDirJob(Job):
    """
    Download a model from HuggingFace.
    This basically does ``huggingface-cli download <model_id>``.

    Requires the huggingface_hub library to be installed.
    """

    def __init__(self, *, hub_cache_dir: tk.Path):
        super().__init__()
        self.hub_cache_dir = hub_cache_dir
        self.out_content_dir = self.output_path("content_dir", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        content_dir = get_content_dir_from_hub_cache_dir(self.hub_cache_dir)
        shutil.rmtree(self.out_content_dir.get_path())
        shutil.copytree(content_dir, self.out_content_dir.get_path())
