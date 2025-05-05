"""
Generic HuggingFace model downloader
"""

from sisyphus import Job, Task


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
