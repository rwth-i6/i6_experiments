__all__ = ["ExperimentContext"]

from sisyphus import gs


class ExperimentContext:
    def __init__(self, dir_name: str) -> None:
        self.dir_name = dir_name
        self.orig_dir = None

    def __enter__(self) -> None:
        self.orig_dir = gs.ALIAS_AND_OUTPUT_SUBDIR
        gs.ALIAS_AND_OUTPUT_SUBDIR = self.dir_name

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        gs.ALIAS_AND_OUTPUT_SUBDIR = self.orig_dir
