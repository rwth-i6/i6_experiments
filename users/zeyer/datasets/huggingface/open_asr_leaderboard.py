import functools
from sisyphus import Job, Task, tk
from i6_core.util import uopen


def text_dict_normalize_file(text: tk.Path) -> tk.Path:
    """
    :param text: in textdict format
    :return: normalized text, in textdict format
    """
    from i6_core.returnn.search import SearchOutputRawReplaceJob

    text = SearchOutputRawReplaceJob(text, replacement_list=[(" c o ", " co ")]).out_search_results
    text = OpenASRLeaderboardTextNormalizationJob(
        text, open_asr_leaderboard_repo_dir=get_open_asr_leaderboard_repo_dir()
    ).out_text
    return text


@functools.cache
def get_open_asr_leaderboard_repo_dir() -> tk.Path:
    """
    :return: path to the open_asr_leaderboard repo
    """
    from i6_core.tools.git import CloneGitRepositoryJob

    open_asr_leaderboard_repo = CloneGitRepositoryJob(
        "https://github.com/huggingface/open_asr_leaderboard.git",
        # 2025-05-01 (includes parakeet-v2, phi4mi, ...)
        commit="2472403cae1434752b7448b8f7cda560bd549e0f",
    )
    tk.register_output("open_asr_leaderboard_repo", open_asr_leaderboard_repo.out_repository)
    return open_asr_leaderboard_repo.out_repository


@functools.cache
def download_esb_datasets_test_only_sorted() -> tk.Path:
    from i6_experiments.users.zeyer.external_models.huggingface import DownloadHuggingFaceRepoJobV2

    dl_esb_datasets_test_only_sorted = DownloadHuggingFaceRepoJobV2(
        repo_id="hf-audio/esb-datasets-test-only-sorted", repo_type="dataset"
    )
    tk.register_output("esb-datasets-test-only-sorted", dl_esb_datasets_test_only_sorted.out_hub_cache_dir)
    return dl_esb_datasets_test_only_sorted.out_hub_cache_dir


class OpenASRLeaderboardTextNormalizationJob(Job):
    """
    https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/data_utils.py
    """

    __sis_version__ = 2

    def __init__(self, text: tk.Path, *, open_asr_leaderboard_repo_dir: tk.Path):
        """
        :param text: e.g. via ExtractTextFromHuggingFaceEsbDatasetJob
        """
        super().__init__()
        self.text = text
        self.open_asr_leaderboard_repo_dir = open_asr_leaderboard_repo_dir

        self.rqmt = {"time": 4, "cpu": 2, "mem": 10}

        self.out_text = self.output_path("normalized.txt.py.gz")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import sys
        import types

        sys.path.insert(0, self.open_asr_leaderboard_repo_dir.get_path())

        # normalizer.eval_utils does `import evaluate`, which is not needed by us,
        # so do this hack to just ignore it, without needing the dependency
        sys.modules["evaluate"] = types.ModuleType("<dummy_evaluate>")

        # noinspection PyUnresolvedReferences
        from normalizer.data_utils import normalizer

        with uopen(self.text.get_path(), "rt") as in_:
            in_text = eval(in_.read())
        assert isinstance(in_text, dict)

        with uopen(self.out_text.get_path(), "wt") as out:
            out.write("{\n")
            for seq_tag, text in in_text.items():
                assert isinstance(text, str)
                norm_text = normalizer(text)
                out.write(f"{seq_tag!r}: {norm_text!r},\n")
            out.write("}\n")
