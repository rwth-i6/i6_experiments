"""
recog helpers
"""


from sisyphus import tk
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import Checkpoint
from i6_core.returnn.search import ReturnnSearchJobV2, SearchBPEtoWordsJob
from returnn_common.datasets.interface import DatasetConfig
from .task import Task, ScoreResultCollection
from i6_experiments.users.zeyer.datasets.base import RecogOutput
from i6_experiments.users.zeyer import tools_paths


def recog_single(dataset: DatasetConfig, model_checkpoint: Checkpoint) -> RecogOutput:
    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model_checkpoint,
        returnn_config=_search_config(),
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    bpe = search_job.out_search_file
    words = SearchBPEtoWordsJob(bpe).out_word_search_results
    return RecogOutput(output=words)


def _search_config() -> ReturnnConfig:
    pass
