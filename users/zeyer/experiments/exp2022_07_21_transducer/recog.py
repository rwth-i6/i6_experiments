"""
recog helpers
"""


from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.search import ReturnnSearchJobV2, SearchBPEtoWordsJob
from returnn_common.datasets.interface import DatasetConfig
from returnn_common import nn
from .task import Task, ScoreResultCollection
from .model import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.base import RecogOutput
from i6_experiments.users.zeyer import tools_paths


def recog(task: Task, model: ModelWithCheckpoint) -> ScoreResultCollection:
    """recog"""
    outputs = {}
    for name, dataset in task.eval_datasets.items():
        recog_out = recog_dataset(dataset=dataset, model=model)
        score_out = task.score_recog_output_func(dataset, recog_out)
        outputs[name] = score_out
    return task.collect_score_results_func(outputs)


def recog_dataset(dataset: DatasetConfig, model: ModelWithCheckpoint) -> RecogOutput:
    """
    recog on the specific dataset
    """
    search_job = ReturnnSearchJobV2(
        search_data=dataset.get_main_dataset(),
        model_checkpoint=model.checkpoint,
        returnn_config=_search_config(model.definition),
        returnn_python_exe=tools_paths.get_returnn_python_exe(),
        returnn_root=tools_paths.get_returnn_root(),
    )
    bpe = search_job.out_search_file
    words = SearchBPEtoWordsJob(bpe).out_word_search_results
    return RecogOutput(output=words)


def _search_config(model_def) -> ReturnnConfig:
    pass  # TODO


def search(decoder, *, beam_size: int = 12) -> nn.Tensor:
    """search"""
    loop = nn.Loop(axis=decoder.align_spatial_dim)
    loop.max_seq_len = decoder.max_seq_len()
    loop.state = decoder.initial_state()
    with loop:
        log_prob, loop.state.decoder = decoder(loop.state.target, state=loop.state.decoder)
        loop.state.target = nn.choice(log_prob, input_type="log_prob", target=None, search=True, beam_size=beam_size)
        loop.end(decoder.end(loop.state.target, loop.state.decoder), include_eos=False)
        found = loop.stack(loop.state.target)
    return found
