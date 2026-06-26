import ast

from i6_experiments.users.schmitt.experiments.exp2026_04_09_unsupervised_asr.sis_recipe.default_tools import (
    RETURNN_EXE,
    RETURNN_ROOT,
)
from .pipeline import (
    search_single,
    get_checkpoint,
)

from .visualization import VisualizeEmbeddingsJob



from typing import List, Optional, Dict, Any, List, Union, Iterator, Tuple, Callable
from dataclasses import dataclass, asdict
import copy

from sisyphus import tk, Task

from i6_core.returnn.training import ReturnnTrainingJob, PtCheckpoint, ReturnnConfig

from i6_experiments.users.schmitt.experiments.exp2025_08_14_speech_llms.recognition.aed.beam_search import (
    DecoderConfig,
)
from i6_experiments.users.zeyer.datasets.score_results import (
    ScoreResultCollection,
    join_score_results,
    ScoreResult,
)

from .data.common import TrainingDatasets
from .config import get_forward_config

default_returnn = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}


class SummarizeScoreResultsJob(tk.Job):
    def __init__(self, score_results: Dict[str, ScoreResultCollection]):
        super().__init__()
        self.score_results = score_results
        self.out_results_all_epochs_json = self.output_path("results_all_epoch.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import json

        with open(self.out_results_all_epochs_json.get_path(), "w") as f:
            f.write("{\n")
            count = 0
            for epoch, score in sorted(self.score_results.items()):
                assert isinstance(score, ScoreResultCollection)
                if count > 0:
                    f.write(",\n")
                res = json.load(open(score.output.get_path()))
                f.write(f'  "{epoch}": {json.dumps(res)}')
                count += 1
            f.write("\n}\n")


class SummarizeScoreResultsJobV2(tk.Job):
    def __init__(self, score_results: Dict[str, ScoreResultCollection]):
        super().__init__()
        self.score_results = score_results
        self.out_results_all_epochs_json = self.output_path("results_all_epoch.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import json
        import ast

        with open(self.out_results_all_epochs_json.get_path(), "w") as f:
            f.write("{\n")
            count = 0
            for epoch, score in sorted(self.score_results.items()):
                assert isinstance(score, ScoreResultCollection)
                if count > 0:
                    f.write(",\n")
                # res = json.load(open(score.output.get_path()))
                res = ast.literal_eval(open(score.output.get_path()).read())
                f.write(f"  {epoch}: {{\n")
                max_key_len = max(map(lambda x: len(x), res.keys()))
                for key, value in res.items():
                    key_str = key.ljust(max_key_len + 2)
                    f.write(f"    {key_str}| {value},\n")
                    f.write("    " + "-" * max_key_len + "--------------\n")
                f.write("  }")
                count += 1
            f.write("\n}\n")


class JoinScoreResultsJobV2(tk.Job):
    """
    Joins the score results of multiple jobs into one ScoreResultCollection.
    """

    def __init__(self, score_results: Dict[str, ScoreResult]):
        self.score_results = score_results
        self.out_score_results = self.output_path("score_results.json")
        self.out_score_results_table = self.output_path("score_results.txt")

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import ast
        import json

        res = {}
        max_key_len = max(map(lambda x: len(x), self.score_results.keys()))
        f = open(self.out_score_results_table.get_path(), "w")
        f.write("{\n")
        for key, score_result in self.score_results.items():
            # value_str = open(score_result.main_measure_value.get_path(), "r").read()
            value_str = score_result.main_measure_value.get()
            if isinstance(value_str, str):
                value_str = open(value_str).read()
                value = ast.literal_eval(value_str)
            else:
                assert isinstance(value_str, float), f"Unexpected type for main_measure_value: {type(value_str)}"
                value = f"{value_str:.2f}"

            res[key] = value
            key_str = key.ljust(max_key_len + 2)
            f.write(f"  {key_str}| {value},\n")
            f.write("  " + "-" * max_key_len + "--------------\n")
        f.write("}\n")
        f.close()

        with open(self.out_score_results.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")


def join_score_results_v2(
    score_results: Dict[str, ScoreResult], main_measure_key: Optional[str] = None
) -> ScoreResultCollection:
    """join score results"""
    return ScoreResultCollection(
        main_measure_value=score_results[main_measure_key].main_measure_value if main_measure_key else None,
        output=JoinScoreResultsJobV2(score_results).out_score_results,
        individual_results=score_results,
    )


def visualize_encoder_embeddings(
    config: Dict,
    training_name: str,
    recog_name: str,
    train_job: ReturnnTrainingJob,
    train_args: Dict[str, Any],
    train_data: TrainingDatasets,
    base_decoder_config: DecoderConfig,
    test_data_dict: Dict[str, Any],
    checkpoints: List[Union[int, str]],
    decoder_module: str,
    callback_module: str,
    loss_name: str = "dev_loss_ce",
    extra_forward_config: Optional[ReturnnConfig] = None,
    main_eval_measure_key: str = "dev",
    rqmt: Optional[Dict[str, int]] = None,
    recog_post_proc_funcs: Optional[List[Callable[[tk.Path], tk.Path]]] = None,
):
    graph_outputs_paht ="output/embedding_visualizations"

    job = VisualizeEmbeddingsJob()
    for index, file_name in enumerate(job.output_file_names): 
            tk.register_output(f"{graph_outputs_paht}/{index}-{file_name}", score_result.report)
    
    