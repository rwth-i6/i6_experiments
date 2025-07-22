import functools
import os
from i6_experiments.users.zeyer.datasets.score_results import ScoreResult
from i6_experiments.users.zeyer.recog import ScoreResultCollection
from typing import Any, Dict, Tuple, Union, List
from i6_experiments.users.dorian_koch.jobs.AggregateOutputsAsCsv import WriteFinishedPathsAsCsv
from sisyphus import tk


def hook_collect_scores(
    score_results: dict[str, ScoreResult], original_collect_scores, collected_results: list
) -> ScoreResultCollection:
    # print(score_results)
    res = original_collect_scores(score_results)
    collected_results.append(res)
    return res


def hook_task(task, collected_results: list):
    original_collect_scores = task.collect_score_results_func
    if original_collect_scores is None:
        original_collect_scores = task.default_collect_score_results

    task.collect_score_results_func = functools.partial(
        hook_collect_scores, original_collect_scores=original_collect_scores, collected_results=collected_results
    )
    return original_collect_scores


def make_target_to_output_map(collected_results: list):
    from sisyphus import graph

    targs = graph.graph.targets
    target_to_output = {}
    for r in collected_results:
        for t in targs:
            if not isinstance(t, graph.OutputPath):
                continue
            t: graph.OutputPath
            if t._sis_path == r.output:
                # print(f"Found target {t.name}")
                target_to_output[t.name] = r.output
                # break
    return target_to_output


def hook_and_make_evals(func, task):
    collected_results = []
    original_collect_scores = hook_task(task, collected_results)

    func(task)

    task.collect_score_results_func = original_collect_scores
    return make_target_to_output_map(collected_results)


# these functions use the hook output:
def get_sclite_report_dirs(
    target_to_output: dict[str, Any], keys: Union[str, List[str]]
) -> Dict[str, Dict[str, tk.Path]]:
    if isinstance(keys, str):
        keys = [keys]

    report_dirs_dict = {}
    for k in target_to_output:
        if any([key not in k for key in keys]):
            continue
        assert k.endswith("score_results.txt"), f"Expected key to end with 'score_results.txt', got {k}"
        ctc_hyps = target_to_output[k]
        report_dirs = {}

        for ds, res in ctc_hyps.creator.score_results.items():
            report_dirs[ds] = res.report
        report_dirs_dict[k] = report_dirs

    return report_dirs_dict


def get_sclite_report_dir(
    target_to_output: dict[str, Any], keys: Union[str, List[str]]
) -> Tuple[str, Dict[str, tk.Path]]:
    l = get_sclite_report_dirs(target_to_output, keys)
    assert len(l) == 1, f"Expected exactly one report dir, got {len(l)}. Keys: {list(l.keys())}"
    return list(l.items())[0]


def write_sclite_report_dirs(target_to_output: dict[str, Any], keys: Union[str, List[str]], out_dir=None):
    for k, path_arr in get_sclite_report_dirs(target_to_output, keys).items():
        log_paths = WriteFinishedPathsAsCsv(inputs=list(path_arr.items()), seperator=": ")
        # TODO maybe just replace this with dirname
        base_pat = os.path.dirname(k)
        if out_dir is not None:
            base_pat = out_dir
        # base_pat = k[: -len("score_results.txt")]
        tk.register_output(
            f"{base_pat}/report_dirs.txt",
            log_paths.out_file,
        )
