"""
Copied from sis-setup-utils, adapted here.

Many functions are already moved elsewhere, e.g. to :mod:`sis_path`,
but this here is to ease the transition of other sis-setup-utils scripts.
"""

from __future__ import annotations
from typing import Optional, Any, List, Tuple, TextIO
from contextlib import contextmanager
from collections.abc import Mapping
import sys
import os
import re
from dataclasses import dataclass
from functools import reduce


_my_dir = os.path.dirname(__file__)
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_checked_setup_base_dir = False


def get_setup_base_dir() -> str:
    """
    :return: The setup base dir, where there is work and recipe.
    """
    global _checked_setup_base_dir
    if not _checked_setup_base_dir:
        assert os.path.exists(f"{_setup_base_dir}/work") and os.path.exists(f"{_setup_base_dir}/recipe")
        _checked_setup_base_dir = True
    return _setup_base_dir


def get_work_dir() -> str:
    from i6_experiments.users.zeyer.utils import sis_path

    return sis_path.get_work_dir()


def get_work_dir_prefix() -> str:
    from i6_experiments.users.zeyer.utils import sis_path

    return sis_path.get_work_dir_prefix()


def get_work_dir_prefix2() -> str:
    from i6_experiments.users.zeyer.utils import sis_path

    return sis_path.get_work_dir_prefix2()


def get_job_aliases(job: str) -> List[str]:
    """
    :param job: without "work/" prefix
    """
    from i6_experiments.users.zeyer.utils import job_aliases_from_info

    return job_aliases_from_info.get_job_aliases(job)


@contextmanager
def open_job_log(job: str, task: str = "run", index: int = 1) -> Tuple[Optional[TextIO], Optional[str]]:
    from i6_experiments.users.zeyer.utils import job_log

    with job_log.open_recent_job_log(job, task=task, index=index) as v:
        yield v


def get_inputs(job: str) -> list[str]:
    work_dir_prefix = get_work_dir_prefix()
    work_dir_prefix2 = get_work_dir_prefix2()
    inputs = []
    with open(work_dir_prefix + job + "/info") as f:
        for line in f.read().splitlines():
            key, value = line.split(": ", 1)
            if key != "INPUT":
                continue
            if not (value.startswith(work_dir_prefix) or value.startswith(work_dir_prefix2)):
                continue
            inputs.append(get_job_from_work_output(value))
    return inputs


def get_inputs_per_key(job: str) -> dict[str, str]:
    """
    param job: e.g. JoinScoreResultsJob
    """
    work_dir_prefix = get_work_dir_prefix()
    info_params = []
    with open(work_dir_prefix + job + "/info") as f:
        for line in f.read().splitlines():
            if line.startswith("PARAMETER: "):
                info_params.append(parse_info_parameter(line))
    return collect_inputs_per_key(info_params)


def get_job_from_work_output(filename: str, *, allow_none: bool = False) -> Optional[str]:
    work_dir_prefix = get_work_dir_prefix()
    work_dir_prefix2 = get_work_dir_prefix2()
    if filename.startswith(work_dir_prefix):
        filename = filename[len(work_dir_prefix) :]
    elif filename.startswith(work_dir_prefix2):
        filename = filename[len(work_dir_prefix2) :]
    elif filename.startswith("work/") and os.path.exists(get_setup_base_dir() + "/" + filename):
        filename = filename[len("work/") :]
    else:
        if "/output/" in filename:
            path = os.path.realpath(filename[: filename.rindex("/output/")])
            if is_job_dir(path):
                f = None
                while True:
                    f = path.rindex("/", None, f)
                    if f <= 0:
                        break
                    if os.path.realpath(get_work_dir_prefix() + path[f + 1 :]) == path:
                        return path[f + 1 :]
        if allow_none:
            return None
        raise ValueError(f"invalid {filename=}, not prefixed by {work_dir_prefix=} or {work_dir_prefix2=}")
    s = 0
    while True:
        f = filename.find("/", s)
        if f < 0:
            if allow_none:
                return None
            raise ValueError(f"No job found from {filename=}")
        if is_job_dir(filename[:f]):
            return filename[:f]
        s = f + 1


def is_job_dir(job: str) -> bool:
    """
    :param job: without "work/" prefix, or absolute
    """
    if job.startswith("/"):
        d = job
    else:
        d = get_work_dir_prefix() + job
    if not os.path.isdir(d):
        return False
    if not os.path.isfile(d + "/info"):
        return False
    if not os.path.isdir(d + "/output"):
        return False
    return True


def get_job_from_arg(job: str) -> str:
    """
    :param job: job path, job name, job output
    :return: job, such that is_job_dir(job) is True
    """
    work_dir_prefix = get_work_dir_prefix()
    if job.startswith("work/"):
        job = job[len("work/") :]
    elif job.startswith(work_dir_prefix):
        job = job[len(work_dir_prefix) :]
    elif job.startswith("output/"):
        job = get_job_from_work_output(os.path.realpath(job))
    elif job.startswith("Hub5ScoreJob."):  # small shortcut, special case
        job = "i6_core/recognition/scoring/" + job
    elif job.startswith("ReturnnSearchJobV2."):  # shortcut
        job = "i6_core/returnn/search/" + job
    assert os.path.exists(work_dir_prefix + job), f"job '{work_dir_prefix}{job}' does not exist"
    assert os.path.exists(work_dir_prefix + job + "/info")
    assert os.path.exists(work_dir_prefix + job + "/output")
    return job


def _get_sisyphus_dir() -> str:
    setup_base_dir = get_setup_base_dir()
    potential_dirs = []
    if os.path.islink(f"{setup_base_dir}/sis"):
        potential_dirs.append(os.path.dirname(os.readlink(f"{setup_base_dir}/sis")))
    potential_dirs.extend([f"{setup_base_dir}/tools/sisyphus", f"{setup_base_dir}/ext/sisyphus"])
    for d in potential_dirs:
        if (
            os.path.exists(f"{d}/sisyphus")
            and os.path.exists(f"{d}/sisyphus/__init__.py")
            and os.path.exists(f"{d}/sis")
        ):
            return d
    # Fallback: Try global import.
    import sisyphus

    return sisyphus.__file__


def _setup_sisyphus():
    """
    Some setup for Sisyphus. This is currently not used anymore.
    Setting SIS_GLOBAL_SETTINGS_FILE might still be useful.
    I'm not sure about _dummy_recog_training_exp, where this was needed/useful.
    """
    sis_dir = _get_sisyphus_dir()
    if sis_dir not in sys.path:
        sys.path.append(sis_dir)

    setup_base_dir = get_setup_base_dir()
    sys.path.append(setup_base_dir + "/recipe")
    os.environ["SIS_GLOBAL_SETTINGS_FILE"] = setup_base_dir + "/settings.py"

    import sisyphus  # noqa

    import i6_experiments.users.zeyer.recog

    i6_experiments.users.zeyer.recog.recog_training_exp = _dummy_recog_training_exp


def _dummy_recog_training_exp(prefix_name: str, task, model, recog_def, *args, **kwargs):
    from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoints

    assert isinstance(model, ModelWithCheckpoints)
    import sisyphus.toolkit as tk

    # ignore the recog jobs, register some dep on the train job
    tk.register_output(prefix_name + "/train-scores", model.scores_and_learning_rates)


def parse_info_parameter(info_line: str) -> ParsedInfoParameter:
    """
    :param info_line: line from Sis job info file
    """
    s = info_line
    assert s.startswith("PARAMETER: ")
    s = s[len("PARAMETER: ") :]
    name, s = s.split(": ", 1)
    if not s:
        return ParsedInfoParameter(name, None)
    if s.startswith("/") and (os.path.exists(s) or " " not in s):  # heuristic
        return ParsedInfoParameter(name, ParsedInfoParameterPath(s))
    # match to sth like "<i6_experiments.users.zeyer.model_interfaces.ModelDefWithCfg object at 0x7f01c870c490>"
    pattern = re.compile(r"<(\S+\.)?(\S+) object at 0x[0-9a-f]+>")
    if "<Path" in s or "<Variable" in s or s[:1] in "[({" or s.isnumeric() or pattern.search(s):  # heuristic
        s = re.sub(r"<Variable (\S+) (\S+)>", r"_Variable('\1', \2)", s)
        s = re.sub(r"<Path (\S+)>", r"_Path('\1')", s)
        s = pattern.sub("'\2'", s)
        try:
            content = eval(
                s, {"_Variable": ParsedInfoParameterVariable, "_Path": ParsedInfoParameterPath}, _AutoLocals()
            )
        except Exception as exc:
            raise ValueError("unhandled:\n" + s) from exc
        return ParsedInfoParameter(name, content)
    # unknown, random string, but does not really matter
    return ParsedInfoParameter(name, s)


@dataclass
class ParsedInfoParameter:
    """info parameter"""

    name: str
    content: Any


@dataclass
class ParsedInfoParameterVariable:
    """
    Variable

    like "<Variable work/i6_core/recognition/scoring/ScliteJob.HcIwh2mW7AZ6/output/wer 2.6>"
    """

    path: str
    value: Any


@dataclass
class ParsedInfoParameterPath:
    """
    Path

    <Path /u/zeyer/.../ScliteJob.aLj6yUUIDD93/output/reports>
    """

    path: str


def collect_inputs_per_key(info_params: list[ParsedInfoParameter]) -> dict[str, str]:
    """
    :param info_params: via :func:`parse_info_parameter`
    :return: path -> job
    """
    import tree

    d = {param.name: param.content for param in info_params}

    visited_jobs = set()
    jobs = {}  # path -> job

    def _visit_func(path, obj):
        if not isinstance(obj, (ParsedInfoParameterVariable, ParsedInfoParameterPath)):
            return
        job = get_job_from_work_output(obj.path, allow_none=True)
        if not job:
            return
        if job in visited_jobs:
            return
        visited_jobs.add(job)
        jobs[path] = job

    tree.map_structure_with_path(_visit_func, d)

    if not jobs:
        return jobs

    if len(jobs) > 1:
        # remove common prefix/postfix
        while True:
            paths = list(jobs.keys())
            if not paths[0]:
                break
            prefix = paths[0][0]
            if all(prefix == path[0] for path in paths):
                jobs = {path[1:]: job for path, job in jobs.items()}
                continue
            postfix = paths[0][-1]
            if all(postfix == path[-1] for path in paths):
                jobs = {path[:-1]: job for path, job in jobs.items()}
                continue
            break

    jobs_ = {"/".join(str(p) for p in path): job for path, job in jobs.items()}
    assert len(jobs_) == len(jobs)  # unique joined paths
    return jobs_


class _AutoLocals(Mapping[str, Any]):
    def __init__(self):
        self._locals = {}

    def __getitem__(self, item: str):
        if item in self._locals:
            return self._locals[item]
        if item[:1].isupper():
            value = type(item, (dict,), {})
            self[item] = value
            return value
        raise KeyError(f"auto locals: var {item} not found")

    def __setitem__(self, key: str, value):
        self._locals[key] = value

    def __contains__(self, item: str) -> bool:
        try:
            self.__getitem__(item)
            return True
        except KeyError:
            return False

    def __len__(self):
        return len(self._locals)

    def __iter__(self):
        return iter(self._locals)


def test_parse_info_parameter():
    s = (
        "PARAMETER: score_results:"
        " {'dev-clean': ScoreResult(dataset_name='dev-clean',"
        " main_measure_value=<Variable work/i6_core/recognition/scoring/ScliteJob.HcIwh2mW7AZ6/output/wer 2.6>,"
        " report=<Path /u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
        "ScliteJob.HcIwh2mW7AZ6/output/reports>),"
        " 'dev-other': ScoreResult(dataset_name='dev-other',"
        " main_measure_value=<Variable work/i6_core/recognition/scoring/ScliteJob.aLj6yUUIDD93/output/wer 5.4>,"
        " report=<Path /u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
        "ScliteJob.aLj6yUUIDD93/output/reports>),"
        " 'test-clean': ScoreResult(dataset_name='test-clean',"
        " main_measure_value=<Variable work/i6_core/recognition/scoring/ScliteJob.IJ1p9FdPgzgi/output/wer 2.81>,"
        " report=<Path /u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
        "ScliteJob.IJ1p9FdPgzgi/output/reports>),"
        " 'test-other': ScoreResult(dataset_name='test-other',"
        " main_measure_value=<Variable work/i6_core/recognition/scoring/ScliteJob.OcYJiwXkWAUI/output/wer 6.5>,"
        " report=<Path /u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
        "ScliteJob.OcYJiwXkWAUI/output/reports>)}"
    )
    param = parse_info_parameter(s)
    print(param)
    assert param.name == "score_results"
    assert isinstance(param.content, dict)
    assert set(param.content) == {"dev-clean", "dev-other", "test-clean", "test-other"}
    for k, scores in param.content.items():
        assert isinstance(scores, dict) and type(scores).__name__ == "ScoreResult"
        assert set(scores) == {"dataset_name", "main_measure_value", "report"}
        assert scores["dataset_name"] == k
        measure = scores["main_measure_value"]
        assert isinstance(measure, ParsedInfoParameterVariable)
        assert measure.path.startswith("work/i6_core/") and measure.path.endswith("/output/wer")
        assert measure.value == {"dev-clean": 2.6, "dev-other": 5.4, "test-clean": 2.81, "test-other": 6.5}[k]
        report = scores["report"]
        assert isinstance(report, ParsedInfoParameterPath)
        assert report.path.startswith("/u/zeyer/setups/") and report.path.endswith("/output/reports")


def test_collect_inputs_per_key():
    param = ParsedInfoParameter(
        name="score_results",
        content={
            "dev-clean": {
                "dataset_name": "dev-clean",
                "main_measure_value": ParsedInfoParameterVariable(
                    path="work/i6_core/recognition/scoring/ScliteJob.HcIwh2mW7AZ6/output/wer", value=2.6
                ),
                "report": ParsedInfoParameterPath(
                    path="/u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
                    "ScliteJob.HcIwh2mW7AZ6/output/reports"
                ),
            },
            "dev-other": {
                "dataset_name": "dev-other",
                "main_measure_value": ParsedInfoParameterVariable(
                    path="work/i6_core/recognition/scoring/ScliteJob.aLj6yUUIDD93/output/wer", value=5.4
                ),
                "report": ParsedInfoParameterPath(
                    path="/u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
                    "ScliteJob.aLj6yUUIDD93/output/reports"
                ),
            },
            "test-clean": {
                "dataset_name": "test-clean",
                "main_measure_value": ParsedInfoParameterVariable(
                    path="work/i6_core/recognition/scoring/ScliteJob.IJ1p9FdPgzgi/output/wer", value=2.81
                ),
                "report": ParsedInfoParameterPath(
                    path="/u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
                    "ScliteJob.IJ1p9FdPgzgi/output/reports"
                ),
            },
            "test-other": {
                "dataset_name": "test-other",
                "main_measure_value": ParsedInfoParameterVariable(
                    path="work/i6_core/recognition/scoring/ScliteJob.OcYJiwXkWAUI/output/wer", value=6.5
                ),
                "report": ParsedInfoParameterPath(
                    path="/u/zeyer/setups/combined/2021-05-31/work/i6_core/recognition/scoring/"
                    "ScliteJob.OcYJiwXkWAUI/output/reports"
                ),
            },
        },
    )

    jobs = collect_inputs_per_key([param])
    assert set(jobs) == {"dev-clean", "dev-other", "test-clean", "test-other"}
    assert all("/ScliteJob." in job for job in jobs.values())


# _setup_sisyphus()
