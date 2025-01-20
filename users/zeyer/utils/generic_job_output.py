"""
Convert any given path like
    "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
into a Sisyphus tk.Path,
but in a way that it is keeps the has just as if it would be the output of the corresponding job.
"""

from __future__ import annotations

import importlib

from sisyphus import Job
from sisyphus import tk

from i6_experiments.common.utils.fake_job import make_fake_job


def generic_job_output(filename: str) -> tk.Path:
    """
    Convert any given path like
        "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
    into a Sisyphus tk.Path,
    but in a way that it is keeps the has just as if it would be the output of the corresponding job.

    :param filename: any path like
        "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
    :return: Sisyphus tk.Path with original job output hash
    """

    parts = filename.split("/")
    basename_idx = min([i for i in range(len(parts)) if "." in parts[i]])
    module_s = ".".join(parts[:basename_idx])
    cls_name, sis_hash = parts[basename_idx].split(".", 1)

    # The import is not really necessary here, but just do it anyway as a sanity check.
    module = importlib.import_module(module_s)
    cls = getattr(module, cls_name)
    assert issubclass(cls, Job)

    fake_job = make_fake_job(module=module_s, name=cls_name, sis_hash=sis_hash)
    expected_sis_id = (module_s + "." + cls_name).replace(".", "/") + "." + sis_hash
    # noinspection PyProtectedMember
    assert fake_job._sis_id() == expected_sis_id, (
        f"fake job {fake_job} with sis_id {fake_job._sis_id()} does not match expected sis_id {expected_sis_id},"
        f" module {module_s}, class {cls_name}, sis_hash {sis_hash}"
    )
    assert parts[basename_idx + 1] == "output"
    path = tk.Path("/".join(parts[basename_idx + 2 :]), creator=fake_job)
    return path
