"""
Make fake jobs: jobs which have same module, name and hash as a real job.

This makes sense when the output is already computed,
and the remaining graph is not needed.
"""

import sisyphus


def make_fake_job(*, module: str, name: str, sis_hash: str) -> sisyphus.Job:
    """
    We want to make a fake job for the creator of tk.Path, such that the dependency_boundary files
    are properly registered as job outputs.
    """
    cls = _fake_job_class_cache.get((module, name))
    if not cls:

        class _FakeJob(_FakeJobBase):
            pass

        cls = _FakeJob
        # Fake these attributes so that JobSingleton.__call__ results in the same sis_id.
        cls.__module__ = module
        cls.__name__ = name
        _fake_job_class_cache[(module, name)] = cls
    job = cls(sis_hash=sis_hash)
    # Note: If this isinstance(...) is not true,
    # it's because the job was already registered before with the real job class,
    # and the JobSingleton returned that real instance.
    # Anyway, this should not really be a problem.
    if isinstance(job, cls):
        # Do not keep the fake job instances registered, in case we later want to create the real instance.
        # noinspection PyProtectedMember
        sisyphus.job.created_jobs.pop(job._sis_id())
    return job


class _FakeJobBase(sisyphus.Job):
    # noinspection PyShadowingNames
    def __init__(self, *, sis_hash: str):
        super().__init__()
        self.sis_hash = sis_hash
        # Make sure our outputs are never cleaned up.
        self.set_keep_value(99)

    @classmethod
    def hash(cls, parsed_args):
        """
        Sisyphus job hash
        """
        return parsed_args["sis_hash"]


_fake_job_class_cache = {}
