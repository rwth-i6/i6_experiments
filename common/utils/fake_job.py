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
        cls.__qualname__ = name + "(FakeJob)"
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

    def __reduce__(self):
        return _make_fake_job, (self.__class__.__module__, self.__class__.__name__, self.sis_hash)


def _make_fake_job(module: str, name: str, sis_hash: str) -> sisyphus.Job:
    """just for _FakeJobBase.__reduce__"""
    return make_fake_job(module=module, name=name, sis_hash=sis_hash)


_fake_job_class_cache = {}


def test_fake_job():
    from sisyphus import tk, gs
    from sisyphus.hash import short_hash
    import pickle

    job = make_fake_job(module="i6_core.audio.encoding", name="BlissChangeEncodingJob", sis_hash="vUdgDkgc97ZK")
    print(f"job: {job} {job.job_id()}/{gs.JOB_OUTPUT}")  # the latter is what tk.Path._sis_hash uses
    print(job.__class__, job.__class__.__qualname__, job.__class__.__name__, job.__class__.__module__)
    path = tk.Path("corpus.xml.gz", creator=job)
    print(f"path: {path} {short_hash(path)}")
    assert short_hash(path) == "6nofT6iiMd7G"

    path_ = pickle.loads(pickle.dumps(path))
    assert isinstance(path_, tk.Path)
    assert path == path_ and short_hash(path) == short_hash(path_)

    job_ = pickle.loads(pickle.dumps(job))
    assert isinstance(job_, _FakeJobBase)
    assert job.job_id() == job_.job_id()
