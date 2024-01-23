

def dump_runnable_jobs_with_dep_paths():
    """
    Iterate through all targets towards runnable jobs, and print the dependency path back to the target.
    This is useful to find out why a job is runnable (supposed to run by manager).
    """
    from sisyphus import tk, Job

    visited = set()
    queue = [((target._sis_path,), target._sis_path.creator) for target in tk.sis_graph.targets]
    while queue:
        dep_path, job = queue.pop()
        job: Job
        if job in visited or job is None:
            continue
        visited.add(job)
        if job._sis_finished():
            # Clear inputs so that all dependent jobs are ignored,
            # and would not rerun, even if they are not finished or existing.
            job._sis_inputs.clear()
            continue
        if job._sis_runnable():
            print("unfinished and runnable:", job, "via:", dep_path)
        queue.extend((dep_path + (path,), path.creator) for path in job._sis_inputs)
