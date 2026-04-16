from sisyphus import Job, Task
from typing import Iterator
class DummyJob(Job):
    """
    Keep Sis running for debug
    """
    def __init__(
        self,
        *,
        version: int = 1,
    ):
        """
        :param model: modelwithcheckpoints, all fixed checkpoints + scoring file for potential other relevant checkpoints (see update())
        :param recog_and_score_func: epoch -> scores. called in graph proc
        """
        super(DummyJob, self).__init__()
        self.version = version
        self.output = self.output_var("run_time")

    def tasks(self) -> Iterator[Task]:
        """tasks"""
        yield Task("run", rqmt={"cpu":1, "time":2})#mini_task=True)

    def run(self):
        """run"""
        import time
        start = time.time()
        while time.time() < start + 60*60:
            time.sleep(10)  # sleep for 10s to reduce CPU usage
        self.output.set(time.time()-start)