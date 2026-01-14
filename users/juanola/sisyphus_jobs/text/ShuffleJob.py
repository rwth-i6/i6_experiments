import random
from sisyphus import Job, Path, Task
import i6_core.util as util


class ShuffleJob(Job):
    """
    Shuffle lines of text file.
    https://stackoverflow.com/questions/4618298/randomly-mix-lines-of-3-million-line-file
    """

    def __init__(
            self,
            text_file: Path,
            *,
            gzip: bool = True,
            seed: int = 42,
    ):
        self.text = text_file
        self.seed = seed

        self.out = self.output_path("out.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        lines = util.uopen(self.text, "rt").readlines()
        random.Random(self.seed).shuffle(lines)
        util.uopen(self.out, "wt").writelines(lines)
