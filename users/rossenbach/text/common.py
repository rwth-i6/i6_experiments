from sisyphus import Job, Task

from i6_core.util import uopen


class PythonLineModificationJob(Job):
    """
    This is a generic Job templates for all jobs that simply change text file
    on a per-line basis
    """

    def __init__(self, text_file, compressed=True, mini_task=False):
        """

        :param Path text_file: raw or .gz text file
        """
        assert self.__class__ != PythonLineModificationJob
        self.text_file = text_file
        self.mini_task = mini_task

        self.out = self.output_path("out.txt.gz" if compressed else "out.txt")

        self.rqmt = {'cpu': 1, 'mem': 2, 'time': 1}

    def tasks(self):
        yield Task('run', rqmt=self.rqmt, mini_task=self.mini_task)

    def _process_line(self, line):
        """

        :param str line:
        :return: processed line
        :rtype: str
        """
        raise NotImplementedError

    def run(self):
        with uopen(self.text_file.get_path(), "rt") as infile, uopen(self.out.get_path(), "wt") as outfile:
            for line in infile.readline():
                outfile.write(self._process_line(line) + "\n")

    @classmethod
    def hash(cls, parsed_args):
        d = parsed_args.copy()
        d.pop("mini_task")
        return super().hash(d)