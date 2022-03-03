import os
import subprocess as sp

from sisyphus import Job, Task, tk

class DirectoryZipJob(Job):
    """
    Archives and potentially compresses a directory using zip
    """
    def __init__(
        self, dir_path, compression_level=0, command_line_args=None,
    ):
        """
        :param tk.Path dir_path: path to directory to be zipped
        :param int compression_level: compression level for zipping, see zip documentation for
            detailed description of possible compressions
        :param list|None command_line_args: Additional command line arguments for configuring zip task
        """
        self.dir_path = dir_path
        self.compression_level = compression_level
        assert 0 <= self.compression_level <= 9 and type(self.compression_level) == int
        self.command_line_args = command_line_args or []

        self.dir_name = os.path.basename(os.path.dirname(self.dir_path.get_path()))

        self.out_zip_path = self.output_path(f"{self.dir_name}.zip")

        self.rqmt = {"time": 8, "mem": 16, "cpu": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        run_cmd = [
            "zip", "-r", "-q", "-j", f"-{self.compression_level}", self.out_zip_path.get_path(), self.dir_path.get_path(),
        ]
        run_cmd += self.command_line_args

        try:
             sp.check_call(run_cmd)
        except sp.CalledProcessError:
            print(f"Error occurred in subprocess: Zipping directory failed for {self.dir_path.get_path()}")
            raise
