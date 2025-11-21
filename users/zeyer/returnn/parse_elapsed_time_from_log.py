import os
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.utils.job_dir import get_job_base_dir
from i6_experiments.users.zeyer.utils.job_log import open_recent_job_log


class ParseElapsedTimeFromReturnnLogJob(Job):
    def __init__(self, *, returnn_job_output_file: tk.Path):
        """
        :param returnn_job_output_file: any output from this RETURNN job
        """
        super().__init__()

        self.returnn_job_output_file = tk.Path(".", creator=returnn_job_output_file.creator)

        self.out_time_secs = self.output_var("time_secs.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # Parse this:
        # print(("elapsed: %s" % hms_fraction(time.time() - start_time)), file=log.v3)

        print("*** Job output file:", self.returnn_job_output_file.get_path())
        job_dir = os.path.dirname(os.path.normpath(self.returnn_job_output_file.get_path()))
        print("*** Job dir:", job_dir)
        print("*** Job base dir:", get_job_base_dir(job_dir))
        with open_recent_job_log(job_dir, as_text=False) as (log_file, log_filename):
            print("*** Log filename:", log_filename)
            assert log_file is not None

            # Parse last part of file

            log_file.seek(0, os.SEEK_END)
            file_size = log_file.tell()
            seek_size = min(file_size, 100000)
            log_file.seek(file_size - seek_size, os.SEEK_SET)
            lines = log_file.readlines()

            elapsed_time_secs = None
            for line in reversed(lines):
                if b"elapsed:" in line:
                    # Example line: elapsed: 0:05:23.456789
                    parts = line.split(b"elapsed:")[-1].strip().split(":")
                    if len(parts) == 3:
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = float(parts[2])
                        elapsed_time_secs = hours * 3600 + minutes * 60 + seconds
                        break

            if elapsed_time_secs is None:
                raise RuntimeError("Could not find elapsed time in log file.")

            self.out_time_secs.set(elapsed_time_secs)
