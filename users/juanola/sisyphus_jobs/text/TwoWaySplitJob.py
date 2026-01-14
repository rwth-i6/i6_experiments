from typing import Optional

from sisyphus import Job, Path, Task


class TowWaySplitJob(Job):
    def __init__(
            self,
            text_file: Path,
            *,
            a_file_n_lines: Optional[int] = None,
            b_file_n_lines: Optional[int] = None,
            gzip: bool = True,
    ):
        """
        Splits text file into 2 new files (A and B). A or B file sizes have to be specified
        """
        self.text_file = text_file

        self.a_size = a_file_n_lines
        self.b_size = b_file_n_lines

        self.out_a = self.output_path("out_a.txt" + (".gz" if gzip else ""))
        self.out_b = self.output_path("out_b.txt" + (".gz" if gzip else ""))

        self.run_rqmt = {"cpu": 1, "mem": 12.0, "time": 6.0}

    def tasks(self):
        yield Task("run", rqmt=self.run_rqmt)

    def run(self):
        # Get length of text file
        num_lines = int(self.sh("zcat -f {text_file} | wc -l", True))

        if self.a_size is None and self.b_size is None:
            self.a_size = int(num_lines / 2)
            self.b_size = num_lines - self.a_size
        if self.a_size is None:
            self.a_size = num_lines - self.b_size
        elif self.b_size is None:
            self.b_size = num_lines - self.a_size

        assert self.a_size + self.b_size == num_lines

        # A file
        a_pipeline = "zcat -f {text_file} | head -n {a_size}"
        if self.out_a:
            a_pipeline += " | gzip"
        a_pipeline += " > {out_a}"

        self.sh(
            a_pipeline,
            except_return_codes=(141,),
        )

        # B file
        b_pipeline = "zcat -f {text_file} | tail -n {b_size}"
        if self.out_a:
            b_pipeline += " | gzip"
        b_pipeline += " > {out_b}"

        self.sh(
            b_pipeline,
            except_return_codes=(141,),
        )