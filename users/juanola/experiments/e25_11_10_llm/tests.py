from sisyphus import tk

from i6_core.text import WriteToTextFileJob
from .default_tools import RETURNN_EXE, RETURNN_ROOT
from ...sisyphus_jobs.text.ShuffleJob import ShuffleJob
from ...sisyphus_jobs.text.TwoWaySplitJob import TowWaySplitJob

ROOT_RETURNN_ROOT = {
    "returnn_exe": RETURNN_EXE,
    "returnn_root": RETURNN_ROOT,
}

TEST_STRING = """hello, first line
second
bla bla

test
hehe
more more
"""

def shuffle_file_test(test_name = "shuffle_file_test"):
    write_job = WriteToTextFileJob(TEST_STRING)
    shuffle_job = ShuffleJob(write_job.out_file)
    tk.register_output(f"test/{test_name}", shuffle_job.out)

def two_way_split_file_test(test_name = "two_way_split_file_test"):
    write_job = WriteToTextFileJob(TEST_STRING)
    split_job = TowWaySplitJob(write_job.out_file, a_file_n_lines=2)
    tk.register_output(f"test/{test_name}", split_job.out_a)

