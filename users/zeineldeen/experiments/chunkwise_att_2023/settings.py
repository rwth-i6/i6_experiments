import getpass
import sys
import os

sys.path.append("/u/beck/dev/cachemanager/")

CPU_SHORT_JOBLIST = ["AverageTFCheckpointsJob", "GetBestTFCheckpointJob"]

def check_engine_limits(current_rqmt, task):
    current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
    curr_sbatch_args = current_rqmt.get("sbatch_args", [])
    if "-p" not in current_rqmt.get("sbatch_args", []):
      if current_rqmt.get("gpu", 0) > 0:
        # gpu 
        if current_rqmt.get("gpu_test", False):
          current_rqmt["sbatch_args"] = ["-p", "gpu_test_24gb"] + curr_sbatch_args
        elif current_rqmt.get("gpu_mem", 0) > 11:
          current_rqmt["sbatch_args"] = ["-p", "gpu_24gb"] + curr_sbatch_args
        else:
          current_rqmt["sbatch_args"] = ["-p", "gpu_11gb"] + curr_sbatch_args
      elif current_rqmt.get("cpu_type", None):
        assert current_rqmt["cpu_type"] in ["cpu_slow", "cpu_fast", "cpu_short"]
        current_rqmt["sbatch_args"] = ["-p", current_rqmt["cpu_type"]]
      else:
        # cpu with SSE4 and AVX 
        if task._job.__class__.__name__ in CPU_SHORT_JOBLIST:
          current_rqmt["sbatch_args"] = ["-p", "cpu_short"] + curr_sbatch_args
    return current_rqmt


def file_caching(path):
    return "`cf %s`" % path


def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.simple_linux_utility_for_resource_management_engine import (
        SimpleLinuxUtilityForResourceManagementEngine,
    )

    temp_exclude = [281, 282, 283, 284, 285]

    default_rqmt={"cpu": 1, "mem": 4, "time": 1}
    if temp_exclude:
      default_rqmt["sbatch_args"] = ["-x", ",".join([f"cn-{node}" for node in temp_exclude])]

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=4),
            "long": SimpleLinuxUtilityForResourceManagementEngine(default_rqmt=default_rqmt),
        },
        default_engine="long",
    )


def update_engine_rqmt(initial_rqmt, last_usage):
    """Update requirements after a job got interrupted.

    This is modified from the default `update_engine_rqmt`
    in that it only updates the time requirement but not the memory requirement.

    :param dict[str] initial_rqmt: requirements that are requested first
    :param dict[str] last_usage: information about the last usage by the task
    :return: updated requirements
    :rtype: dict[str]
    """

    # Contains the resources requested for interrupted run
    requested_resources = last_usage.get("requested_resources", {})
    requested_time = requested_resources.get("time", initial_rqmt.get("time", 1))

    # How much was actually used
    used_time = last_usage.get("used_time", 0)

    # Did it (nearly) break the limits?
    out_of_time = requested_time - used_time < 0.1

    # Double limits if needed
    if out_of_time:
        requested_time = max(initial_rqmt.get("time", 0), requested_time * 2)

    # create updated rqmt dict
    out = initial_rqmt.copy()
    out.update(requested_resources)
    out["time"] = requested_time

    return out


MAIL_ADDRESS = getpass.getuser()

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"
VERBOSE_TRACEBACK_TYPE = "better_exchook"
JOB_AUTO_CLEANUP = False
START_KERNEL = False
SHOW_JOB_TARGETS = False
PRINT_ERROR = False

RASR_ROOT = "/work/tools/asr/rasr/20220603_github_default/"
RASR_ARCH = "linux-x86_64-standard"

SCTK_PATH = "/u/beck/programs/sctk-2.4.0/bin/"
G2P_PATH = "/u/beck/dev/g2p/release/lib/python/g2p.py"
G2P_PYTHON = "python2.7"
SUBWORD_NMT = "https://github.com/albertz/subword-nmt.git"

RETURNN_ROOT = "/u/zeineldeen/dev/returnn"

# set in .bashrc
DEFAULT_ENVIRONMENT_KEEP = {
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "PWD",
    "SGE_STDERR_PATH",
    "SGE_TASK_ID",
    "TMP",
    #"TMPDIR",
    "USER",
    "LD_LIBRARY_PATH",
}

DEFAULT_ENVIRONMENT_SET["NUMBA_CACHE_DIR"] = "/var/tmp/numba_cache_{}/".format(os.environ["USER"])  # for librosa

# related: https://stackoverflow.com/questions/48610132/tensorflow-crash-with-cudnn-status-alloc-failed
DEFAULT_ENVIRONMENT_SET["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

TMP_PREFIX = "/var/tmp/"
DEFAULT_ENVIRONMENT_SET["TMPDIR"] = TMP_PREFIX


WAIT_PERIOD_CACHE = 10
WAIT_PERIOD_JOB_FS_SYNC = 10
WAIT_PERIOD_BETWEEN_CHECKS = 10
