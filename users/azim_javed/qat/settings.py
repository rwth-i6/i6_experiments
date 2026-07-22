import getpass
import os
import sys

sys.path.append("/u/beck/dev/cachemanager/")


def file_caching(path, is_output=False):
    if is_output:
        return "`cf -d %s`" % path
    else:
        return "`cf %s`" % path


CPU_SLOW_JOBLIST = [
    "ScliteJob",
    "Hub5ScoreJob",
    "PipelineJob",
]


def check_engine_limits(current_rqmt, task):
    """
    i6 support for gpu_mem
    """
    current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
    bad_nodes = ["cn-508"]
    if current_rqmt.get("gpu", 0) > 0 and "-p" not in current_rqmt.get(
        "sbatch_args", []
    ):
        if current_rqmt.get("gpu_mem", 0) > 24:
            current_rqmt["sbatch_args"] = ["-p", "gpu_48gb,gpu_24gb"]
        elif current_rqmt.get("gpu_mem", 0) > 11:
            current_rqmt["sbatch_args"] = ["-p", "gpu_24gb"]
        else:
            current_rqmt["sbatch_args"] = ["-p", "gpu_11gb"]
    current_rqmt["sbatch_args"] = current_rqmt.get("sbatch_args", []) + [
        "--exclude=%s" % ",".join(bad_nodes)
    ]
    if task._job.__class__.__name__ in CPU_SLOW_JOBLIST:
        current_rqmt["sbatch_args"] = ["-p", "cpu_slow"]

    return current_rqmt


def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.simple_linux_utility_for_resource_management_engine import (
        SimpleLinuxUtilityForResourceManagementEngine,
    )

    default_rqmt = {
        "cpu": 1,
        "mem": 4,
        "gpu": 0,
        "time": 1,
    }

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=4, mem=16),
            "long": SimpleLinuxUtilityForResourceManagementEngine(
                default_rqmt=default_rqmt
            ),
        },
        default_engine="long",
    )


def worker_wrapper(job, task_name, call):
    # image = "/work/asr4/berger/apptainer/images/torch-2.8_onnx-1.22.sif"
    image = "/work/asr4/hilmes/apptainer/torch-2.8_onnx-1.22_v3.sif"

    binds = [
        "/u",
        "/work/smt4",
        "/work/asr4",
        "/work/common",
        # "/work/tools/",
        # "/u/corpora",
        "/run",
    ]

    app_call = [
        "apptainer",
        "exec",
        "--env",
        f"NUMBA_CACHE_DIR=/var/tmp/numba_cache_{getpass.getuser()}",
        "--pwd",
        os.environ.get("PWD", os.getcwd()),
        "--nv",
    ]

    for path in binds:
        app_call += ["-B", path]

    app_call += [image, "python3"]

    app_call += call[1:]

    return app_call


# need to run bob for ivector
SIS_COMMAND = ["python3", sys.argv[0]]

WAIT_PERIOD_CACHE = 1  # stopping to wait for actionable jobs to appear
WAIT_PERIOD_JOB_FS_SYNC = 1  # finishing a job

JOB_AUTO_CLEANUP = False
JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"

SHOW_JOB_TARGETS = False
SHOW_VIS_NAME_IN_MANAGER = False
PRINT_ERROR = False
PRINT_STALE_STATE_OVERVIEW_PERIOD = 1800

MAIL_ADDRESS = None

WORK_DIR = "/work/smt4/azim.javed/sisyphus_work"


DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join(
    [
        # "/usr/local/lib/tensorflow/",
        # "/usr/local/lib/python3.8/dist-packages/tensorflow/",
        # "/usr/local/lib/python3.8/dist-packages/scipy/.libs",
        # "/usr/local/lib/python3.8/dist-packages/numpy.libs",
        # "/usr/local/lib/python3.10/dist-packages/tensorflow/",
        # "/usr/local/lib/python3.10/dist-packages/scipy/.libs",
        # "/usr/local/lib/python3.10/dist-packages/numpy.libs",
        "/.singularity.d/libs",
        "/usr/local/cuda/extras/CUPTI/lib64",
        "/usr/local/cuda/compat/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
    ]
)

TMP_PREFIX = "/var/tmp/"
DEFAULT_ENVIRONMENT_SET.update(
    {
        "TMPDIR": TMP_PREFIX,
        "TMP": TMP_PREFIX,
        "PYTHONPATH": os.environ.get("PWD", os.getcwd()) + ":/work/smt4/azim.javed/repositories/" + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""),
        "NUMBA_CACHE_DIR": f"{TMP_PREFIX}/numba_cache_{getpass.getuser()}",  # used for librosa
        "PYTORCH_KERNEL_CACHE_PATH": f"{TMP_PREFIX}/pt_kernel_cache_{getpass.getuser()}", 
        "OMP_NUM_THREADS": 2,
        "MKL_NUM_THREADS": 2,
        "CXXFLAGS": "-include cstdint"
    }
)