import getpass
import os
import sys

# Paths and resources shared by the settings hooks below.
SETUP_ROOT = "/u/azim.javed/experiments/training/qat"
TMP_PREFIX = "/var/tmp/"
NUMBA_CACHE_DIR = os.path.join(TMP_PREFIX, f"numba_cache_{getpass.getuser()}")

sys.path.append("/u/beck/dev/cachemanager/")

# Compatibility path for the baseline configuration.
if "/u/azim.javed/experiments/training/baselines_clean" not in sys.path:
    sys.path.append("/u/azim.javed/experiments/training/baselines_clean")


def file_caching(path, is_output=False):
    return f"`cf{' -d' if is_output else ''} {path}`"


CPU_SLOW_JOBLIST = {
    "ScliteJob",
    "Hub5ScoreJob",
    "PipelineJob",
}
EXCLUDED_NODES = ["cn-281", "cn-230", "cn-236"]


def _gpu_partition(gpu_mem):
    """Select the smallest GPU partition satisfying the memory requirement (GB)."""
    if gpu_mem > 32:
        return "gpu_48gb"
    if gpu_mem > 24:
        return "gpu_32gb"
    if gpu_mem > 11:
        return "gpu_24gb"
    return "gpu_11gb"


def check_engine_limits(current_rqmt, task):
    """Apply runtime limits, partition selection and node exclusions."""
    current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
    sbatch_args = list(current_rqmt.get("sbatch_args", []))
    if current_rqmt.get("gpu", 0) > 0 and "-p" not in sbatch_args:
        sbatch_args = ["-p", _gpu_partition(current_rqmt.get("gpu_mem", 0))]

    if task._job.__class__.__name__ in CPU_SLOW_JOBLIST:
        sbatch_args = ["-p", "cpu_slow"]

    current_rqmt["sbatch_args"] = sbatch_args + [f"--exclude={','.join(EXCLUDED_NODES)}"]

    return current_rqmt


def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine
    from sisyphus.simple_linux_utility_for_resource_management_engine import (
        SimpleLinuxUtilityForResourceManagementEngine,
    )

    def job_name_mapping(name):
        name = name.split("/")[-1]
        return name.replace("/", ".")

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
                default_rqmt=default_rqmt,
                job_name_mapping=job_name_mapping,
            ),
        },
        default_engine="long",
    )


DEFAULT_CONTAINER_IMAGE = "/work/asr4/hilmes/apptainer/torch-2.8_onnx-1.22_v3.sif"
CONTAINER_IMAGES_BY_JOB = {
    "ReturnnTrainingJob": "/work/asr4/berger/apptainer/images/torch-2.8_onnx-1.22.sif",
}
CONTAINER_BINDS = ("/u", "/work/smt4", "/work/asr4", "/work/common", "/var/tmp", "/run")

# Jobs that run memristor inference (recognition / prior): import synaptogen_ml,
# so enable the fast-inference path via env (SYN_FAST is read at import time in
# synaptogen_ml/__init__.py). Scoping to ReturnnForwardJobV2 keeps training jobs
# on the bit-exact eager path.
RECOGNITION_JOBS = {
    "ReturnnForwardJobV2",
}


def worker_wrapper(job, task_name, call):
    """Run workers in the job's container with shared mounts and cache settings."""
    job_name = job.__class__.__name__
    image = CONTAINER_IMAGES_BY_JOB.get(job_name, DEFAULT_CONTAINER_IMAGE)
    app_call = [
        "apptainer",
        "exec",
        "--env",
        f"NUMBA_CACHE_DIR={NUMBA_CACHE_DIR}",
    ]
    if job_name in RECOGNITION_JOBS:
        app_call += ["--env", "SYN_FAST=1"]
    app_call += [
        "--pwd",
        os.environ.get("PWD", os.getcwd()),
        "--nv",
    ]

    for path in CONTAINER_BINDS:
        app_call += ["-B", path]

    return app_call + [image, "python3"] + call[1:]


# Manager and job lifecycle settings.
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

# DEFAULT_ENVIRONMENT_SET is injected by Sisyphus when loading this module.
DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join(
    [
        "/.singularity.d/libs",
        "/usr/local/cuda/extras/CUPTI/lib64",
        "/usr/local/cuda/compat/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
    ]
)

DEFAULT_ENVIRONMENT_SET.update(
    {
        "TMPDIR": TMP_PREFIX,
        "TMP": TMP_PREFIX,
        "PYTHONPATH": os.environ.get("PWD", os.getcwd())
        + ":"
        + SETUP_ROOT
        + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else ""),
        "NUMBA_CACHE_DIR": NUMBA_CACHE_DIR,  # used for librosa
        "PYTORCH_KERNEL_CACHE_PATH": os.path.join(TMP_PREFIX, f"pt_kernel_cache_{getpass.getuser()}"),
        "CXXFLAGS": "-include cstdint",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)
