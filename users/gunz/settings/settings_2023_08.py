import getpass
import multiprocessing
import os
import sys


sys.path.append("/usr/local/cache-manager")

WORK_DIR = "work"
IMPORT_PATHS = ["config", "recipe", "recipe/"]


def engine():
    from sisyphus.localengine import LocalEngine
    from sisyphus.engine import EngineSelector
    from sisyphus.simple_linux_utility_for_resource_management_engine import (
        SimpleLinuxUtilityForResourceManagementEngine,
    )

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=multiprocessing.cpu_count()),
            "long": SimpleLinuxUtilityForResourceManagementEngine(default_rqmt={"cpu": 1, "mem": 1, "time": 1}),
        },
        default_engine="long",
    )


MAIL_ADDRESS = getpass.getuser()

WAIT_PERIOD_CACHE = 8
WAIT_PERIOD_JOB_FS_SYNC = 8

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"

SHOW_JOB_TARGETS = False
MAX_SUBMIT_RETRIES = 3
PRINT_ERROR_TASKS = 0
PRINT_ERROR_LINES = 10

RASR_ARCH = "linux-x86_64-standard"
SCTK_PATH = "/u/beck/programs/sctk-2.4.0/bin/"

SHOW_JOB_TARGETS = False
SHOW_VIS_NAME_IN_MANAGER = False

DEFAULT_ENVIRONMENT_KEEP = {
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "HOME",
    "HOST",
    "PWD",
    "USER",
}

DEFAULT_ENVIRONMENT_SET["HOME"] = "/u/mgunz"

TMP_PREFIX = "/var/tmp/"
DEFAULT_ENVIRONMENT_SET["TMPDIR"] = TMP_PREFIX


def check_engine_limits(current_rqmt, task):
    CPU_BOTH_JOBLIST = {
        "AdvancedTreeSearchLmImageAndGlobalCacheJob",
        # "BuildGlobalCacheJob",
        "CreateLmImageJob",
    }
    CPU_SLOW_JOBLIST = [
        "ExtractSearchStatisticsJob",
        "Hub5ScoreJob",
        "JoinRightContextPriorsJob",
        "LatticeToCtmJob",
        "PipelineJob",
        "PlotPhonemeDurationsJob",
        "PlotViterbiAlignmentsJob",
        "RasrFeaturesToHdf",
        "ReshapeCenterStatePriorsJob",
        "ScliteJob",
        "SmoothenPriorsJob",
        "VisualizeBestTraceJob",
        "RasrFeatureAndAlignmentWithRandomAllophonesToHDF",
    ]

    """
    i6 support for gpu_mem
    """
    current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
    if current_rqmt.get("gpu", 0) > 0 and "-p" not in current_rqmt.get("sbatch_args", []):
        if current_rqmt.get("gpu_mem", 0) > 11:
            current_rqmt["sbatch_args"] = ["-p", "gpu_24gb", *current_rqmt.get("sbatch_args", [])]
        else:
            current_rqmt["sbatch_args"] = ["-p", "gpu_11gb", *current_rqmt.get("sbatch_args", [])]

    if "-p" not in current_rqmt.get("sbatch_args", []):
        if task._job.__class__.__name__ in CPU_BOTH_JOBLIST:
            current_rqmt["sbatch_args"] = ["-p", "cpu_slow,cpu_fast"]
        elif (task._job.__class__.__name__ in CPU_SLOW_JOBLIST) and "-p" not in current_rqmt.get("sbatch_args", []):
            current_rqmt["sbatch_args"] = ["-p", "cpu_slow"]

    return current_rqmt


def file_caching(path):
    return "`cf %s`" % path


def copy_cached_file(path):
    from subprocess import run

    src = path
    trg = file_caching(path, is_output=True)
    run(["cf", "-cp", src, trg])


def copy_remote_file(path):
    from subprocess import check_output

    return check_output(["cf", path]).decode("utf-8")


def worker_wrapper(job, task_name, call):
    wrapped_jobs = {
        "AdvancedTreeSearchJob",
        "AdvancedTreeSearchLmImageAndGlobalCacheJob",
        "AlignmentJob",
        "CompileNativeOpJob",
        "CompileTFGraphJob",
        "EstimateMixturesJob",
        "GenericSeq2SeqLmImageAndGlobalCacheJob",
        "GenericSeq2SeqSearchJob",
        "LatticeToCtmJob",
        "MakeJob",
        "OptimizeAMandLMScaleJob",
        "ReturnnComputePriorJobV2",
        "ReturnnRasrComputePriorJob",
        "ReturnnRasrComputePriorJobV2",
        "ReturnnRasrTrainingJob",
        "ReturnnTrainingJob",
        "Seq2SeqAlignmentJob",
        "StoreAllophonesJob",
        "TdpFromAlignmentJob",
        "TransformCheckpointJob",
    }

    if type(job).__name__ not in wrapped_jobs:
        return call

    ts = {t.name(): t for t in job.tasks()}
    t = ts[task_name]

    app_call = (
        ["/u/mgunz/src/bin/apptainer_u16_gpu_launcher.sh"]
        if t._rqmt.get("gpu", 0) > 0
        else ["/u/mgunz/src/bin/apptainer_u16_launcher.sh"]
    )
    app_call += call[1:]

    return app_call
