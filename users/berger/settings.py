import sys

sys.path.append("/u/beck/dev/cachemanager/")


def file_caching(path, is_output=False):
    if is_output:
        return "`cf -d %s`" % path
    else:
        return "`cf %s`" % path


def check_engine_limits(current_rqmt, task):
    """
    i6 support for gpu_mem
    """
    current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
    if current_rqmt.get("gpu", 0) > 0 and "-p" not in current_rqmt.get("sbatch_args", []):
        if current_rqmt.get("gpu_mem", 0) > 11:
            current_rqmt["sbatch_args"] = ["-p", "gpu_24gb"]
        else:
            current_rqmt["sbatch_args"] = ["-p", "gpu_11gb"]
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
            "short": LocalEngine(cpus=4, mem=8),
            "long": SimpleLinuxUtilityForResourceManagementEngine(default_rqmt=default_rqmt),
        },
        default_engine="long",
    )


def worker_wrapper(job, task_name, call):
    wrapped_jobs = {
        "MakeJob",
        "ReturnnTrainingJob",
        "ReturnnRasrTrainingJob",
        "OptunaReturnnTrainingJob",
        "CompileTFGraphJob",
        "OptunaCompileTFGraphJob",
        "ReturnnRasrComputePriorJob",
        "ReturnnComputePriorJob",
        "ReturnnComputePriorJobV2",
        "OptunaReturnnComputePriorJob",
        "CompileNativeOpJob",
        "AdvancedTreeSearchJob",
        "AdvancedTreeSearchLmImageAndGlobalCacheJob",
        "GenericSeq2SeqSearchJob",
        "GenericSeq2SeqLmImageAndGlobalCacheJob",
        "LatticeToCtmJob",
        "OptimizeAMandLMScaleJob",
        "AlignmentJob",
        "Seq2SeqAlignmentJob",
        "EstimateMixturesJob",
        "EstimateCMLLRJob",
        "DumpStateTyingJob",
        "StoreAllophonesJob",
        "FeatureExtractionJob",
    }
    if type(job).__name__ not in wrapped_jobs:
        return call
    binds = ["/work/asr4", "/work/asr3", "/work/common", "/u/corpora"]
    ts = {t.name(): t for t in job.tasks()}
    t = ts[task_name]

    app_call = [
        "apptainer",
        "exec",
    ]
    if t._rqmt.get("gpu", 0) > 0:
        app_call += ["--nv"]

    for path in binds:
        app_call += ["-B", path]

    app_call += [
        "/work/asr4/berger/apptainer/images/tf_2_8.sif",
        "python3",
    ]

    app_call += call[1:]

    return app_call


# need to run bob for ivector
SIS_COMMAND = ["/u/berger/software/env/python3.10_sisyphus/bin/python", sys.argv[0]]

WAIT_PERIOD_CACHE = 1  # stopping to wait for actionable jobs to appear
WAIT_PERIOD_JOB_FS_SYNC = 1  # finishing a job

JOB_AUTO_CLEANUP = False
JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"

SHOW_JOB_TARGETS = False
SHOW_VIS_NAME_IN_MANAGER = False
PRINT_ERROR = False

# G2P_PATH = "/u/beck/dev/g2p/release/lib/python/g2p.py"
# G2P_PYTHON = "/u/beck/programs/python/2.7.10/bin/python2"
SCTK_PATH = "/u/beck/programs/sctk-2.4.0/bin/"

# BLAS_LIB = "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so"

DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join(
    [
        "/usr/local/lib/tensorflow/",
        "/usr/local/lib/python3.8/dist-packages/tensorflow/",
        "/usr/local/lib/python3.8/dist-packages/scipy/.libs",
        "/usr/local/lib/python3.8/dist-packages/numpy.libs",
        "/usr/local/cuda/extras/CUPTI/lib64",
        "/usr/local/cuda/compat/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
        "/.singularity.d/libs",
    ]
)

TMP_PREFIX = "/var/tmp/"
DEFAULT_ENVIRONMENT_SET["TMPDIR"] = TMP_PREFIX
