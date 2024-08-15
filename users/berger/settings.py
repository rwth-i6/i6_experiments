import getpass
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
    if current_rqmt.get("gpu", 0) > 0 and "-p" not in current_rqmt.get(
        "sbatch_args", []
    ):
        if current_rqmt.get("gpu_mem", 0) > 11:
            current_rqmt["sbatch_args"] = ["-p", "gpu_24gb"]
        else:
            current_rqmt["sbatch_args"] = ["-p", "gpu_11gb"]

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
    rasr_jobs = {
        "MakeJob",
        "CompileNativeOpJob",
        "AdvancedTreeSearchJob",
        "AdvancedTreeSearchLmImageAndGlobalCacheJob",
        "FeatureExtractionJob",
        "GenericSeq2SeqSearchJob",
        "GenericSeq2SeqSearchJobV2",
        "GenericSeq2SeqLmImageAndGlobalCacheJob",
        "CreateLmImageJob",
        "BuildGenericSeq2SeqGlobalCacheJob",
        "LatticeToCtmJob",
        "OptimizeAMandLMScaleJob",
        "AlignmentJob",
        "Seq2SeqAlignmentJob",
        "EstimateMixturesJob",
        "EstimateCMLLRJob",
    }
    torch_jobs = {
        "ReturnnTrainingJob",
        "ReturnnRasrTrainingJob",
        "OptunaReturnnTrainingJob",
        "ReturnnDumpHDFJob",
        "CompileTFGraphJob",
        "OptunaCompileTFGraphJob",
        "ReturnnRasrComputePriorJob",
        "ReturnnComputePriorJob",
        "ReturnnComputePriorJobV2",
        "OptunaReturnnComputePriorJob",
        "ReturnnForwardJob",
        "ReturnnForwardJobV2",
        "ReturnnForwardComputePriorJob",
        "OptunaReturnnForwardComputePriorJob",
        "CompileKenLMJob",
        "OptunaReportIntermediateScoreJob",
        "OptunaReportFinalScoreJob",
    }
    onnx_jobs = {
        "ExportPyTorchModelToOnnxJob",
        "TorchOnnxExportJob",
        "OptunaExportPyTorchModelToOnnxJob",
        "OptunaTorchOnnxExportJob",
    }
    jobclass = type(job).__name__
    if jobclass in rasr_jobs:
        image = "/work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif"
    elif jobclass in torch_jobs:
        image = "/work/asr4/berger/apptainer/images/i6_torch-2.2_onnx-1.16.sif"
    elif jobclass in onnx_jobs:
        # use this one because mhsa is not onnx exportable in torch 2 yet
        image = "/work/asr4/berger/apptainer/images/i6_u22_pytorch1.13_onnx.sif"
    else:
        return call

    binds = ["/work/asr4", "/work/common", "/work/tools/", "/work/asr4/rossenbach"]
    ts = {t.name(): t for t in job.tasks()}
    t = ts[task_name]

    app_call = [
        "apptainer",
        "exec",
    ]

    app_call += ["--env", f"NUMBA_CACHE_DIR=/var/tmp/numba_cache_{getpass.getuser()}"]

    if t._rqmt.get("gpu", 0) > 0:
        app_call += ["--nv"]

    for path in binds:
        app_call += ["-B", path]

    app_call += [image, "python3"]

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

MAIL_ADDRESS = None

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
        "/usr/local/lib/python3.10/dist-packages/tensorflow/",
        "/usr/local/lib/python3.10/dist-packages/scipy/.libs",
        "/usr/local/lib/python3.10/dist-packages/numpy.libs",
        "/usr/local/cuda/extras/CUPTI/lib64",
        "/usr/local/cuda/compat/lib",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
        "/.singularity.d/libs",
    ]
)

TMP_PREFIX = "/var/tmp/"
DEFAULT_ENVIRONMENT_SET.update(
    {
        "TMPDIR": TMP_PREFIX,
        "TMP": TMP_PREFIX,
        "NUMBA_CACHE_DIR": f"{TMP_PREFIX}/numba_cache_{getpass.getuser()}",  # used for librosa
        "PYTORCH_KERNEL_CACHE_PATH": f"{TMP_PREFIX}/pt_kernel_cache_{getpass.getuser()}",  # used for cuda pytorch
    }
)
