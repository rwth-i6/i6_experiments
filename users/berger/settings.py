import getpass
import sys

sys.path.append("/u/beck/dev/cachemanager/")


def file_caching(path, is_output=False):
    if is_output:
        return "`cf -d %s`" % path
    else:
        return "`cf %s`" % path


def check_engine_limits(current_rqmt, task):
    """Check if requested requirements break and hardware limits and reduce them.
    By default ignored, a possible check for limits could look like this::

        current_rqmt['time'] = min(168, current_rqmt.get('time', 2))
        if current_rqmt['time'] > 24:
            current_rqmt['mem'] = min(63, current_rqmt['mem'])
        else:
            current_rqmt['mem'] = min(127, current_rqmt['mem'])
        return current_rqmt

    :param dict[str] current_rqmt: requirements currently requested
    :param sisyphus.task.Task task: task that is handled
    :return: requirements updated to engine limits
    :rtype: dict[str]
    """
    current_rqmt["time"] = min(168, current_rqmt.get("time", 1))
    return current_rqmt


def engine():
    from sisyphus.localengine import LocalEngine
    from sisyphus.engine import EngineSelector
    from sisyphus.son_of_grid_engine import SonOfGridEngine

    default_rqmt = {
        "cpu": 1,
        "mem": 2,
        "gpu": 0,
        "time": 1,
        "qsub_args": "-l qname=!4C* -l hostname=!*cluster-cn-21*",
    }

    return EngineSelector(
        engines={
            "short": LocalEngine(cpus=3),
            # "short": SonOfGridEngine(default_rqmt=default_rqmt),
            "long": SonOfGridEngine(default_rqmt=default_rqmt),
        },
        default_engine="long",
    )


# need to run bob for ivector
SIS_COMMAND = ["/u/berger/software/env/python3.8_sisyphus/bin/python3", sys.argv[0]]

# how many seconds should be waited before ...
WAIT_PERIOD_CACHE = 1  # stoping to wait for actionable jobs to appear
WAIT_PERIOD_JOB_FS_SYNC = 1  # finishing a job

JOB_CLEANUP_KEEP_WORK = True
JOB_FINAL_LOG = "finished.tar.gz"

SHOW_JOB_TARGETS = False
SHOW_VIS_NAME_IN_MANAGER = False
PRINT_ERROR = False

# RASR_ROOT = "/u/berger/rasr"
# RASR_ROOT = "/u/berger/rasr_tf2"
RASR_ROOT = "/u/berger/rasr_review"
RASR_ARCH = "linux-x86_64-standard"

# needed for returnn sprint interface
# RASR_PYTHON_EXE = "/u/berger/util/returnn_tf2.3.4_mkl_launcher.sh"
# RASR_PYTHON_HOME = "/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl"
# RASR_PYTHON_EXE = "/u/kitza/software/syspy2-tfselfcompile"
# RASR_PYTHON_HOME = "/u/kitza/software/syspy2-tfselfcompile/bin/python2"

# tf 2.3.4
RETURNN_ROOT = "/u/berger/returnn_new"
# RETURNN_PYTHON_EXE = "/u/berger/util/returnn_tf2.3.4_mkl_launcher.sh"
# RETURNN_PYTHON_HOME = "/work/tools/asr/python/3.8.0_tf_2.3.4-generic+cuda10.1+mkl"
RETURNN_PYTHON_HOME = "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1"
RETURNN_PYTHON_EXE = "/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"
# RETURNN_ROOT = "/u/berger/returnn"
# RETURNN_PYTHON_EXE = "/u/berger/software/python/3.6.1/bin/python3.6"
# RETURNN_PYTHON_HOME = "/u/berger/software/python/3.6.1"

G2P_PATH = "/u/beck/dev/g2p/release/lib/python/g2p.py"
G2P_PYTHON = "/u/beck/programs/python/2.7.10/bin/python2"
SCTK_PATH = "/u/beck/programs/sctk-2.4.0/bin/"

BLAS_LIB = "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/libmklml_intel.so"

DEFAULT_ENVIRONMENT_SET["PATH"] = ":".join(
    [
        "/rbi/sge/bin",
        "/rbi/sge/bin/lx-amd64",
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
        "/snap/bin",
        DEFAULT_ENVIRONMENT_SET["PATH"],
    ]
)

DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join(
    [
        "/work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/bazel_out/external/mkl_linux/lib/",
        "/usr/local/cuda-10.1/lib64",
        "/usr/local/cuda-10.1/extras/CUPTI/lib64",
        "/usr/local/cudnn-10.1-v7.6/lib64",
        "/usr/lib/nvidia-450",
        "/work/tools/asr/python/3.8.0/lib",
    ]
)
# DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join([
#     "/usr/local/cuda-10.1_new/extras/CUPTI/lib64",
#     "/usr/local/cuda-10.1_new/lib64",
#     "/usr/local/cudnn-10.1-v7.6/lib64",
#     "/usr/local/cuda-10.0/lib64",
#     "/usr/local/cuda-10.1_new/nsight-systems-2019.3.7.5/Target-x86_64/x86_64",
#     "/work/tools/asr/python/3.8.0/lib",
#     "/usr/lib/nvidia-418",
# ])
# DEFAULT_ENVIRONMENT_SET['PATH'] = '/usr/local/cuda-10.1_new/bin:' + DEFAULT_ENVIRONMENT_SET['PATH']
# DEFAULT_ENVIRONMENT_SET['PATH'] = '/usr/local/cuda-10.0/bin:' + DEFAULT_ENVIRONMENT_SET['PATH']

DEFAULT_ENVIRONMENT_SET["CUDA_HOME"] = "/usr/local/cuda-10.1/"
DEFAULT_ENVIRONMENT_SET["CUDNN"] = "/usr/local/cudnn-10.1-v7.6/"

# DEFAULT_ENVIRONMENT_SET["LD_LIBRARY_PATH"] = ":".join([
#     "/work/tools/asr/python/3.8.0/lib/",
#     "/work/tools/asr/python/3.8.0_tf_1.15-generic+cuda10.1/lib/python3.8/site-packages/tensorflow_core",
#     "/work/tools/asr/python/3.8.0/lib/python3.8/site-packages/numpy/.libs",
#     "/usr/local/cuda-10.1/extras/CUPTI/lib64/",
#     "/usr/local/cuda-10.1/lib64",
#     "/usr/local/cudnn-10.1-v7.6/lib64",
#     "/usr/local/acml-4.4.0/gfortran64_mp/lib/",
#     "/usr/local/acml-4.4.0/gfortran64/lib",
#     "/usr/local/acml-4.4.0/cblas_mp/lib"
#     ])

# DEFAULT_ENVIRONMENT_SET["PYTHONPATH"] = "/u/berger/returnn_new"

DEFAULT_ENVIRONMENT_SET["TMPDIR"] = "/var/tmp"
