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
      "long": SimpleLinuxUtilityForResourceManagementEngine(
        default_rqmt={"cpu": 1, "mem": 1, "time": 1}
      ),
    },
    default_engine="long",
  )


MAIL_ADDRESS = getpass.getuser()

WAIT_PERIOD_CACHE = 8
WAIT_PERIOD_JOB_FS_SYNC = 8

JOB_CLEANUP_KEEP_WORK = True
JOB_AUTO_CLEANUP = False
# JOB_FINAL_LOG = "finished.tar.gz"

SHOW_JOB_TARGETS = False
PRINT_ERROR_TASKS = 0
PRINT_ERROR_LINES = 10

# RASR_ROOT = "/u/berger/repositories/rasr_versions/gen_seq2seq_dev/"
RASR_ROOT = "/u/schmitt/src/rasr_versions/gen_seq2seq_dev"
RASR_ARCH = "linux-x86_64-standard"
SCTK_PATH = "/u/beck/programs/sctk-2.4.0/bin/"
RETURNN_ROOT = "/u/schmitt/src/returnn_new"

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

DEFAULT_ENVIRONMENT_SET["HOME"] = "/u/schmitt"

TMP_PREFIX = "/var/tmp/"
DEFAULT_ENVIRONMENT_SET["TMPDIR"] = TMP_PREFIX
DEFAULT_ENVIRONMENT_SET["TMP"] = TMP_PREFIX

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
DEFAULT_ENVIRONMENT_SET["NUMBA_CACHE_DIR"] = "/var/tmp/numba_cache_{}/".format(os.environ["USER"])  # for librosa


def check_engine_limits(current_rqmt, task):
  """
  i6 support for gpu_mem
  """
  current_rqmt["time"] = min(168, current_rqmt.get("time", 2))
  if current_rqmt.get("gpu", 0) > 0 and "-p" not in current_rqmt.get(
          "sbatch_args", []
  ):
    if "sbatch_args" not in current_rqmt:
      current_rqmt["sbatch_args"] = []
      
    if current_rqmt.get("gpu_mem", 0) > 11:
      current_rqmt["sbatch_args"] += ["-p", "gpu_24gb"]
    else:
      current_rqmt["sbatch_args"] += ["-p", "gpu_11gb"]

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
  app_blacklist = {}

  if type(job).__name__ in app_blacklist:
    return call

  binds = ["/work/asr4", "/work/asr3", "/work/common", "/work/tools", "/u/berger", "/u/zeineldeen", "/u/rossenbach",
           "/u/beck", "/work/speech/tuske", "/u/zeyer", "/u/schmitt", "/u/atanas.gruev", "/u/zhou", "/work/smt4/thulke/schmitt"]
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

  if type(job).__name__ == "ReturnnForwardJob":
    """
    Problems with h5py version: with Simon's apptainer, the "labels" entry of hdf files stores bytes instead of strings.
    This leads to problems with RETURNN's HDFDataset since it expects strings.
    """
    app_call += ["/work/asr3/zeyer/schmitt/apptainer/images/u16.sif", "python3", ]
  else:
  #  app_call += ["/work/asr4/berger/apptainer/images/i6_torch-2.2_onnx-1.16.sif", "python3", ]
    app_call += ["/work/tools22/users/schmitt/apptainer/images/i6_torch-2.2_onnx-1.16.sif", "python3", ]

  app_call += call[1:]

  return app_call
