# Sisyphus settings file
import getpass
import os.path

#############################
# Local Settings File Options
#############################

# can be "apptainer", "docker", "singularity" or None
# docker is still experimental and not really tested
CONTAINER_MODE = "apptainer"

# local path or e.g. docker registry image path
CONTAINER_IMAGE = "/home/pv653172/setups/librispeech/20230328_wav2vec2/dependencies/u22_torch1.13_fairseq.sif"

# file systems to bind in a "<source_path>:<target_path>" format
CONTAINER_BINDS = [
    "/work/pv653172",
    "/hpcwork/pv653172",
    "/rwthfs/rz/cluster/work/pv653172/",
    "/rwthfs/rz/cluster/hpcwork/pv653172/",
]

# can be "sge", "slurm" or "pbs" (pbs is experimental)
SUBMIT_ENGINE = "slurm"

# hostname or ip of machine to use for cluster access
SUBMIT_GATEWAY = None

# the username
USER = getpass.getuser()

# List if extra env vars to set "before" sisyphus execution in a "ENV_VAR=content" format.
# Can for example be used to set the PYTHONPATH to a custom sisyphus
# different to the one installed in the container for debugging / development purposes
EXTRA_ENV = []


def check_engine_limits(current_rqmt, task):
    """
    Alter engine requests based on personal needs
    """
    # Example:
    # The cluster allows only 7 days of runtime, so never
    # schedule jobs with more than that

    current_rqmt['time'] = min(168, current_rqmt.get('time', 2))

    # Example:
    # A slurm cluster has different partitions for differently sized GPUs,
    # so select partition based on gpu_mem

    if current_rqmt.get("gpu", 0) > 0 and "-p" not in current_rqmt.get("sbatch_args", []):
        current_rqmt['sbatch_args'] = ['-p', 'dgx2', '-A', 'supp0003']

    # Example:
    # Set specficic queue options based on the alias name of a job

    # aliases = []
    # for prefix in list(task._job._sis_alias_prefixes) + [""]:
    #    for alias in task._job.get_aliases() or [""]:
    #        aliases.append(prefix + alias)

    # if "gmm_align" in "\t".join(aliases or ""):
    #    current_rqmt['sbatch_args'] = ['-p', 'cpu_slow']

    return current_rqmt


##########################
# Sisyphus Global Settings
##########################
try:
    MAIL_ADDRESS = getpass.getuser()
except KeyError:
    MAIL_ADDRESS = None

# Those are some recommended defaults, can be changed to your liking
# See the global_settings.py file of Sisyphus for documentation

JOB_USE_TAGS_IN_PATH = False
JOB_AUTO_CLEANUP = False
SHOW_JOB_TARGETS = False
PRINT_ERROR = False
DELAYED_CHECK_FOR_WORKER = False

WARNING_ABSPATH = False

SHORT_JOB_NAMES = True

# For debugging to 1
GRAPH_WORKER = 1

DEFAULT_ENVIRONMENT_KEEP = {
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "PWD",
    "SGE_STDERR_PATH",
    "SGE_TASK_ID",
    "TMP",
    "TMPDIR",
    "USER",
    "LD_LIBRARY_PATH",
}

DEFAULT_ENVIRONMENT_SET = {
    "LANG": "en_US.UTF-8",
    "MKL_NUM_THREADS": 2,
    "OMP_NUM_THREADS": 2,
    "PATH": ":".join(["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin"]),
    "SHELL": "/bin/bash",
    "NUMBA_CACHE_DIR": f"/var/tmp/numba_cache_{USER}",  # used for librosa
    "PYTORCH_KERNEL_CACHE_PATH": f"/var/tmp/",  # used for cuda pytorch
}


###########################
# Sisyphus Code Definitions
###########################


def engine():
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine

    def job_name_mapping(name):
        name = name.split("/")[-1]
        return name.replace("/", ".")

    if SUBMIT_ENGINE == "slurm":
        from sisyphus.simple_linux_utility_for_resource_management_engine import (
            SimpleLinuxUtilityForResourceManagementEngine,
        )

        return EngineSelector(
            engines={
                "short": LocalEngine(cpus=4),
                "long": SimpleLinuxUtilityForResourceManagementEngine(
                    default_rqmt={"cpu": 1, "mem": 2, "gpu": 0, "time": 1},
                    gateway=SUBMIT_GATEWAY,
                    job_name_mapping=job_name_mapping,
                ),
            },
            default_engine="long",
        )
    elif SUBMIT_ENGINE == "sge":
        raise NotImplementedError()
    elif SUBMIT_ENGINE == "openpbs":
        raise NotImplementedError()
    else:
        raise ValueError("settings.py: Unsupported Sisyphus Engine: %s" % SUBMIT_ENGINE)


def build_apptainer_command(call):
    """
    Apptainer specific launch code
    """
    command = []
    command += EXTRA_ENV
    command += ["apptainer", "exec", "--nv"]
    for bind in CONTAINER_BINDS:
        command += ["--bind", bind]
    command += [CONTAINER_IMAGE]
    return command + ["sis"] + call[2:]


def build_singularity_command(call):
    """
    Singularity specific launch code
    """
    command = []
    command += EXTRA_ENV
    command += ["singularity", "exec", "--nv"]
    for bind in CONTAINER_BINDS:
        command += ["--bind", bind]
    command += [CONTAINER_IMAGE]
    return command + ["sis"] + call[2:]


def build_docker_command(call, rqmt):
    """
    Docker specific launch code
    """
    from pwd import getpwnam

    userid, groupid = getpwnam(USER)[2:4]
    exp_dir = os.path.dirname(__file__)
    command = [
        "docker",
        "run",
        "-t",
        "--rm",  # delete container after execution
        "-u",
        "%i:%i" % (userid, groupid),  # passing the username directly does not work with LDAP users
        "-w",
        exp_dir,
        "-m",
        f"{rqmt['mem']}g",
        "--shm-size",
        f"{rqmt['mem']}g",
    ]
    if rqmt.get("gpu", 0) > 0:
        command += [
            "--runtime=nvidia",
            "--gpus",
            "device=0",  # TODO: Check how to do this
        ]
    for env in EXTRA_ENV:
        command += ["-e", env]
    for bind in CONTAINER_BINDS:
        command += ["-v", bind]
    command += [CONTAINER_IMAGE]
    command += ["sh", "-e", "-c"]
    return command + ["sis"] + call[2:]


def worker_wrapper(job, task_name, call):
    """
    All worker calls are passed through this function.
    Is used to wrap the execution call with the correct container command.
    Usually it is not necessary to alter things here,
    but any worker call can be fully customized here.
    """
    from sisyphus.engine import EngineSelector
    from sisyphus.localengine import LocalEngine

    ts = {t.name(): t for t in job.tasks()}
    t = ts[task_name]

    if CONTAINER_MODE == "apptainer":
        command = build_apptainer_command(call)
    elif CONTAINER_MODE == "docker":
        command = build_docker_command(call, t.rqmt())
    elif CONTAINER_MODE == "singularity":
        command = build_singularity_command(call)
    else:
        raise ValueError("Invalid CONTAINER_MODE %s" % CONTAINER_MODE)

    e = engine()  # Usually EngineSelector, but can be LocalEngine if no settings file is present
    if isinstance(e, EngineSelector):
        e = engine().get_used_engine_by_rqmt(t.rqmt())
    if isinstance(e, LocalEngine):
        print(f"running ${' '.join(call)}")
        return call
    else:
        print(f"running ${' '.join(command)}")
        return command
