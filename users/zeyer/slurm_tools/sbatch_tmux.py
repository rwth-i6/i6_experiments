import socket
import sys
import os
import time
from functools import reduce
import subprocess as sp
from typing import TypeVar
import argparse
import signal


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.slurm_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run a command in a tmux session on a cluster using sbatch.",
        usage="%(prog)s [options] [sbatch arguments...]",
    )
    arg_parser.add_argument("--start-tmux", action="store_true", help=argparse.SUPPRESS)
    arg_parser.add_argument("--group", type=str, help="chgrp for the tmux socket")
    args, sbatch_args = arg_parser.parse_known_args()

    if args.start_tmux:
        _run_tmux(args)
        return

    run(
        "sbatch",
        *sbatch_args,
        "--signal=B:USR1@5",
        "--wrap=exec %s"
        % " ".join([sys.executable, __file__, "--start-tmux"] + (["--group", args.group] if args.group else [])),
    )


def _run_tmux(args):
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    job_id = os.environ.get("SLURM_JOB_ID", "unknown_slurm_job_id")
    hostname = socket.gethostname()
    socket_filename = f"tmux.sbatch.job_{job_id}.host_{hostname}.pid_{os.getpid()}.socket"

    while True:
        run("tmux", "-S", socket_filename, "new-session", "-d")
        print(f"Started tmux session with socket: {socket_filename}")
        print(f"To attach to the tmux session, run: ssh {hostname} tmux -S {socket_filename} attach")
        print("Waiting for tmux session to end...")
        sys.stdout.flush()

        assert os.path.exists(socket_filename)
        if args.group:
            run("chgrp", args.group, socket_filename)
            run("chmod", "g+rw", socket_filename)

        # sanity check that the tmux session is running
        sp.check_call(["tmux", "-S", socket_filename, "has-session"])

        # wait until the tmux session ends, either by the user or by receiving a signal
        try:
            while True:
                try:
                    sp.check_call(["tmux", "-S", socket_filename, "has-session"])
                except sp.CalledProcessError:
                    print("Tmux session has ended.")
                    break
                time.sleep(1)

        except BaseException:
            run("tmux", "-S", socket_filename, "kill-session")
            raise

        finally:
            os.unlink(socket_filename)

        # if we got here, it means the tmux session ended without us receiving a signal.
        # so restart a new tmux session to wait for the next signal.
        print("Restart a new tmux session...")


def _signal_handler(signum, frame):
    print("Signal handler got signal", signum)
    raise KeyboardInterrupt("Received signal to terminate tmux session.")


def run(*args):
    print("Running:", *args)
    return sp.check_call(args)


if __name__ == "__main__":
    try:
        main()
    except sp.CalledProcessError as exc:
        sys.exit(exc.returncode)
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(1)
