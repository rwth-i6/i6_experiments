from sisyphus import Job, Task, tk, Path, gs

import os
import subprocess as sp
import pprint

class ReturnnCleanupOldModelsJob(Job):
    def __init__(self,
        config,
        cwd = None,
        model = None,
        scores = None,
        dry_run = False,
        returnn_python_exe = None,
        returnn_root = None,
        gpu=1,
    ):
        self.config = config
        self.cwd = cwd
        self.model = model
        self.scores = scores
        self.dry_run = dry_run
        self.returnn_python_exe = (
            returnn_python_exe
            if returnn_python_exe is not None
            else gs.RETURNN_PYTHON_EXE
        )
        self.returnn_root = (
            returnn_root if returnn_root is not None else gs.RETURNN_ROOT
        )

        self.rqmt = {
            "gpu": gpu,
            "cpu": 1,
            "mem": 1,
            "time": 0.01,
        }

        self.out_log_file = self.output_path("cleanup.log")

    def tasks(self):
        extra_args = {}
        if self.rqmt["gpu"] == 0:
            extra_args["mini_task"] = True
        yield Task('run', rqmt = self.rqmt, **extra_args)
    
    def run(self):
        if isinstance(self.config, dict):
            pp = pprint.PrettyPrinter(indent=2, width=150)
            code = "cleanup_old_models = %s" % pp.pformat(self.config)
            config_file = "returnn_cleanup.config"
            with open(config_file, "w") as f:
                f.write(code)
            self.config = config_file
        args = [
            tk.uncached_path(self.returnn_python_exe),
            os.path.join(tk.uncached_path(self.returnn_root), "tools/cleanup-old-models.py"),
            "--config",
            tk.uncached_path(self.config),
        ]
        if self.cwd is not None:
            args += ["--cwd", self.cwd.get_path()]
        if self.model is not None:
            args += ["--model", self.model.get_path()]
        if self.scores is not None:
            args += ["--scores", self.scores.get_path()]
        if self.dry_run:
            args += ["--dry-run"]
        
        args += ["--skip_confirm"]
        
        # args += [">", self.out_log_file.get_path()]
        
        sp.check_call(args, stdout = open(self.out_log_file.get_path(), "w"))

        with open(self.out_log_file.get_path(), "r") as f:
            print(f.read())
    
    @classmethod
    def from_returnn_training_job(cls, returnn_training_job, **kwargs):
        job_work_dir = Path(
            returnn_training_job.work_path(),
            creator = returnn_training_job,
            available = super(Job, returnn_training_job).path_available,
        )
        return cls(
            config = returnn_training_job.out_returnn_config_file,
            cwd = job_work_dir,
            model = None,
            **kwargs,
        )

