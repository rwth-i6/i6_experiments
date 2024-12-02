import shutil
import subprocess as sp

from i6_core import util
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from sisyphus import Job, Task, tk



class ExportPyTorchModelToOnnxJobV2(Job):
    def __init__(
        self,
        pytorch_checkpoint: PtCheckpoint,
        returnn_config: ReturnnConfig,
        returnn_python_exe: tk.Path,
        returnn_root: tk.Path,
        device: str = "cpu",
        verbosity: int = 4,
    ):
        self.pytorch_checkpoint = pytorch_checkpoint
        self.returnn_python_exe = returnn_python_exe
        self.returnn_config = returnn_config
        self.returnn_root = returnn_root
        self.device = device
        self.verbosity = verbosity

        self.out_returnn_config = self.output_path("returnn.config")
        self.out_onnx_model = self.output_path("model.onnx")

    def tasks(self):
        rqmt = {"gpu": 1} if self.device in ["cuda"] else {}
        yield Task("run", rqmt=rqmt)

    def run(self):
        if isinstance(self.returnn_config, tk.Path):
            returnn_config_path = self.returnn_config.get_path()
            shutil.copy(returnn_config_path, self.out_returnn_config.get_path())
        elif isinstance(self.returnn_config, ReturnnConfig):
            returnn_config_path = self.out_returnn_config.get_path()
            self.returnn_config.write(returnn_config_path)
        else:
            returnn_config_path = self.returnn_config
            shutil.copy(self.returnn_config, self.out_returnn_config.get_path())

        args = [
            self.returnn_python_exe.get_path(),
            self.returnn_root.join_right("tools/torch_export_to_onnx.py").get_path(),
            returnn_config_path,
            str(self.pytorch_checkpoint),
            self.out_onnx_model.get(),
            f"--verbosity={self.verbosity}",
            f"--device={self.device}",
        ]

        util.create_executable("run.sh", args)

        sp.check_call(args)

