import subprocess as sp
from typing import Optional

import numpy as np
import numpy.typing as npt
from i6_core import returnn, util
from sisyphus import Job, Task, tk


class ConvertPytorchToReturnnJob(Job):
    """
    Conversion Job for PyTorch model checkpoints to RETURNN model dictionary and checkpoint
    """

    def __init__(
        self,
        pytorch_config: tk.Path,
        model_func: str,
        input: npt.NDArray,
        conversion_python_exe: tk.Path,
        fairseq_root: tk.Path,
        pytorch_to_returnn_root: tk.Path,
        returnn_root: tk.Path,
        device: str = "cpu",
        converter_kwargs: Optional[dict] = None,
    ):
        """
        :param pytorch_config: checkpoint to be converted
        :param model_func: model function used for conversion. It is not yet
            possible to pass a function object to the sisypus job so the function
            needs to be passed as a string. The model function needs to have type
            Callable[[Optional[Callable[[str], types.ModuleType]], torch.Tensor], torch.Tensor, str].
            The first two inputs are the parameters required by the converter and
            the third parameter is set to the path to the pytorch_config before
            given to the converter.
        :param input: used for forwarding through the model
        :param device: cpu or gpu
        :param converter_kwargs: additional parameters to converter
        :param conversion_python_exe: python exe being used for conversion
        :param fairseq_root: path to fairseq version to use for conversion
        :param pytorch_to_returnn_root: path to pytorch_to_returnn version to use for conversion
        :param returnn_root: returnn root for conversion
        """
        assert isinstance(
            model_func, str
        ), "model function is required to be string, see docstring"
        assert device in ["cpu", "gpu"]
        self.input = input
        self.conversion_python_exe = conversion_python_exe

        self._config = self.get_returnn_config(
            pytorch_config=pytorch_config,
            model_func=model_func,
            converter_kwargs=converter_kwargs,
            fairseq_root=fairseq_root,
            pytorch_to_returnn_root=pytorch_to_returnn_root,
            returnn_root=returnn_root,
        )

        self.out_conversion_config_file = self.output_path("conversion.py")
        self.out_returnn_checkpoint = self.output_path("returnn.checkpoint")
        self.out_returnn_model_dict = self.output_path("returnn_net_dict.py")

        self._config.config["out_returnn_checkpoint"] = self.out_returnn_checkpoint
        self._config.config["out_returnn_model_dict"] = self.out_returnn_model_dict

        self.rqmt = {
            "time": 2,
            "mem": 32,
            "cpu": 1,
            "gpu": 1 if device == "gpu" else 0,
        }

    def tasks(self):
        yield Task("create_files", mini_task=True)
        yield Task("run", rqmt=self.rqmt)

    def create_files(self):
        self._config.write(self.out_conversion_config_file.get_path())
        np.save("numpy_input", self.input)
        util.create_executable("conversion.sh", self._get_run_cmd())

    def run(self):
        sp.check_call(["./conversion.sh"])

    def _get_run_cmd(self):
        run_cmd = [
            self.conversion_python_exe.get_path(),
            self.out_conversion_config_file.get_path(),
        ]
        return run_cmd

    @classmethod
    def get_returnn_config(
        cls,
        pytorch_config,
        model_func,
        converter_kwargs=None,
        fairseq_root=None,
        pytorch_to_returnn_root=None,
        returnn_root=None,
        **kwargs,
    ):
        fairseq_root = fairseq_root.get_path() if fairseq_root is not None else None
        pytorch_to_returnn_root = (
            pytorch_to_returnn_root.get_path()
            if pytorch_to_returnn_root is not None
            else None
        )

        config = returnn.ReturnnConfig(
            config={
                "pytorch_config": pytorch_config.get_path(),
                "converter_kwargs": converter_kwargs,
                "fairseq_root": fairseq_root,
                "pytorch_to_returnn_root": pytorch_to_returnn_root,
                "returnn_root": returnn_root,
                "model_func_name": model_func.split(" ")[1].split("(")[0],
            },
            python_prolog=[
                "import torch",
                "import better_exchook",
                "import sys",
                "import numpy as np",
                "from numpy import array",
            ],
            python_epilog=[
                model_func,
                main,
                "if __name__ == '__main__':"
                "   main("
                "       pytorch_config,"
                "       converter_kwargs,"
                "       fairseq_root,"
                "       pytorch_to_returnn_root,"
                "       returnn_root,"
                "       out_returnn_checkpoint,"
                "       out_returnn_model_dict,"
                "       model_func_name,"
                ")",
            ],
            hash_full_python_code=True,
        )
        return config

    @classmethod
    def hash(cls, kwargs):
        d = {
            "conversion_config": cls.get_returnn_config(**kwargs),
            "conversion_python_exe": kwargs["conversion_python_exe"],
        }
        return super().hash(d)
