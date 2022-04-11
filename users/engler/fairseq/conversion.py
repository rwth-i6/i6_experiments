import subprocess as sp
import numpy as np
import copy
import pickle

from sisyphus import Job, Task, tk

from i6_core.returnn.config import ReturnnConfig
import i6_core.util as util


def main(
    pytorch_config,
    converter_kwargs,
    fairseq_root,
    pytorch_to_returnn_root,
    returnn_root,
    out_returnn_checkpoint,
    out_returnn_model_dict,
    model_func_name,
):
    if fairseq_root:
        sys.path.append(fairseq_root)
    if pytorch_to_returnn_root:
        sys.path.append(pytorch_to_returnn_root)
    if returnn_root:
        sys.path.append(returnn_root)

    from returnn.tf.util.basic import (
        debug_register_better_repr,
        setup_tf_thread_pools,
        print_available_devices,
    )
    from returnn.log import log

    better_exchook.install()
    debug_register_better_repr()
    log.initialize(verbosity=[5])
    log.Verbosity = 10
    setup_tf_thread_pools()
    print_available_devices()

    def model_func_caller(wrapped_import, inputs):
        return eval(model_func_name)(wrapped_import, inputs, pytorch_config)

    input = np.load("numpy_input.npy")
    input_pt = torch.from_numpy(input)
    print(f"Input shape: {input.shape}")

    import pytorch_to_returnn.log

    pytorch_to_returnn.log.Verbosity = 6
    from pytorch_to_returnn.converter import verify_torch_and_convert_to_returnn

    sys.setrecursionlimit(3000)

    converter = verify_torch_and_convert_to_returnn(
        model_func_caller,
        inputs=input,
        returnn_dummy_input_shape=input.shape,
        export_tf_checkpoint_save_path=out_returnn_checkpoint,
        **converter_kwargs,
    )
    with open(out_returnn_model_dict, "wt", encoding="utf-8") as f:
        f.write(converter.get_returnn_config_serialized())

    model_func(None, inputs=input_pt)


class ConvertPytorchToReturnnJob(Job):
    """
    Conversion Job for fairseq model checkpoints to RETURNN model dictionary and checkpoint
    """

    def __init__(
        self,
        pytorch_config,
        model_func,
        input,
        device="cpu",
        converter_kwargs=None,
        conversion_python_exe=None,
        fairseq_root=None,
        pytorch_to_returnn_root=None,
        returnn_root=None,
    ):
        """
        :param tk.Path pytorch_config: checkpoint to be converted
        :param str model_func: model function used for conversion. It is not yet
            possible to pass a function object to the sisypus job so the function
            needs to be passed as a string
        :param numpy.ndarray: input used for forwarding through the model
        :param str device: cpu or gpu
        :param dict converter_kwargs: additional parameters to converter
        :param tk.Path conversion_python_exe: python exe being used for conversion
        :param tk.Path|None fairseq_root: path to fairseq version to use for conversion
        :param tk.Path|None pytorch_to_returnn_root: path to pytorch_to_returnn version to use for conversion
        :param tk.Path|None returnn_root: returnn root for conversion
        """
        self.pytorch_config = pytorch_config
        self.model_func = model_func
        assert type(model_func) == str, "model function is required to be string"
        self.input = input
        self.device = device
        assert self.device in ["cpu", "gpu"]
        self.converter_kwargs = converter_kwargs
        self.conversion_python_exe = conversion_python_exe
        self.fairseq_root = fairseq_root
        self.pytorch_to_returnn_root = pytorch_to_returnn_root
        self.returnn_root = returnn_root

        if not self.fairseq_root:
            print("WARNING: no explicit fairseq root directory given")
        if not self.pytorch_to_returnn_root:
            print("WARNING: no explicit pytorch_to_returnn root directory given")

        self._config = self.get_returnn_config(
            pytorch_config=pytorch_config,
            model_func=self.model_func,
            converter_kwargs=converter_kwargs,
            fairseq_root=fairseq_root,
            pytorch_to_returnn_root=pytorch_to_returnn_root,
            returnn_root=returnn_root,
        )

        self.out_conversion_config_file = self.output_path("conversion.config")
        self.out_model_func_file = self.output_path("model_func.py")
        self.out_returnn_checkpoint = self.output_path("returnn.checkpoint")
        self.out_returnn_model_dict = self.output_path("returnn.net.dict.py")

        self._config.config["out_returnn_checkpoint"] = self.out_returnn_checkpoint
        self._config.config["out_returnn_model_dict"] = self.out_returnn_model_dict

        self.rqmt = {
            "time": 2,
            "mem": 32,
            "cpu": 1,
            "gpu": 1 if self.device == "gpu" else 0,
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
        config = ReturnnConfig(
            config={
                "pytorch_config": pytorch_config.get_path(),
                "converter_kwargs": converter_kwargs,
                "fairseq_root": fairseq_root.get_path(),
                "pytorch_to_returnn_root": pytorch_to_returnn_root.get_path(),
                "returnn_root": returnn_root
                if returnn_root is not None
                else tk.gs.RETURNN_ROOT,
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
            "returnn_root": kwargs["returnn_root"],
            "conversion_python_exe": kwargs["conversion_python_exe"],
        }
        return super().hash(d)
