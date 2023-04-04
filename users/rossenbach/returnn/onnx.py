import sys
from sisyphus import Job, Task, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint

class ExportPyTorchModelToOnnxJob(Job):
    """
    Experimental exporter job

    JUST FOR DEBUGGING, THIS FUNCTIONALITY SHOULD BE IN RETURNN ITSELF
    """

    def __init__(self, pytorch_checkpoint: PtCheckpoint, returnn_config: ReturnnConfig, returnn_root: tk.Path):

        self.pytorch_checkpoint = pytorch_checkpoint
        self.returnn_config = returnn_config
        self.returnn_root = returnn_root

        self.out_onnx_model = self.output_path("model.onnx")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        sys.path.insert(0, self.returnn_root.get())
        import torch
        from returnn.config import Config
        config = Config()
        self.returnn_config.write("returnn.config")
        config.load_file("returnn.config")

        get_model_func = config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        model = get_model_func()
        assert isinstance(model, torch.nn.Module)
        model_state = torch.load(str(self.pytorch_checkpoint))
        model.load_state_dict(model_state)

        export_func = config.typed_value("export")
        assert export_func
        export_func(model=model, model_filename=self.out_onnx_model.get())