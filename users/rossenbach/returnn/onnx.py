import sys
from sisyphus import Job, Task, tk

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint

class ExportPyTorchModelToOnnxJob(Job):
    """
    Experimental exporter job

    JUST FOR DEBUGGING, THIS FUNCTIONALITY SHOULD BE IN RETURNN ITSELF
    """
    __sis_hash_exclude__ = {"quantize_dynamic": False}

    def __init__(self, pytorch_checkpoint: PtCheckpoint, returnn_config: ReturnnConfig, returnn_root: tk.Path, quantize_dynamic: bool = False):

        self.pytorch_checkpoint = pytorch_checkpoint
        self.returnn_config = returnn_config
        self.returnn_root = returnn_root
        self.quantize_dynamic = quantize_dynamic

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
        
        model_state = torch.load(str(self.pytorch_checkpoint), map_location=torch.device('cpu'))
        if isinstance(model_state, dict):
            epoch = model_state["epoch"]
            step = model_state["step"]
            model_state = model_state["model"]
        else:
            epoch = 1
            step = 0
        
        get_model_func = config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        model = get_model_func(epoch=epoch, step=step, onnx_export=True)
        assert isinstance(model, torch.nn.Module)

        model.load_state_dict(model_state)

        export_func = config.typed_value("export")
        assert export_func
        if self.quantize_dynamic:
            import onnx
            from onnxruntime.quantization import quantize_dynamic

            model_fp32 = 'tmp_model.onnx'
            export_func(model=model, model_filename=model_fp32)
            quantized_model = quantize_dynamic(model_fp32, self.out_onnx_model.get())
        else:
            export_func(model=model, model_filename=self.out_onnx_model.get())
