import sys
import os
from sisyphus import Job, Task, tk
from typing import Any, Dict, Optional, Tuple, List
import logging

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from onnxruntime.quantization import quant_pre_process, quantize_static, CalibrationDataReader, CalibrationMethod, QuantType, QuantFormat
from onnxruntime import InferenceSession, SessionOptions
from returnn.datasets import Dataset, init_dataset
import numpy as np

class ExportPyTorchModelToOnnxJob(Job):
    """
    Experimental exporter job

    JUST FOR DEBUGGING, THIS FUNCTIONALITY SHOULD BE IN RETURNN ITSELF
    """
    __sis_hash_exclude__ = {"quantize_dynamic": False, "quantize_static": False}

    def __init__(self, pytorch_checkpoint: PtCheckpoint, returnn_config: ReturnnConfig, returnn_root: tk.Path, quantize_dynamic: bool = False):

        self.pytorch_checkpoint = pytorch_checkpoint
        self.returnn_config = returnn_config
        self.returnn_root = returnn_root
        self.quantize_dynamic = quantize_dynamic

        self.out_onnx_model = self.output_path("model.onnx")
        self.rqmt = {"time": 2, "cpu": 4, "mem": 16}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        sys.path.insert(0, self.returnn_root.get())
        import torch
        from returnn.config import Config
        config = Config()
        self.returnn_config.write("returnn.config")
        config.load_file("returnn.config")
        
        model_state = torch.load(str(self.pytorch_checkpoint),map_location=torch.device("cpu"))
        if isinstance(model_state, dict):
            epoch = model_state["epoch"]
            step = model_state["step"]
            model_state = model_state["model"]
        else:
            epoch = 1
            step = 0
        
        get_model_func = config.typed_value("get_model")
        assert get_model_func, "get_model not defined"
        model = get_model_func(epoch=epoch, step=step)
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


class ModelQuantizeStaticJob(Job):

    __sis_hash_exclude__ = {
        "moving_average": False,
        "smoothing_factor": 0.0,
        "symmetric": False,
        "activation_type": QuantType.QInt8,
        "quant_format": QuantFormat.QDQ,
        "weight_type": QuantType.QInt8,
        "final_skip": (None, None),
        "ops_to_quant": None,
        "smooth_quant": False,
    }

    def __init__(self,
        model: tk.Path,
        dataset: Dict[str, Any],
        num_seqs: int = 10,
        num_parallel_seqs: int = 25,
        calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
        moving_average: bool = False,
        smoothing_factor: float = 0.0,
        symmetric: bool = False,
        activation_type = QuantType.QInt8,
        quant_format = QuantFormat.QDQ,
        weight_type = QuantType.QInt8,
        final_skip: Tuple[Optional[int], Optional[int]] = (None, None),
        ops_to_quant: Optional[List[str]] = None,
        smooth_quant: bool = False,
    ):
        """
        :param model:
        :param dataset:
        :param num_seqs:
        :param num_parallel_seqs:
        :param moving_average: whether to use moving average for MinMax or Symmetry for Entropy
        """
        self.model = model
        self.dataset = dataset
        self.num_seqs = num_seqs
        self.num_parallel_seqs = num_parallel_seqs
        self.moving_average = moving_average
        self.activation_type = activation_type
        self.quant_format = quant_format
        self.weight_type = weight_type

        self.out_model = self.output_path("model.onnx")
        if num_seqs >= 5000:
            time = 12
        elif num_seqs >= 2500:
            time = 6
        elif num_seqs >= 1000:
            time = 4
        else:
            time = 1
        if not calibrate_method == CalibrationMethod.MinMax:
            time *= 2

        self.rqmt = {"cpu": 8 if num_seqs > 100 else 4, "mem": 16.0 if calibrate_method == CalibrationMethod.MinMax else 48, "time": time}
        self.calibration_method = calibrate_method
        self.smoothing_factor = smoothing_factor
        self.symmetric = symmetric
        self.final_skip = final_skip
        self.smooth_quant = smooth_quant
        self.ops_to_quant = ops_to_quant
        self.out_dev_log = self.output_path("dev_log")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        print("Start")
        quant_pre_process(
            input_model_path=self.model.get_path(),
            output_model_path="model_prep.onnx")

        class DummyDataReader(CalibrationDataReader):

            def __init__(self, model_str: str, data: Dataset, max_seqs: int, final_skip:  Optional[Tuple[int, int]] = (None, None)):

                self.max_seqs = max_seqs
                self.data = data
                self.idx: int = 0
                sess_option = SessionOptions()
                logging.info(f"Data Loading {os.getenv('SLURM_CPUS_PER_TASK')}")
                sess_option.intra_op_num_threads = int(os.getenv('SLURM_CPUS_PER_TASK'))
                session = InferenceSession(model_str, sess_option)
                self.input_name_1 = session.get_inputs()[0].name
                self.input_name_2 = session.get_inputs()[1].name
                self.final_skip_step = final_skip[0]
                self.final_skip_count = final_skip[1]

            def get_next(self):
                init_dataset(self.data)
                if not self.data.is_less_than_num_seqs(self.idx) or self.idx >= self.max_seqs:
                    if self.final_skip_step is not None and self.idx < self.max_seqs + self.final_skip_step * self.final_skip_count:
                        self.idx += self.final_skip_step
                        logging.info(f"Skipping to Seq {self.idx}")
                        self.data.load_seqs(self.idx, self.idx + 1)
                        seq_len: np.ndarray = self.data.get_seq_length(self.idx)["data"]
                        data: np.ndarray = self.data.get_data(self.idx, "data")
                        seq_len = np.array([seq_len], dtype=np.int32)
                        data = np.expand_dims(data, axis=0)
                        return {self.input_name_1: data, self.input_name_2: seq_len}
                    else:
                        return None
                self.data.load_seqs(self.idx, self.idx + 1)
                seq_len: np.ndarray = self.data.get_seq_length(self.idx)["data"]
                data: np.ndarray = self.data.get_data(self.idx, "data")
                if self.idx % 10 == 0:
                    logging.info(f"{self.idx} seqs seen")
                seq_len = np.array([seq_len], dtype=np.int32)
                data = np.expand_dims(data, axis=0)
                self.idx += 1
                return {self.input_name_1: data, self.input_name_2: seq_len}

            def __iter__(self):
                data = []
                x = self.get_next()
                while x is not None:
                    data.append(x)
                    x = self.get_next()
                shape = {arr["data"].shape for arr in data}
                shape2 = {arr["data_len"].shape for arr in data}
                for x in data:
                    yield x

        dataset: Dataset = init_dataset(self.dataset)
        dataset.init_seq_order(1)
        y = DummyDataReader(model_str="model_prep.onnx", data=dataset, max_seqs=self.num_seqs, final_skip=self.final_skip)
        quant_options = {
                "CalibMaxIntermediateOutputs": self.num_parallel_seqs,
                "CalibMovingAverage": self.moving_average,
                "CalibTensorRangeSymmetric": self.symmetric,
            }
        if self.smoothing_factor > 0.0:
            quant_options["CalibSmoothRange"] = self.smoothing_factor
        if self.smooth_quant:
            quant_options["SmoothQuant"] = True
        quantize_static(
            model_input="model_prep.onnx",
            model_output=self.out_model.get_path(),
            calibration_data_reader=y,
            calibrate_method=self.calibration_method,
            extra_options=quant_options,
            quant_format=self.quant_format,
            activation_type=self.activation_type,
            weight_type=self.weight_type,
            op_types_to_quantize=self.ops_to_quant,
        )
        import shutil
        if self.final_skip[0] or self.final_skip[1]:
            shutil.move("calibrate_tensors_dev", self.out_dev_log)

