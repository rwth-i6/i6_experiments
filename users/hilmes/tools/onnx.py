import sys
import os
from sisyphus import Job, Task, tk
from typing import Any, Dict, Optional, Tuple, List, Union
import logging

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from onnxruntime.quantization import quant_pre_process, quantize_static, CalibrationDataReader, CalibrationMethod, QuantType, QuantFormat
from onnxruntime import InferenceSession, SessionOptions
from returnn.datasets import Dataset, init_dataset
from returnn.datasets.hdf import HDFDataset
from returnn.datasets.meta import MetaDataset
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
        
        model_state = torch.load(str(self.pytorch_checkpoint), map_location=torch.device("cpu"))
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

        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        missing_keys = [k for k in missing_keys if k not in config.typed_value("save_ignore_params", []) and not any(
            k.startswith(prefix) for prefix in config.typed_value("save_ignore_params_prefixes", []))]
        assert len(missing_keys) == len(unexpected_keys) == 0, ("Some keys where not found", missing_keys, unexpected_keys)

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
        "percentile": None,
        "num_bins": None,
        "random_seed": 0,
        "filter_opts": None
    }

    def __init__(self,
        model: tk.Path,
        dataset: Dict[str, Any],
        num_seqs: Union[int, str] = 10,
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
        percentile: Optional[float] = None,
        num_bins: Optional[int] = None,
        random_seed: int = 0,
        filter_opts: Optional[Dict[str, Any]] = None,
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
        self.filter_opts = filter_opts

        self.out_model = self.output_path("model.onnx")
        if num_seqs >= 50000:
            time = 48
        elif num_seqs >= 25000:
            time = 24
        elif num_seqs >= 10000:
            time = 16
        elif num_seqs >= 5000:
            time = 12
        elif num_seqs >= 2500:
            time = 6
        elif num_seqs >= 1000:
            time = 4
        else:
            time = 1
        if not calibrate_method == CalibrationMethod.MinMax:
            time *= 2

        self.rqmt = {"cpu": 8 if num_seqs > 100 else 4, "mem": 16.0 if calibrate_method == CalibrationMethod.MinMax else 64, "time": time}
        self.calibration_method = calibrate_method
        self.percentile = percentile
        self.num_bins = num_bins
        self.random_seed = random_seed
        if percentile or num_bins:
            assert self.calibration_method == CalibrationMethod.Percentile
        self.smoothing_factor = smoothing_factor
        self.symmetric = symmetric
        self.final_skip = final_skip
        self.smooth_quant = smooth_quant
        self.ops_to_quant = ops_to_quant
        self.out_dev_log = self.output_path("dev_log")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def convert_to_str(self, dataset: Dict):
        res = {}
        for x in dataset:
            if isinstance(dataset[x], dict):
                res[x] = self.convert_to_str(dataset[x])
            elif isinstance(dataset[x], tk.Path):
                res[x] = str(dataset[x])
            else:
                res[x] = dataset[x]
        return res

    def run(self):
        logging.info("Start Prep")
        quant_pre_process(
            input_model_path=self.model.get_path(),
            output_model_path="model_prep.onnx")
        logging.info("Start Quant")
        seed = self.random_seed
        import random
        random.seed(seed)
        class DummyDataReader(CalibrationDataReader):

            def __init__(self, model_str: str, data: Union[Dataset, MetaDataset], max_seqs: int, final_skip:  Optional[Tuple[int, int]] = (None, None), filter_opts: Optional[Dict[str, Any]] = None):

                self.max_seqs = max_seqs
                self.data = data
                self.counter: int = 0
                sess_option = SessionOptions()
                logging.info(f"Data Loading {os.getenv('SLURM_CPUS_PER_TASK')}")
                sess_option.intra_op_num_threads = int(os.getenv('SLURM_CPUS_PER_TASK'))
                session = InferenceSession(model_str, sess_option)
                self.input_name_1 = session.get_inputs()[0].name
                inputs = []
                for x in session.get_inputs():
                    inputs.append(x.name)
                logging.info(f"Session Inputs: {inputs}")
                self.input_name_2 = session.get_inputs()[1].name if len(session.get_inputs()) > 1 else None
                self.final_skip_step = final_skip[0]
                self.final_skip_count = final_skip[1]
                self.seen_seqs = []
                self.filter_opts = filter_opts
                self.visited_seqs = set()

            def get_next(self):
                key = "data" if "data" in self.data.get_data_keys() else "raw_audio"  # hack to make it compatible with both setups for now
                seq_number = None
                if not self.data.is_less_than_num_seqs(self.counter) or self.counter >= self.max_seqs:
                    if not self.data.is_less_than_num_seqs(self.counter):
                        logging.info(f"Finished after {self.counter} sequences")
                        return None
                    elif self.final_skip_step is not None and self.counter < self.max_seqs + self.final_skip_step * self.final_skip_count:
                        logging.info("Drawing skip step")
                        for _ in range(self.final_skip_step):
                            seq_number = random.randint(0, self.data.num_seqs - 1)
                            self.visited_seqs.add(seq_number)
                        while seq_number in self.seen_seqs:
                            seq_number = random.randint(0, self.data.num_seqs - 1)
                            self.visited_seqs.add(seq_number)
                            assert len(self.visited_seqs) < self.data.num_seqs, "Visited all sequences"
                    else:
                        logging.info("Seen all sequences in dataset")
                        return None
                if seq_number is None:
                        while not seq_number or seq_number in self.seen_seqs or not self.check_filter(seq_number):
                            seq_number = random.randint(0, self.data.num_seqs - 1)
                            self.visited_seqs.add(seq_number)
                            assert len(self.visited_seqs) < self.data.num_seqs, "Visited all sequences"
                self.seen_seqs.append(seq_number)
                self.data.load_seqs(seq_number, seq_number+1)
                data: np.ndarray = self.data.get_data(seq_number, key)
                seq_len: np.ndarray = self.data.get_seq_length(seq_number)[key]
                logging.info(f"Next Seq Tag {self.data.get_tag(seq_number)} with idx number {seq_number} and len {seq_len}")
                if self.counter % 10 == 0:
                    logging.info(f"{self.counter} seqs seen")
                data = np.expand_dims(data, axis=0)
                if self.input_name_2 is not None:
                    assert seq_len == data.shape[1], (data.shape, seq_len)
                    seq_len = np.array([seq_len], dtype=np.int32)
                    self.counter += 1
                    return {self.input_name_1: data, self.input_name_2: seq_len}
                else:
                    self.counter += 1
                    return {self.input_name_1: data}

            def check_filter(self, seq_number) -> bool:
                if self.filter_opts is not None:
                    for name, value in self.filter_opts.items():
                        if name == "max_seq_len":
                            seq_len = self.data.get_seq_length(seq_number)["data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            if seq_len > value:
                                logging.info(
                                    f"FILTER: Seq {self.data.get_tag(seq_number)} has length {seq_len} longer than {value}")
                                return False
                        elif name == "min_seq_len":
                            seq_len = self.data.get_seq_length(seq_number)[
                                "data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            if seq_len < value:
                                logging.info(
                                    f"FILTER: Seq {self.data.get_tag(seq_number)}has length {seq_len} shorter than {value}")
                                return False
                        elif name == "partition":
                            seq_len = self.data.get_seq_length(seq_number)[
                                "data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            for lower, upper in value:
                                if seq_len > upper or seq_len < lower:
                                    continue
                                logging.info(f"FILTER: Removing {(lower, upper)} for Seq {self.data.get_tag(seq_number)} of length {seq_len}")
                                value.remove((lower, upper))
                                logging.info(value)
                                return True
                            logging.info(
                                f"FILTER: {self.data.get_tag(seq_number)} of length {seq_len} not matching {value}")
                            return False
                        else:
                            raise NotImplementedError
                return True

        self.dataset = self.convert_to_str(self.dataset)
        dataset: Dataset = init_dataset(self.dataset)
        dataset.init_seq_order(1)

        y = DummyDataReader(model_str="model_prep.onnx", data=dataset, max_seqs=self.num_seqs, final_skip=self.final_skip, filter_opts=self.filter_opts)
        quant_options = {
                "CalibMaxIntermediateOutputs": self.num_parallel_seqs,
                "CalibMovingAverage": self.moving_average,
                "CalibTensorRangeSymmetric": self.symmetric,
            }
        if self.smoothing_factor > 0.0:
            quant_options["CalibSmoothRange"] = self.smoothing_factor
        if self.smooth_quant:
            quant_options["SmoothQuant"] = True
        if self.num_bins:
            quant_options["CalibNumBins"] = self.num_bins
        if self.percentile:
            quant_options["CalibPercentile"] = self.percentile
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

