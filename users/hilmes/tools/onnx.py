import sys
import os
from sisyphus import Job, Task, tk
from typing import Any, Dict, Optional, Tuple, List, Union
import logging

from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from onnxruntime.quantization import quant_pre_process, quantize_static, CalibrationDataReader, CalibrationMethod, QuantType, QuantFormat
from onnxruntime import InferenceSession, SessionOptions
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
        "filter_opts": None,
        "loss_table": None
    }

    def __init__(self,
        model: tk.Path,
        dataset: Dict[str, Any],
        num_seqs: Optional[Union[int, str]] = 10,
        num_parallel_seqs: Optional[int] = 25,
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
        loss_table: Optional[Tuple[tk.Path, Any]] = None,  # Path to loss table + args
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
        self.loss_table = loss_table

        self.out_model = self.output_path("model.onnx")
        if num_seqs is None:
            assert "budget" in filter_opts
            assert filter_opts['budget'][0] // 1000 < 1000, "Might need more time for this one"
            time = 4
        elif num_seqs >= 75000:
            time = 60
        elif num_seqs >= 50000:
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
        # cpu slow bandaid
        if time >= 1:
            time += 5
        if not calibrate_method == CalibrationMethod.MinMax:
            time *= 2
        if self.filter_opts is not None and "single_tag" in self.filter_opts:
            time += 1


        self.rqmt = {"cpu": 8 if num_seqs is not None and num_seqs > 100 else 4, "mem": 16.0 if calibrate_method == CalibrationMethod.MinMax else 100, "time": time}
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
        self.budget = (None, None) if filter_opts is None else filter_opts.get("budget", (None, None))
        self.out_dev_log = self.output_path("dev_log")
        self.out_seq_info = self.output_path("seq_info")
        self.out_num_seqs = self.output_var("num_seqs")
        self.out_left_bud = self.output_var("unused_budget")
        self.num_seqs = self.num_seqs or 10000000

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
        from returnn.datasets import Dataset, init_dataset
        from returnn.datasets.hdf import HDFDataset
        from returnn.datasets.meta import MetaDataset
        logging.info("Start Prep")
        quant_pre_process(
            input_model_path=self.model.get_path(),
            output_model_path="model_prep.onnx")
        logging.info("Start Quant")
        seed = self.random_seed
        import random
        random.seed(seed)
        loss_table = None
        if self.loss_table is not None:
            loss_table = []
            with open(self.loss_table[0], "rt") as f:
                for line in f:
                    loss_table.append(line.split(" "))  # (Tag, loss)
            if "reverse" in self.loss_table[1]:
                loss_table.reverse()

        class DummyDataReader(CalibrationDataReader):

            def __init__(self,
                         model_str: str,
                         data: Union[Dataset, MetaDataset],
                         max_seqs: int, final_skip:  Optional[Tuple[int, int]] = (None, None),
                         filter_opts: Optional[Dict[str, Any]] = None,
                         open_budget: Tuple[Optional[int], Optional[float]] = (None, None),
                         loss_table: Optional[List[Tuple[str, str]]] = None

                ):

                self.max_seqs = max_seqs
                self.data = data
                self.counter: int = 0
                sess_option = SessionOptions()
                logging.info(f"Data Loading {os.getenv('SLURM_CPUS_PER_TASK', 4)}")
                sess_option.intra_op_num_threads = int(os.getenv('SLURM_CPUS_PER_TASK', 4))
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
                self.seq_infos = []
                self.filter_opts = filter_opts
                self.visited_seqs = set()
                self.open_budget = open_budget[0]
                self.budget_thresh = None if open_budget[0] is None else open_budget[0] * open_budget[1]
                self.loss_table = loss_table

            def compare_budget(self):
                if self.budget_thresh is None or self.open_budget is None:
                    return False
                else:
                    return self.budget_thresh > self.open_budget

            def get_next(self):
                if isinstance(self.data, MetaDataset):
                    return self.get_next_meta()
                else:
                    return self.get_next_hdf()

            def get_next_meta(self):
                key = "raw_audio"
                if not self.data.is_less_than_num_seqs(self.counter) or self.counter >= self.max_seqs:
                    return None
                else:
                    seq_number = self.counter
                    self.data.load_seqs(seq_number, seq_number + 1)
                    data: np.ndarray = self.data.get_data(seq_number, key)
                    seq_len: np.ndarray = self.data.get_seq_length(seq_number)[key]
                    self.seq_infos.append((self.data.get_tag(seq_number), seq_len))
                    assert self.data.get_tag(seq_number) not in self.seen_seqs, "In the base case this should never happen"
                    self.seen_seqs.append(self.data.get_tag(seq_number))
                    logging.info(
                        f"Next Seq Tag {self.data.get_tag(seq_number)} with idx number {seq_number} and len {seq_len}")
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


            def get_next_hdf(self):
                key = "data" if "data" in self.data.get_data_keys() else "raw_audio"  # hack to make it compatible with both setups for now
                seq_number = None
                if not self.data.is_less_than_num_seqs(
                        self.counter) or self.counter >= self.max_seqs or self.compare_budget():
                    if self.data.is_less_than_num_seqs(self.counter) and self.final_skip_step is None:
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
                            assert len(
                                self.visited_seqs) <= self.data.num_seqs, f"Visited all sequences {len(self.visited_seqs)} vs. {self.data.num_seqs}"
                    else:
                        logging.info("Seen all sequences in dataset")
                        return None
                if self.loss_table is not None:
                    seq_number = self.counter
                    name = self.loss_table[seq_number][0]
                    real_seq_number = self.data.get_all_tags().index(name)
                    while real_seq_number in self.seen_seqs or not self.check_filter(real_seq_number):
                        seq_number += 1
                        name = self.loss_table[seq_number][0]
                        real_seq_number = self.data.get_all_tags().index(name)
                        assert self.loss_table[seq_number][0] == self.data.get_tag(real_seq_number), (
                        self.loss_table[seq_number][0], self.data.get_tag(real_seq_number))
                        logging.info(
                            f"Position {seq_number} is real {real_seq_number} with Tag {self.data.get_tag(real_seq_number)} matching {self.loss_table[seq_number]}")
                        self.visited_seqs.add(real_seq_number)
                        if self.open_budget is not None and self.open_budget == 0:
                            assert False, "This path should not be reached"
                            logging.info("Budget Full")
                            return None
                        if len(self.visited_seqs) == self.data.num_seqs:
                            self.visited_seqs = set()
                            self.open_budget += 1
                            seq_number = seq_number
                    seq_number = real_seq_number
                if seq_number is None:
                    while not seq_number or seq_number in self.seen_seqs or not self.check_filter(seq_number):
                        seq_number = random.randint(0, self.data.num_seqs - 1)
                        self.visited_seqs.add(seq_number)  # +2 because seen seqs has not been updated
                        logging.info(f"{len(self.visited_seqs)} {self.data.num_seqs} {len(self.seen_seqs)}")
                        logging.info(
                            f"{seq_number}, {seq_number in self.seen_seqs} {not self.check_filter(seq_number)}")
                        logging.info(
                            f"{len(self.visited_seqs) == self.data.num_seqs} {len(self.seen_seqs) + 2 == self.data.num_seqs} {len(self.seen_seqs) + 1 == self.data.num_seqs}")
                        if len(self.visited_seqs) == self.data.num_seqs and (
                                len(self.seen_seqs) + 2 == self.data.num_seqs or len(
                                self.seen_seqs) + 1 == self.data.num_seqs):
                            return None
                        if len(self.visited_seqs) == self.data.num_seqs and not (
                                len(self.seen_seqs) + 2 == self.data.num_seqs or len(
                                self.seen_seqs) + 1 == self.data.num_seqs) and any(
                                x in self.filter_opts for x in ["single_tag", "unique_tags"]):
                            return None
                        if self.open_budget is not None and self.open_budget == 0:
                            logging.info("Budget Full")
                            return None
                        if len(self.visited_seqs) == self.data.num_seqs and not (
                                len(self.seen_seqs) + 2 == self.data.num_seqs or len(
                                self.seen_seqs) + 1 == self.data.num_seqs):
                            self.visited_seqs = set()
                            self.open_budget += 1
                        # assert len(self.visited_seqs) < self.data.num_seqs, "Visited all sequences"
                self.seen_seqs.append(seq_number)
                logging.info(len(self.seen_seqs))
                # if isinstance(self.data, MetaDataset):
                # if not self.data.expected_load_seq_start > seq_number:
                self.data.load_seqs(seq_number, seq_number + 1)
                # else:
                #    self.data.load_seqs(seq_number, seq_number+1)
                data: np.ndarray = self.data.get_data(seq_number, key)
                seq_len: np.ndarray = self.data.get_seq_length(seq_number)[key]
                self.seq_infos.append((self.data.get_tag(seq_number), seq_len))
                logging.info(
                    f"Next Seq Tag {self.data.get_tag(seq_number)} with idx number {seq_number} and len {seq_len}")
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
                if self.filter_opts is not None and len(self.filter_opts) > 0:
                    for name, value in self.filter_opts.items():
                        if name == "max_seq_len":
                            seq_len = self.data.get_seq_length(seq_number)["data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            if seq_len > value:
                                logging.warning(
                                    f"FILTER: Seq {self.data.get_tag(seq_number)} has length {seq_len} longer than {value}")
                                return False
                        elif name == "min_seq_len":
                            seq_len = self.data.get_seq_length(seq_number)[
                                "data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            if seq_len < value:
                                logging.warning(
                                    f"FILTER: Seq {self.data.get_tag(seq_number)}has length {seq_len} shorter than {value}")
                                return False
                        elif name == "range_len":
                            seq_len = self.data.get_seq_length(seq_number)[
                                "data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            if seq_len > value[0] and seq_len < value[1]:
                                logging.info(
                                    f"FILTER: Seq {self.data.get_tag(seq_number)}has length {seq_len} shorter than {value}")
                                return True
                            else:
                                logging.warning(
                                    f"FILTER: Seq {self.data.get_tag(seq_number)}has length {seq_len} not in range {value}")
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
                            logging.warning(
                                f"FILTER: {self.data.get_tag(seq_number)} of length {seq_len} not matching {value}")
                            return False
                        elif name == "budget":
                            seq_len = self.data.get_seq_length(seq_number)[
                                "data" if "data" in self.data.get_data_keys() else "raw_audio"]
                            if seq_len <= self.open_budget:
                                logging.info(f"FILTER: Seq with len {seq_len} within budget {self.open_budget}")
                                self.open_budget -= seq_len
                                self.visited_seqs = set()
                                return True
                            else:
                                pass
                                #logging.warning(f"FILTER: Seq with len {seq_len} NOT in budget {self.open_budget}")
                            return False
                        elif name == "unique_tags":
                            seq_tag = self.data.get_tag(seq_number)
                            if self.seq_infos is not None and any(seq_tag.split("/")[1] == info[0].split("/")[1] for info in self.seq_infos):
                                logging.info(f"FILTER: {seq_tag} found in {self.seq_infos}")
                                return False
                            else:
                                return True
                        elif name == "single_tag":
                            seq_tag = self.data.get_tag(seq_number)
                            if self.seq_infos is not None and not all(seq_tag.split("/")[1] == info[0].split("/")[1] for info in self.seq_infos):
                                #logging.info(f"FILTER: {seq_tag} prefix not found in {self.seq_infos[0]}")
                                return False
                            else:
                                return True
                        else:
                            raise NotImplementedError
                if self.open_budget is not None:
                    assert False
                    seq_len = self.data.get_seq_length(seq_number)[
                        "data" if "data" in self.data.get_data_keys() else "raw_audio"]
                    if seq_len < self.open_budget:
                        logging.info(f"FILTER: Seq with len {seq_len} within budget {self.open_budget}")
                        self.open_budget -= seq_len
                        return True
                    else:
                        logging.warning(f"FILTER: Seq with len {seq_len} NOT in budget {self.open_budget}")
                    return False
                return True

        self.dataset = self.convert_to_str(self.dataset)
        if loss_table is not None:
            with open("segments", "wt") as f:
                for tag, loss in loss_table[:self.num_seqs]:
                    f.write(f"{tag}\n")
            #self.dataset["seq_list_filter_file"] = "segments"
        dataset: Dataset = init_dataset(self.dataset)
        dataset.init_seq_order(1)

        y = DummyDataReader(
            model_str="model_prep.onnx",
            data=dataset,
            max_seqs=self.num_seqs,
            final_skip=self.final_skip,
            filter_opts=self.filter_opts,
            open_budget=self.budget,
            loss_table=loss_table,
        )
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
        #if self.final_skip[0] or self.final_skip[1]:
        if not os.path.exists("calibrate_tensors_dev"): # if onnxruntime didn't write the file just put a dummy
            open("calibrate_tensors_dev", "wt").close()
        shutil.move("calibrate_tensors_dev", self.out_dev_log)

        with open("seq_infos", "wt") as f:
            for seq_tag, seq_len in y.seq_infos:
                f.write(f"{seq_tag}: {seq_len}\n")
        shutil.move("seq_infos", self.out_seq_info)
        if y.open_budget is not None:
            self.out_left_bud.set(y.open_budget)
            self.out_num_seqs.set(len(y.seq_infos))
        else:
            self.out_left_bud.set(0)
            self.out_num_seqs.set(len(y.seen_seqs))
