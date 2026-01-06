"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation

v2 adds prep quant and adds detach to RTF
v3 removes prep quant
v5 has new ip
v7 removes overhead code
"""

from dataclasses import dataclass
import time
import numpy as np
from typing import Any, Dict, Optional, Union
import abc
import dataclasses
import logging
import multiprocessing
import subprocess
import os
from timeit import default_timer as timer
from typing import Optional, Dict

import json
import time
import os
import paho.mqtt.client as paho
from s_tui.sources.rapl_read import get_power_reader

class ProfilingMeter(abc.ABC):
    @abc.abstractmethod
    def get_output(self) -> Optional[Dict]:
        raise NotImplementedError()

    @abc.abstractmethod
    def update_stage(self, stage):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def meter_name(self) -> str:
        raise NotImplementedError()

@dataclasses.dataclass
class TimeMeter(ProfilingMeter):
    last_stage: Optional[str] = None
    last_time: Optional[float] = None  # in seconds
    total_measurements: Dict[str, float] = dataclasses.field(default_factory=dict)  # time per stage

    def get_output(self):
        return {
            "total_time_in_search": sum(seconds for stage, seconds in self.total_measurements.items() if stage.startswith("search-")),
            "measurements": self.total_measurements
        }

    def update_stage(self, stage):
        current_time = timer()
        if self.last_stage is not None:
            assert self.last_time is not None
            self.total_measurements.setdefault(self.last_stage, 0)
            self.total_measurements[self.last_stage] += current_time - self.last_time
        self.last_time = current_time
        self.last_stage = stage

    @property
    def meter_name(self) -> str:
        return "time"


@dataclasses.dataclass
class EnergyMeter(ProfilingMeter, abc.ABC):
    last_time: Optional[float] = None  # in seconds
    total_time: float = 0  # in seconds
    total_num_polls: int = 0
    total_measurements: Dict[str, float] = dataclasses.field(default_factory=dict)  # energy usage in joules (= watt * seconds), per each "profiling_stage"

    def __post_init__(self):
        self.queue_to_child = multiprocessing.Queue()
        self.queue_from_child = multiprocessing.Queue()

        logging.info(f"Starting child process reading power for {self}")
        self.process = multiprocessing.Process(target=self._subprocess_loop, args=(self.queue_to_child, self.queue_from_child))
        self.process.daemon = True
        self.process.start()

    def get_output(self) -> Optional[Dict]:
        if self.process.is_alive():
            self.queue_to_child.put(None)  # None signals to return the output
            return self.queue_from_child.get(block=True)
        else:
            return None

    def update_stage(self, stage):
        self.queue_to_child.put_nowait(stage)

    @abc.abstractmethod
    def _read_power_diff(self, current_time: float) -> Optional[float]:
        # returning None stops the subprocess, e.g. when power reading is not available here.
        raise NotImplementedError()

    def _compile_outputs(self):
        if self.total_num_polls > 0:
            return {
                "total_measurement_time": self.total_time,
                "total_num_polls": self.total_num_polls,
                "mean_time_per_poll": self.total_time / self.total_num_polls,
                "total_energy_in_search": sum(
                    joule for stage, joule in self.total_measurements.items() if stage.startswith("search-")),
                "measurements": self.total_measurements
            }
        else:
            return None

    def _subprocess_loop(self, queue_from_parent: multiprocessing.Queue, queue_to_parent: multiprocessing.Queue):
        current_stage = "startup"
        while True:
            if not queue_from_parent.empty():
                current_stage = queue_from_parent.get_nowait()
                if current_stage is None:  # None signals to return the current output and stop the process
                    logging.info(f"{self} got exit signal! Stopping subprocess.")
                    queue_to_parent.put(self._compile_outputs())
                    return
            current_time = timer()
            power_diff = self._read_power_diff(current_time=current_time)
            if power_diff is None:
                return  # abort
            if self.last_time is not None and power_diff > 0:
                time_diff = current_time - self.last_time
                self.total_time += time_diff
                self.total_num_polls += 1
                self.total_measurements.setdefault(current_stage, 0)
                self.total_measurements[current_stage] += power_diff
            self.last_time = current_time


@dataclasses.dataclass
class NvidiaSMIEnergyMeter(EnergyMeter):
    def _read_power_diff(self, current_time: float) -> Optional[float]:
        # Get the current watt usage of all visible GPUs
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            # Extract the power usage from the output
            current_watts = sum(float(r) for r in result.stdout.split())
            if self.last_time is not None:
                time_diff = current_time - self.last_time
                return time_diff * current_watts
            else:
                return 0  # ignore this time, it's the initial reading.
        else:
            logging.info("Error running nvidia-smi:", result.stderr)
            return

    @property
    def meter_name(self) -> str:
        return "nvidia-smi-power"


@dataclasses.dataclass
class CPUEnergyMeter(EnergyMeter):
    last_power: Optional[float] = None
    num_failures = 0
    reader = get_power_reader()

    def _read_power_diff(self, current_time: float) -> Optional[float]:
        # Incremental register with CPU power usage
        try:
            assert self.reader is not None, "No reader found"
            power = self.reader.read_power()
            assert len(power) > 0, "No power read"
            result = power[0].current / 1e6 # convert to Watts

            # Extract the power usage from the output
            current_power = float(result)
            if self.last_power is not None:
                if current_power < self.last_power:
                    # it overflowed, ignore this time and reset.
                    power_diff = 0
                    logging.info("CPU power meter overflowed")
                else:
                    power_diff = current_power - self.last_power
            else:
                # initial measurement is also ignored
                power_diff = 0

            self.last_power = current_power
            return power_diff
        except:
            logging.info("Error running CPU power meter:", result)
            self.num_failures += 1
            if self.num_failures >= 100:
                logging.info("Giving up after 100 failures to measure CPU power")
                return None
            return 0  # simply don't count it.

    @property
    def meter_name(self) -> str:
        return "cpu-power"


class I6EnergyCapture:
    def __init__(self, device: str):
        self.time = None
        self.ws = 0
        self.client = paho.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        assert os.environ.get("MQTT_USERNAME") and os.environ.get("MQTT_PASSWORD")
        self.client.username_pw_set(os.environ.get("MQTT_USERNAME"), password=os.environ.get("MQTT_PASSWORD"))
        self.connected = False
        self.device = device
        # self.last_power = None

    def start(self):
        self.client.connect("137.226.116.9", 1883, 60)
        self.client.loop_start()
        start = time.time()
        while not (ok := self.connected) and (time.time() - start < 5.0):  # timeout after 5 seconds
            time.sleep(0.1)
        if not ok:
            raise Exception("Could not connect to MQTT server")
        self.step_time = time.time()
        self.start_time = time.time()

    def _on_connect(self, client, userdata, flags, rc):
        print("EnergyCapture connected to MQTT server with result code " + str(rc))
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        if any(x in self.device for x in ["amd", "intel"]):
            client.subscribe("gude1/#")
            client.subscribe("paho/#")
        elif any(x in self.device for x in ["gpu"]):
            client.subscribe("tele/nous1/SENSOR")
        else:
            raise NotImplementedError
        self.connected = True

    # The callback for when a PUBLISH message is received from the server.
    def _on_message(self, client, userdata, msg):
        print(msg.topic)
        # print(msg.topic+" "+str(msg.payload))
        d = json.loads(msg.payload)
        if "amd" in self.device:
            self.last_power = d["line_in"][0]["voltage"] * d["line_in"][0]["current"]  # in watt = voltage * current
        elif "intel" in self.device:
            self.last_power = d["line_in"][1]["voltage"] * d["line_in"][1]["current"]  # in watt = voltage * current
        elif "gpu" in self.device:
            self.last_power = d["ENERGY"]["Voltage"] * d["ENERGY"]["Current"]
        else:
            raise NotImplementedError
        print(f"current W {self.device}: {self.last_power}")
        current_time = time.time()
        self.ws += self.last_power * (current_time - self.step_time)  # in watt seconds ( = Joule)
        self.step_time = current_time

    def finish(self):
        # update one more time with the latest values
        current_time = time.time()
        self.ws += self.last_power * (current_time - self.step_time)  # in watt seconds ( = Joule)

        self.client.disconnect()
        total_time = time.time() - self.start_time
        return total_time, self.ws


@dataclass
class DecoderConfig:
    # search related options:
    beam_size: int
    beam_size_token: int
    beam_threshold: float

    # needed files
    lexicon: str
    returnn_vocab: str

    energy_device: str

    # additional search options
    lm_weight: float = 0.0
    sil_score: float = 0.0
    word_score: float = 0.0

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[str] = None

    arpa_lm: Optional[str] = None

    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None

    turn_off_quant: Union[
        bool, str
    ] = 'leave_as_is'  # parameter for sanity checks, call self.prep_dequant instead of self.prep_quant


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    """

    :param run_ctx:
    :param kwargs:
    :return:
    """
    import torch
    from torchaudio.models.decoder import ctc_decoder

    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    if config.arpa_lm is not None:
        lm = cf(config.arpa_lm)
    else:
        lm = None

    vocab = Vocabulary.create_vocab(vocab_file=config.returnn_vocab, unknown_label=None)
    labels = vocab.labels

    run_ctx.ctc_decoder = ctc_decoder(
        lexicon=config.lexicon,
        lm=lm,
        lm_weight=config.lm_weight,
        tokens=labels + ["[blank]"],
        blank_token="[blank]",
        sil_token="[blank]",
        unk_word="[unknown]",
        nbest=1,
        beam_size=config.beam_size,
        beam_size_token=config.beam_size_token,
        beam_threshold=config.beam_threshold,
        sil_score=config.sil_score,
        word_score=config.word_score,
    )
    run_ctx.labels = labels
    run_ctx.blank_log_penalty = config.blank_log_penalty

    if config.prior_file:
        run_ctx.prior = np.loadtxt(config.prior_file, dtype="float32")
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None

    if config.use_torch_compile:
        options = config.torch_compile_options or {}
        run_ctx.engine._model = torch.compile(run_ctx.engine._model, **options)

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_am_time = 0
        run_ctx.total_search_time = 0
        run_ctx.rtf_file = open("rtf", "wt")

    run_ctx.print_hypothesis = extra_config.print_hypothesis
    if config.turn_off_quant is False:
        print("Run quantization with torch")
        run_ctx.engine._model.prep_quant()
    elif config.turn_off_quant == "decomposed":
        run_ctx.engine._model.prep_quant(decompose=True)
        print("Use decomposed version, should match training")
    elif config.turn_off_quant == "leave_as_is":
        print("Use same version as in training")
    else:
        raise NotImplementedError
        run_ctx.engine._model.prep_dequant()  # TODO: needs fix
    run_ctx.engine._model.to(device=run_ctx.device)

    if not "gpu" in config.energy_device:
        import psutil
        usage = psutil.cpu_percent(interval=1)
        assert usage> 50, f"Load Script probably not started {usage}"

    run_ctx.energy_device = config.energy_device
    run_ctx.energy_capture = I6EnergyCapture(device=config.energy_device)
    run_ctx.start_time = time.time()
    run_ctx.started_capture = False

    run_ctx.meters = [TimeMeter(), CPUEnergyMeter()]
    if "gpu" in config.energy_device:
        run_ctx.meters.append(NvidiaSMIEnergyMeter())


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.recognition_file.write("}\n")
    run_ctx.recognition_file.close()

    if run_ctx.print_rtf:
        print(
            "Total-AM-Time: %.2fs, AM-RTF: %.6f"
            % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
        )
        run_ctx.rtf_file.write(
            "Total-AM-Time: %.2fs, AM-RTF: %.6f \n"
            % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
        )
        print(
            "Total-Search-Time: %.2fs, Search-RTF: %.6f"
            % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
        )
        run_ctx.rtf_file.write(
            "Total-Search-Time: %.2fs, Search-RTF: %.6f \n"
            % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
        )
        total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
        print(
            "Total-time: %.2f, Total-recog-time: %.2f, Batch-RTF: %.6f"
            % (time.time() - run_ctx.start_time, total_proc_time, total_proc_time / run_ctx.running_audio_len_s)
        )
        run_ctx.rtf_file.write(
            "Total-time: %.2f, Total-recog-time: %.2f, Batch-RTF: %.6f \n"
            % (time.time() - run_ctx.start_time, total_proc_time, total_proc_time / run_ctx.running_audio_len_s)
        )
        run_ctx.rtf_file.close()

    total_time, ws = run_ctx.energy_capture.finish()
    print(f"{run_ctx.energy_device} Energy: {ws} ws")
    run_ctx.energy_file = open("energy", "wt")
    run_ctx.energy_file.write(str(ws))
    run_ctx.energy_file.close()

    for meter in run_ctx.meters:
        meter.update_stage("done")

    all_outputs = {meter.meter_name: meter.get_output() for meter in run_ctx.meters}
    if any(x is None for x in all_outputs.values()):
        assert False, f"Measurements went wrong {all_outputs}"
    print(f"Energy measurements are: {all_outputs}")

    with open("energy_software", "wt") as f:
        json.dump(all_outputs, f, indent=2)
    print(f"Wrote energy measurements to energy_software")

def forward_step(*, model, data, run_ctx, **kwargs):
    if run_ctx.started_capture == False:
        run_ctx.energy_capture.start()
        for meter in run_ctx.meters:
            meter.update_stage("run")
        run_ctx.started_capture = True

    import torch

    raw_audio = data["raw_audio"]  # [B, T', F]

    raw_audio_len = data["raw_audio:size1"]  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch
    am_start = time.time()
    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    tags = data["seq_tag"]

    if isinstance(logprobs, list):
        assert len(logprobs) == 1
        logprobs = logprobs[0]

    logprobs_cpu = logprobs.cpu()
    if run_ctx.blank_log_penalty is not None:
        # assumes blank is last
        logprobs_cpu[:, :, -1] -= run_ctx.blank_log_penalty
    if run_ctx.prior is not None:
        logprobs_cpu -= run_ctx.prior_scale * run_ctx.prior

    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time
    if torch.cuda.is_available():
        torch.cuda.synchronize(run_ctx.device)
    else:
        torch.cpu.synchronize(run_ctx.device)

    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())
    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.6f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.6f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.6f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        if run_ctx.print_hypothesis:
            print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
