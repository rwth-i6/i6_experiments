"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation
v2 adds cuda sync
"""

from dataclasses import dataclass
from sisyphus import tk
import time
from time import perf_counter
import numpy as np
from typing import Optional, Union, List, Protocol
import paho.mqtt.client as paho
import os, json

class TracebackItem(Protocol):
    lemma: str
    am_score: float
    lm_score: float
    start_time: int
    end_time: int

def _traceback_to_string(traceback: List[TracebackItem]) -> str:
    traceback_str = " ".join(item.lemma for item in traceback)
    traceback_str = traceback_str.replace("<s>", "")
    traceback_str = traceback_str.replace("</s>", "")
    traceback_str = traceback_str.replace("<blank>", "")
    traceback_str = traceback_str.replace("[BLANK] [1]", "")
    traceback_str = traceback_str.replace("[BLANK]", "")
    traceback_str = traceback_str.replace("<silence>", "")
    traceback_str = traceback_str.replace("[SILENCE]", "")
    traceback_str = traceback_str.replace("[SENTENCE-END]", "")
    traceback_str = " ".join(traceback_str.split())
    return traceback_str


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

    energy_device: str
    # search related options:
    from i6_core.rasr import RasrConfig
    rasr_config_file: Union[str, tk.Path, RasrConfig]
    rasr_post_config: Optional[Union[str, tk.Path, RasrConfig]] = None

    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[str] = None


    turn_off_quant: Union[
        bool, str
    ] = False  # parameter for sanity checks, call self.prep_dequant instead of self.prep_quant

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
    from returnn.util.basic import cf

    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)

    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    run_ctx.rasr_config_file = config.rasr_config_file
    run_ctx.sample_rate = extra_config.sample_rate
    from librasr import Configuration, SearchAlgorithm

    rasr_config = Configuration()
    rasr_config.set_from_file(run_ctx.rasr_config_file)
    run_ctx.search_algorithm = SearchAlgorithm(config=rasr_config)

    run_ctx.blank_log_penalty = config.blank_log_penalty
    if config.prior_file:
        run_ctx.prior = np.loadtxt(config.prior_file, dtype="float32")
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None


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
    run_ctx.energy_file = open("energy", "wt")
    run_ctx.start_time = time.time()
    run_ctx.started_capture = False


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
    run_ctx.energy_file.write(str(ws))
    run_ctx.energy_file.close()


def forward_step(*, model, data, run_ctx, **kwargs):
    if run_ctx.started_capture == False:
        run_ctx.energy_capture.start()
        run_ctx.started_capture = True

    import torch

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch

    hypothesis = []
    encoder_times = []
    search_times = []

    for b in range(raw_audio.shape[0]):
        seq_samples_size = raw_audio_len[b: b + 1]
        seq_samples = raw_audio[b: b + 1, : seq_samples_size[0]]  # [1, T, 1]

        encoder_start = perf_counter()
        encoder_states, audio_features_len = model.forward(seq_samples, seq_samples_size)

        if not isinstance(encoder_states, list):
            encoder_states_cpu = encoder_states.cpu()
        else:
            assert len(encoder_states) == 1
            encoder_states_cpu = encoder_states[0].cpu()

        if run_ctx.blank_log_penalty is not None:
            # assumes blank is last
            encoder_states_cpu[:, :, -1] -= run_ctx.blank_log_penalty
        if run_ctx.prior is not None:
            encoder_states_cpu -= run_ctx.prior_scale * run_ctx.prior

        encoder_states_cpu = -encoder_states_cpu # RASR wants negative logprobs
        encoder_time = perf_counter() - encoder_start
        encoder_times.append(encoder_time)
        if torch.cuda.is_available():
            torch.cuda.synchronize(run_ctx.device)
        else:
            torch.cpu.synchronize(run_ctx.device)

        search_start = perf_counter()
        traceback = run_ctx.search_algorithm.recognize_segment(features=encoder_states_cpu)
        search_time = perf_counter() - search_start
        search_times.append(search_time)

        recog_str = _traceback_to_string(traceback)
        hypothesis.append(recog_str)

    search_time = sum(search_times)
    run_ctx.total_search_time += search_time
    am_time = sum(encoder_times)
    run_ctx.total_am_time += am_time

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    tags = data["seq_tag"]
    for hyp, tag in zip(hypothesis, tags):
        if run_ctx.print_hypothesis:
            print(hyp)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(hyp)))
