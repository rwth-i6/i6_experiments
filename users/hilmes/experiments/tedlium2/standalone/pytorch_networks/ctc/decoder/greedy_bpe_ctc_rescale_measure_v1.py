"""
Greedy CTC decoder without any extras

"""
from dataclasses import dataclass
import time
import torch
from typing import Union
from sisyphus import tk
import paho.mqtt.client as paho
import os
import json

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
    returnn_vocab: Union[str, tk.Path]

    energy_device: str

    turn_off_quant: Union[
        bool, str
    ] = "leave_as_is"  # parameter for sanity checks, call self.prep_dequant instead of self.prep_quant


@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = ExtraConfig(**extra_config_dict)
    
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    run_ctx.print_rtf = extra_config.print_rtf
    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s = 0
        run_ctx.total_time = 0
        run_ctx.rtf_file = open("rtf", "wt")

    run_ctx.print_hypothesis = extra_config.print_hypothesis

    if config.turn_off_quant is False:
        run_ctx.engine._model.prep_quant()
    elif config.turn_off_quant == "decomposed":
        run_ctx.engine._model.prep_quant(decompose=True)
        print("Use decomposed version, should match training")
    elif config.turn_off_quant == "leave_as_is":
        print("Use same version as in training")
    else:
        raise NotImplementedError
        run_ctx.engine._model.prep_dequant()

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

    print("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))
    run_ctx.rtf_file.write("Total-time: %.2f, Batch-RTF: %.3f" % (run_ctx.total_time, run_ctx.total_time / run_ctx.running_audio_len_s))
    run_ctx.rtf_file.close()

    total_time, ws = run_ctx.energy_capture.finish()
    print(f"{run_ctx.energy_device} Energy: {ws} ws")
    run_ctx.energy_file.write(str(ws))
    run_ctx.energy_file.close()

def forward_step(*, model, data, run_ctx, **kwargs):
    if run_ctx.started_capture == False:
        run_ctx.energy_capture.start()
        run_ctx.started_capture = True

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s += audio_len_batch
        am_start = time.time()

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    if isinstance(logprobs, list):
        logprobs = logprobs[-1]

    batch_indices = []
    for lp, l in zip(logprobs, audio_features_len):
        batch_indices.append(torch.unique_consecutive(torch.argmax(lp[:l], dim=-1), dim=0).detach().cpu().numpy())

    if run_ctx.print_rtf:
        am_time = time.time() - am_start
        run_ctx.total_time += am_time
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time, am_time / audio_len_batch))

    tags = data["seq_tag"]

    for indices, tag in zip(batch_indices, tags):
        sequence = [run_ctx.labels[idx] for idx in indices if idx < len(run_ctx.labels)]
        sequence = [s for s in sequence if (not s.startswith("<") and not s.startswith("["))]
        text = " ".join(sequence).replace("@@ ", "")
        if run_ctx.print_hypothesis:
            print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))
