"""
Greedy CTC decoder without any extras

v4 proper LSTM/Trafo support
"""
from dataclasses import dataclass
import torch
from typing import Any, Dict, Optional, Union
import numpy as np
import json
import time
import os
import paho.mqtt.client as paho

from .search.documented_ctc_beam_search_v4 import CTCBeamSearch

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
    returnn_vocab: str

    beam_size: int
    # e.g. "lm.lstm.some_lstm_variant_file.Model"
    lm_module: Optional[str]
    lm_model_args: Dict[str, Any]
    lm_checkpoint: Optional[str]
    lm_states_need_label_axis: bool

    energy_device: str

    prior_scale: float = 0.0
    prior_file: Optional[str] = None

    lm_weight: float = 0.0

    turn_off_quant: Union[
        bool, str
    ] = 'leave_as_is'  # parameter for sanity checks, call self.prep_dequant instead of self.prep_quant
    

@dataclass
class DecoderExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True
    
    # LM model package path
    lm_package: Optional[str] = None


def forward_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    config = DecoderConfig(**kwargs["config"])
    extra_config_dict = kwargs.get("extra_config", {})
    extra_config = DecoderExtraConfig(**extra_config_dict)
    
    run_ctx.recognition_file = open("search_out.py", "wt")
    run_ctx.recognition_file.write("{\n")

    from returnn.datasets.util.vocabulary import Vocabulary
    vocab = Vocabulary.create_vocab(
        vocab_file=config.returnn_vocab, unknown_label=None)
    run_ctx.labels = vocab.labels

    model = run_ctx.engine._model
    run_ctx.beam_size = config.beam_size
    
    if config.prior_file:
        run_ctx.prior = torch.tensor(np.loadtxt(config.prior_file, dtype="float32"), device=run_ctx.device)
        run_ctx.prior_scale = config.prior_scale
    else:
        run_ctx.prior = None

    lm_model = None
    if config.lm_module is not None:
        # load LM
        assert extra_config.lm_package is not None
        lm_module_prefix = ".".join(config.lm_module.split(".")[:-1])
        lm_module_class = config.lm_module.split(".")[-1]

        LmModule = __import__(
            ".".join([extra_config.lm_package, lm_module_prefix]),
            fromlist=[lm_module_class],
        )
        LmClass = getattr(LmModule, lm_module_class)

        lm_model = LmClass(config.lm_model_args)  # TODO: this was changed from nick
        checkpoint_state = torch.load(
            config.lm_checkpoint,
            map_location=run_ctx.device,
        )
        lm_model.load_state_dict(checkpoint_state["model"])
        lm_model.to(device=run_ctx.device)
        lm_model.eval()
        run_ctx.lm_model = lm_model

        print("loaded external LM")
        print()
    CTCBeamSearch
    run_ctx.ctc_decoder = CTCBeamSearch(
        model=model,
        blank=model.train_config.label_target_size,
        device=run_ctx.device,
        lm_model=lm_model,
        lm_scale=config.lm_weight,
        lm_sos_token_index=0,
        lm_states_need_label_axis=config.lm_states_need_label_axis,
        prior=run_ctx.prior,
        prior_scale=run_ctx.prior_scale,

    )
    print("done!")
    

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

    print(
        "Total-AM-Time: %.2fs, AM-RTF: %.4f"
        % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
    )
    run_ctx.rtf_file.write(
        "Total-AM-Time: %.2fs, AM-RTF: %.4f \n"
        % (run_ctx.total_am_time, run_ctx.total_am_time / run_ctx.running_audio_len_s)
    )
    print(
        "Total-Search-Time: %.2fs, Search-RTF: %.4f"
        % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
    )
    run_ctx.rtf_file.write(
        "Total-Search-Time: %.2fs, Search-RTF: %.4f \n"
        % (run_ctx.total_search_time, run_ctx.total_search_time / run_ctx.running_audio_len_s)
    )
    total_proc_time = run_ctx.total_am_time + run_ctx.total_search_time
    print(
        "Total-time: %.2f, Total-recog-time: %.2f, Batch-RTF: %.4f"
        % (time.time() - run_ctx.start_time, total_proc_time, total_proc_time / run_ctx.running_audio_len_s)
    )
    run_ctx.rtf_file.write(
        "Total-time: %.2f, Total-recog-time: %.2f, Batch-RTF: %.4f \n"
        % (time.time() - run_ctx.start_time, total_proc_time, total_proc_time / run_ctx.running_audio_len_s)
    )
    run_ctx.rtf_file.close()

    total_time, ws = run_ctx.energy_capture.finish()
    print(f"{run_ctx.energy_device} Energy: {ws} ws")
    run_ctx.energy_file.write(str(ws))
    run_ctx.energy_file.close()

def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000

    tags = data["seq_tag"]

    if run_ctx.print_rtf:
        run_ctx.running_audio_len_s += audio_len_batch
        am_start = time.time()

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    if isinstance(logprobs, list):
        logprobs = logprobs[-1]


    if run_ctx.print_rtf:
        if torch.cuda.is_available():
            torch.cuda.synchronize(run_ctx.device)
        am_time = time.time() - am_start
        run_ctx.total_am_time += am_time
        search_start = time.time()

    hyps = []
    for lp, l in zip(logprobs, audio_features_len):
        hypothesis = run_ctx.ctc_decoder.forward(lp, l, run_ctx.beam_size)
        hyps.append(hypothesis[0].tokens[1:])
        # hyps = [hypothesis[0].tokens for hypothesis in batched_hypotheses]  # exclude last sentence end token

    if run_ctx.print_rtf:
        search_time = (time.time() - search_start)
        run_ctx.total_search_time += search_time
        print("Batch-AM-Time: %.2fs, AM-RTF: %.3f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.3f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.3f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    for hyp, tag in zip(hyps, tags):
        sequence = [run_ctx.labels[idx] for idx in hyp if idx < len(run_ctx.labels)]
        text = " ".join(sequence).replace("@@ ","")
        if run_ctx.print_hypothesis:
            print(text)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(text)))
