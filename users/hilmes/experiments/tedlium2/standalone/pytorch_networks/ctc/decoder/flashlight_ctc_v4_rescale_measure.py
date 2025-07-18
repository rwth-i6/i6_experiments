"""
Flashlight/Torchaudio CTC decoder

includes handling of prior computation

v2 adds prep quant and adds detach to RTF
v3 removes prep quant
"""

from dataclasses import dataclass
import time
import numpy as np
from typing import Any, Dict, Optional, Union

import json
import time
import os
import paho.mqtt.client as paho


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
        client.subscribe("gude1/#")
        client.subscribe("paho/#")
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
    if run_ctx.started_capture == False:
        run_ctx.energy_capture.start()
        run_ctx.started_capture = True
    import torch

    raw_audio = data["raw_audio"]  # [B, T', F]

    raw_audio_len = data["raw_audio:size1"]  # [B]

    if run_ctx.print_rtf:
        audio_len_batch = torch.sum(raw_audio_len).detach().cpu().numpy() / 16000
        run_ctx.running_audio_len_s += audio_len_batch
    tmp = raw_audio.detach().numpy()
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

    tmp = logprobs_cpu.detach().numpy()
    am_time = time.time() - am_start
    run_ctx.total_am_time += am_time

    search_start = time.time()
    hypothesis = run_ctx.ctc_decoder(logprobs_cpu, audio_features_len.cpu())
    search_time = time.time() - search_start
    run_ctx.total_search_time += search_time

    if run_ctx.print_rtf:
        print("Batch-AM-Time: %.2fs, AM-RTF: %.4f" % (am_time, am_time / audio_len_batch))
        print("Batch-Search-Time: %.2fs, Search-RTF: %.4f" % (search_time, search_time / audio_len_batch))
        print("Batch-time: %.2f, Batch-RTF: %.4f" % (am_time + search_time, (am_time + search_time) / audio_len_batch))

    for hyp, tag in zip(hypothesis, tags):
        words = hyp[0].words
        sequence = " ".join([word for word in words if not word.startswith("[")])
        if run_ctx.print_hypothesis:
            print(sequence)
        run_ctx.recognition_file.write("%s: %s,\n" % (repr(tag), repr(sequence)))
