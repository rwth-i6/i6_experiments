import json
from dataclasses import asdict

from sisyphus import *

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.data.label_config import \
    LabelConfig
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.network.decoder_config import \
    DecoderConfig


class Qwen2DecoderConfigJobV2(Job):
    def __init__(self, decoder_config: DecoderConfig, label_config: LabelConfig, target_filename: str):
        """
        Modify qwen2decoder configuration.

        """
        self.decoder_config = decoder_config
        self.label_config = label_config

        self.target_filename = target_filename

        self.out_file = self.output_path(self.target_filename)
        if self.target_filename:
            self.set_vis_name(target_filename)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        config = asdict(self.decoder_config)

        config["bos_token_id"] = self.label_config.bos_idx
        config["eos_token_id"] = self.label_config.eos_idx
        config["vocab_size"] = self.label_config.vocab_size

        with open(self.out_file, "w") as f:
            json.dump(config, f, indent=4)

