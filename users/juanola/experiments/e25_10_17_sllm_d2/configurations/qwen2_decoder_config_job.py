import copy
import json

from sisyphus import *

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.qwen2.qwen2_configs import qwen2_configs


class Qwen2DecoderConfigJob(Job):
    def __init__(self, bos:int, eos:int, vocab_size:int, target_filename: str):
        """
        Modify qwen2decoder configuration.

        """
        self.bos = bos
        self.eos = eos
        self.vocab_size = vocab_size
        self.target_filename = target_filename

        self.out_file = self.output_path(self.target_filename)
        if self.target_filename:
            self.set_vis_name(target_filename)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        config = copy.deepcopy(qwen2_configs)

        config["bos_token_id"] = self.bos
        config["eos_token_id"] = self.eos
        config["vocab_size"] = self.vocab_size

        with open(self.out_file, "w") as f:
            json.dump(config, f)

