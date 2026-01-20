import yaml
import os
import math
import kenlm
from copy import deepcopy
import os
import random
import subprocess as sp
from typing import Optional, Type, Dict, Any
from sisyphus import Job, Task, tk
import logging
import numpy as np
from sisyphus import tools
import shutil
from recipe.i6_experiments.users.enrique.jobs.gan.utils import cache_path, resolve_tk_paths, flatten_dictionary_to_text

class GANw2vGenerateJob(Job):
    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}
    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path], 
        decoding_config: Dict[str, Any],
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.decoding_config  = decoding_config
        
        self.config_dir = self.output_path("config", directory=True)
        self.config_file = self.output_path("config/decoding_config.yaml")
        
        self.results_path = self.output_path("transcriptions", directory=True)


        gen_subset_str = str(self.decoding_config["fairseq"]["dataset"]["gen_subset"])
        num_shards_int = int(self.decoding_config["fairseq"]["dataset"]["num_shards"])
        shard_id_str = str(self.decoding_config["fairseq"]["dataset"].get("shard_id", 0)) + "_" if num_shards_int > 1 else ""

        self.output_trans_units = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_units.txt")
        self.output_ref = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_ref.txt")
        self.output_trans = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}.txt")
        self.output_ref_units = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_ref_units.txt")
        self.nbest_words = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_nbest_words.txt")

        if decoding_config.get("w2l_decoder", None) == "KALDI":
            self.rqmt = {"time": 2000, "cpu": 1, "mem": 64}
        else:
            self.rqmt = {"time": 1000, "cpu": 1, "mem": 16}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):

        decoding_config = deepcopy(self.decoding_config)

        # Cache the paths for the task data, text data, and KenLM binary
        #decoding_config['fairseq']['task']['data'] = cache_path(decoding_config['fairseq']['task']['data'].get_path())
        decoding_config['fairseq']['common']['user_dir'] = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised")
        decoding_config['results_path'] = self.results_path

        decoding_config = resolve_tk_paths(decoding_config)
        
        with open(self.config_file.get_path(), "w") as f:
            yaml.dump(decoding_config, f)
        
        sh_call_str = f"""
            source {self.fairseq_python_env.get_path()}/bin/activate && \
            export PYTHONNOUSERSITE=1 && \
            export HYDRA_FULL_ERROR=1 && \
            /opt/conda/bin/python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/w2vu_generate.py \
            --config-dir {self.config_dir.get_path()} \
            --config-name decoding_config \
        """
    
        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:
        """
        :param parsed_args:
        :return: hash for job given the arguments
        """
        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 


