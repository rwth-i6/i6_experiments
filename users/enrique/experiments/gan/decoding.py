import os
import math
import kenlm
from copy import deepcopy
import os
from random import random
import subprocess as sp
from typing import Optional, Type, Dict, Any
from sisyphus import Job, Task, tk
import logging
import numpy as np
from sisyphus import tools
import shutil
from recipe.i6_experiments.users.enrique.experiments.gan.utils import cache_path, resolve_tk_paths, flatten_dictionary_to_text

class GANw2vGenerateJob(Job):
    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        decoding_config: Dict[str, Any],
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.decoding_config  = decoding_config
        
        self.results_path = self.output_path("transcriptions", directory=True)


        gen_subset_str = str(self.decoding_config["fairseq"]["dataset"]["gen_subset"])
        num_shards_int = int(self.decoding_config["fairseq"]["dataset"]["total_shards"])
        shard_id_str = str(self.decoding_config["fairseq"]["dataset"].get("shard_id", 0)) + "_" if num_shards_int > 1 else ""

        self.output_trans_units = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_units.txt")
        self.output_ref = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_ref.txt")
        self.output_trans = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}.txt")
        self.output_ref_units = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_ref_units.txt")
        self.nbest_words = self.output_path(f"transcriptions/{gen_subset_str}{shard_id_str}_nbest_words.txt")

        self.rqmt = {"time": 1000, "cpu": 1, "gpu": 1, "mem": 16}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):

        final_fairseq_hydra_config = deepcopy(self.fairseq_hydra_config)

        # Cache the paths for the task data, text data, and KenLM binary
        final_fairseq_hydra_config['fairseq']['task']['data'] = cache_path(self.fairseq_hydra_config['fairseq']['task']['data'].get_path())
        final_fairseq_hydra_config['fairseq']['common']['user_dir'] = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised")

        final_fairseq_hydra_config = resolve_tk_paths(final_fairseq_hydra_config)
        final_fairseq_hydra_config_str = flatten_dictionary_to_text(final_fairseq_hydra_config)
        
        sh_call_str = f"""
            source {self.fairseq_python_env.get_path()}/bin/activate && \
            PYTHONPATH=$PYTHONPATH:{self.fairseq_root.get_path()} \
            python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/config/generate.py \
            {final_fairseq_hydra_config_str}
        """
    
        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)


