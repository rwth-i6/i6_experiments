from copy import deepcopy
import os
import subprocess as sp
from typing import Optional, Type, Dict, Any
from sisyphus import Job, Task, tk
import logging
from sisyphus import tools
from recipe.i6_experiments.users.enrique.jobs.gan.utils import cache_path, resolve_tk_paths, flatten_dictionary_to_text
import yaml

class FairseqHydraTrainJob(Job):
    __sis_hash_exclude__ = {"fairseq_python_env", "fairseq_root"}
    def __init__(
        self,
        fairseq_python_env: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        fairseq_hydra_config: dict,
    ):
        self.fairseq_python_env = fairseq_python_env
        self.fairseq_root = fairseq_root
        self.fairseq_hydra_config = fairseq_hydra_config

        self.config_folder = self.output_path("config", directory=True)
        self.config_file = self.output_path("config/hydra_train.yaml")
        self.out_dir = self.output_path("out_dir", directory=True)
        self.out_best_model = self.output_path("out_dir/checkpoint_best.pt")

        # Resource requirements
        self.rqmt = {"time": 3000, "gpu": 1, "mem": 25}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):

        final_fairseq_hydra_config = deepcopy(self.fairseq_hydra_config)

        # Cache the paths for the task data, text data, and KenLM binary
        final_fairseq_hydra_config['task']['data'] = cache_path(self.fairseq_hydra_config['task']['data'].get_path())
        final_fairseq_hydra_config['task']['text_data'] = cache_path(self.fairseq_hydra_config['task']['text_data'].get_path())
        final_fairseq_hydra_config['task']['kenlm_path'] = cache_path(self.fairseq_hydra_config['task']['kenlm_path'].get_path())
        final_fairseq_hydra_config['checkpoint']['save_dir'] = self.out_dir.get_path()
        final_fairseq_hydra_config['common']['user_dir'] = os.path.join(self.fairseq_root.get_path(), "examples/wav2vec/unsupervised")

        final_fairseq_hydra_config = resolve_tk_paths(final_fairseq_hydra_config)
        
        with open(self.config_file.get_path(), "w") as f:
            yaml.dump(final_fairseq_hydra_config, f)
        
        

        sh_call_str = f"""
            source {self.fairseq_python_env.get_path()}/bin/activate && \
            export HYDRA_FULL_ERROR=1 && \
            PYTHONPATH=$PYTHONPATH:{self.fairseq_root.get_path()} \
            fairseq-hydra-train \
            --config-dir {self.config_folder.get_path()} \
            --config-name hydra_train \
        """
        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)

    @classmethod
    def hash(cls, parsed_args: Dict[str, Any]) -> str:

        d = {}
        for k, v in parsed_args.items():
            if k not in cls.__sis_hash_exclude__:
                d[k] = v

        if cls.__sis_version__ is None:
            return tools.sis_hash(d)
        else:
            return tools.sis_hash((d, cls.__sis_version__)) 