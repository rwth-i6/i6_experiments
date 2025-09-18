import os
import subprocess as sp
from typing import Optional, Type, List, Dict, Any
from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import get_fairseq_root, cache_path
from sisyphus import Job, Task, tk
import logging
import shutil


class FairseqHydraTrainWav2VecUJob(Job):
    """
    Run the fairseq-hydra-train script with dynamically specified configurations for Wav2Vec training.
    """

    def __init__(
        self,
        environment: Optional[tk.Path],
        task_data: Type[tk.Path],
        task_text_data: Type[tk.Path],
        fairseq_root: Type[tk.Path],
        prefix: str,
        config_dir: str,
        config_name: str,
        extra_configs: Dict[str, Any],
        task_kenlm_path: Type[tk.Path] = None,
    ):
        """
        :param environment: Path to the Python virtual environment.
        :param task_data: Path to the task data directory.
        :param task_text_data: Path to the text data directory.
        :param task_kenlm_path: Path to the KenLM binary.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param prefix: Prefix for the training run.
        :param config_dir: Path to the configuration directory.
        :param config_name: Name of the Hydra configuration yaml file.
        :param extra_configs: Dictionary of additional model and task configurations.
        :param seed_range: Range of seed values for experimentation (e.g., range(0, 5)).
        """
        self.environment = environment
        self.task_data = task_data
        self.task_text_data = task_text_data
        # TODO : search for the specific lm inside the task_text_data folder
        self.task_kenlm_path = (
            os.path.join(task_text_data.get_path(), "lm.phones.filtered.04.bin")
            if task_kenlm_path is None
            else task_kenlm_path
        )
        self.fairseq_root = fairseq_root
        self.prefix = prefix
        self.config_dir = config_dir
        self.config_name = config_name
        self.configs = extra_configs

        self.out_dir = self.output_path("out_dir", directory=True)

        # Resource requirements
        self.rqmt = {"time": 1000, "cpu": 1, "gpu": 1, "mem": 32}

    def tasks(self):
        yield Task("copy_dict_phn_txt", mini_task=True)
        yield Task("train", rqmt=self.rqmt)

    def copy_dict_phn_txt(self):
        dict_phn_path = os.path.join(self.task_text_data.get_path(), "dict.phn.txt")
        dest = os.path.join(self.task_data.get_path(), "dict.phn.txt")
        logging.info(f"coppied {dict_phn_path} to {dest}")
        shutil.copy2(dict_phn_path, dest)

    def train(self):

        # Cache the paths for the task data, text data, and KenLM binary
        task_data = cache_path(self.task_data.get_path())
        task_text_data = cache_path(self.task_text_data.get_path())
        task_kenlm_path = (
            cache_path(self.task_kenlm_path.get_path())
            if type(self.task_kenlm_path) is tk.Path
            else self.task_kenlm_path
        )

        # Construct the list of Hydra arguments dynamically based on the passed config
        hydra_args = []

        for key, value in self.configs.items():
            if isinstance(value, list):
                # Format lists properly for Hydra (as comma-separated values)
                hydra_args.append(f"{key}={','.join(map(str, value))}")
            elif isinstance(value, range):
                # Format ranges properly for Hydra
                hydra_args.append(f"{key}=range({value.start},{value.stop})")
            else:
                # Directly append other value types as-is
                hydra_args.append(f"{key}={value}")

        sh_call_str = f"""
            source {self.environment.get_path()}/bin/activate && \
            export HYDRA_FULL_ERROR=1 && \
            PYTHONPATH=$PYTHONPATH:{self.fairseq_root.get_path()} \
            PREFIX={self.prefix} \
            fairseq-hydra-train -m \
            --config-dir {self.config_dir} \
            --config-name {self.config_name} \
            task.data={task_data} \
            task.text_data={task_text_data} \
            task.kenlm_path={task_kenlm_path} \
            common.user_dir={self.fairseq_root.get_path()}/examples/wav2vec/unsupervised \
            checkpoint.save_dir={self.out_dir.get_path()} \
            {' '.join(hydra_args)} 
        """
        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)