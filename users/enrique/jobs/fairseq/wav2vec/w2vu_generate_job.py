import os
import subprocess as sp
from typing import Optional, Type
from sisyphus import Job, Task, tk
import logging


class FairseqGenerateWav2VecUJob(Job):
    """
    Run the w2vu_generate.py script with specified configurations for Wav2Vec generation.
    """

    def __init__(
        self,
        environment: Optional[tk.Path],
        fairseq_root: Type[tk.Path],
        task_data: Type[tk.Path],
        checkpoint_path: Type[tk.Path],
        config_dir: Optional[str] = None,
        config_name: Optional[str] = None,
        gen_subset: Optional[str] = "valid",
    ):
        """
        :param environment: Path to the Python virtual environment.
        :param fairseq_root: Path to the user directory for Fairseq.
        :param task_data: Path to the directory with features.
        :param checkpoint_path: Path to the GAN checkpoint.
        :param config_dir: Path to the configuration directory (default: "config/generate").
        :param config_name: Name of the Hydra configuration (default: "viterbi").
        :param gen_subset: Subset to generate (default: "valid").
        """
        self.environment = environment
        self.fairseq_root = fairseq_root
        self.task_data = task_data
        self.checkpoint_path = checkpoint_path
        self.config_dir = config_dir
        self.config_name = config_name
        self.gen_subset = gen_subset

        self.results_path = self.output_path("transcriptions", directory=True)

        # Resource requirements (adjust as needed)
        self.rqmt = {"time": 1000, "cpu": 1, "gpu": 1, "mem": 70}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        if not self.config_dir:
            logging.warning("No config_dir provided, using default.")
            self.config_dir = os.path.join(
                self.fairseq_root.get_path(), "examples/wav2vec/unsupervised/config/generate"
            )

        if not self.config_name:
            logging.warning("No config_name provided, using default.")
            self.config_name = "viterbi"

        sh_call_str = ""
        if self.environment is not None:
            sh_call_str += f"export PYTHONNOUSERSITE=1 && source {self.environment.get_path()}/bin/activate && "

        sh_call_str = (
            sh_call_str
            + f""" \
            export HYDRA_FULL_ERROR=1 && \
            /opt/conda/bin/python {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/w2vu_generate.py \
            --config-dir {self.config_dir} \
            --config-name {self.config_name} \
            fairseq.common_eval.path={self.checkpoint_path.get_path()} \
            fairseq.task.data={self.task_data.get_path()} \
            fairseq.dataset.gen_subset={self.gen_subset} \
            fairseq.common.user_dir={self.fairseq_root.get_path()}/examples/wav2vec/unsupervised \
            results_path={self.results_path.get_path()} \
        """
        )

        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)
