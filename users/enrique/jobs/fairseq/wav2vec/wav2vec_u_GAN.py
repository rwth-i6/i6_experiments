import os
import subprocess as sp
from typing import Optional, Type, List, Dict, Any
from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import get_fairseq_root, cache_path
from sisyphus import Job, Task, tk
import logging


class FairseqHydraTrainWav2VecUJob(Job):
    """
    Run the fairseq-hydra-train script with dynamically specified configurations for Wav2Vec training.
    """

    def __init__(
        self,
        environment: Optional[tk.Path],
        task_data: Type[tk.Path],
        task_text_data: Type[tk.Path],
        task_kenlm_path: Type[tk.Path],
        fairseq_root: Type[tk.Path],
        prefix: str,
        config_dir: str,
        config_name: str,
        configs: Dict[str, Any],
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
        :param configs: Dictionary of additional model and task configurations.
        :param seed_range: Range of seed values for experimentation (e.g., range(0, 5)).
        """
        self.environment = environment
        self.task_data = task_data
        self.task_text_data = task_text_data
        self.task_kenlm_path = task_kenlm_path
        self.fairseq_root = fairseq_root
        self.prefix = prefix
        self.config_dir = config_dir
        self.config_name = config_name
        self.configs = configs

        self.out_dir = self.output_path("out_dir", directory=True)

        # Resource requirements
        self.rqmt = {"time": 1000, "cpu": 1, "gpu": 1, "mem": 32}

    def tasks(self):
        yield Task("train", rqmt=self.rqmt)

    def train(self):

        # Cache the paths for the task data, text data, and KenLM binary
        task_data = cache_path(self.task_data.get_path())
        task_text_data = cache_path(self.task_text_data.get_path())
        task_kenlm_path = cache_path(self.task_kenlm_path.get_path())

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
            {' '.join(hydra_args)} 
        """
        logging.info(f"Running command: {sh_call_str}")
        sp.run(["bash", "-c", sh_call_str], check=True)


def gan():
    # Example usage
    # fairseq_root = get_fairseq_root(fairseq_python_exe="/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3/bin/python", commit="a0ceabc287e26f64517fadb13a54c83b71e8e469")
    fairseq_root = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
    )

    task_data = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/audio_test/processed_audio/prepare_audio_features/precompute_pca512_cls128_mean_pooled"
    )
    task_text_data = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_data_utils/PrepareWav2VecTextDataJob.bGFecSJcedj2/output/audio/phones"
    )
    task_kenlm_path = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_data_utils/PrepareWav2VecTextDataJob.bGFecSJcedj2/output/audio/phones/lm.phones.filtered.04.bin"
    )
    environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")

    # Training configuration parameters
    prefix = "w2v_unsup_gan_xp"
    config_dir = fairseq_root.get_path() + "/examples/wav2vec/unsupervised/config/gan"
    config_name = "w2vu"

    # Configurations compatible with Fairseq-Hydra
    configs = {
        "model.code_penalty": [2, 4],
        "model.gradient_penalty": [1.5, 2.0],
        "model.smoothness_weight": [0.5, 0.75, 1.0],
    }
    seed_range = range(0, 5)  # Example seed range

    # Create the job instance with all configurations
    job = FairseqHydraTrainWav2VecUJob(
        environment,
        task_data,
        task_text_data,
        task_kenlm_path,
        fairseq_root,
        prefix,
        config_dir,
        config_name,
        configs,
        seed_range,
    )

    # Register task output (if applicable)
    tk.register_output(f"{prefix}", job.out_text_dir)

    # Start the process (if necessary)
    logging.info("Job is set up and ready to run.")


# Entry point of the script
def py():
    gan()
