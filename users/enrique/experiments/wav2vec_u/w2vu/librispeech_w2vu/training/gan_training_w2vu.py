from recipe.i6_experiments.users.enrique.jobs.training.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob
from i6_experiments.users.enrique.jobs.wav2vec_data_utils import get_fairseq_root

from sisyphus import tk
import logging

import subprocess as sp


def calculate_all_configs(configs, seed_range: range):
    """
    Calculate the number of models to be trained based on the provided configurations.
    This is a mini-task that runs before the main training task.
    """
    n_models = 1
    for value in configs.values():
        if isinstance(value, list):
            n_models *= len(value)

        elif isinstance(value, range):
            n_models *= len(value)

        else:
            n_models *= 1

    n_models *= len(seed_range)

    logging.info(f"Number of models to train: {n_models}")

    all_configs = [{}]

    keys = list(configs.keys())

    for key in keys:
        all_configs_aux = []
        for con in all_configs:
            if isinstance(configs[key], list):
                for value in configs[key]:
                    new_config = con.copy()
                    new_config[key] = value
                    all_configs_aux.append(new_config)
            elif isinstance(configs[key], range):
                for value in configs[key]:
                    new_config = con.copy()
                    new_config[key] = value
                    all_configs_aux.append(new_config)
            else:
                new_config = con.copy()
                new_config[key] = configs[key]
                all_configs_aux.append(new_config)
        all_configs = all_configs_aux

    all_configs_aux = []
    for con in all_configs:
        for seed in seed_range:
            new_config = con.copy()
            new_config["common.seed"] = seed
            all_configs_aux.append(new_config)
    all_configs = all_configs_aux

    logging.info(f"All configurations: {all_configs}")
    assert len(all_configs) == n_models

    return all_configs, n_models


def gan(config):

    environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")
    # Example usage
    fairseq_root = get_fairseq_root(
        python_env=environment,
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
        identifier="wav2vec_u",
    )

    # # Test data
    # task_data = tk.Path(
    #     "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUFeaturizeAudioJob.ARwP2pLfOCxL/output/audio_features/precompute_pca512_cls128_mean_pooled"
    # )

    # w2vu
    task_data = tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUFeaturizeAudioJob.TyJVh2DlIY8F/output/audio_features/precompute_pca512_cls128_mean_pooled")

    # # w2vu2
    # task_data = tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUFeaturizeAudioJob.TyJVh2DlIY8F/output/audio_features")

    task_text_data = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_data_utils/PrepareWav2VecTextDataJob.CsHVpj6ippjz/output/text/phones"
    )

    task_kenlm_path = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_data_utils/PrepareWav2VecTextDataJob.CsHVpj6ippjz/output/text/phones/lm.phones.filtered.04.bin"
    )

    # Training configuration parameters
    prefix = "w2v_unsup_gan_xp"
    config_dir = fairseq_root.get_path() + "/examples/wav2vec/unsupervised/config/gan"
    config_name = config

    # Configurations compatible with Fairseq-Hydra
    configs = {
        "model.code_penalty": [2, 4],
        "model.gradient_penalty": [1.5, 2.0],
        "model.smoothness_weight": [0.5, 0.75],
    }
    seed_range = range(0, 5)

    all_configs, n_models = calculate_all_configs(configs, seed_range)

    for conf in all_configs:
        job = FairseqHydraTrainWav2VecUJob(
            environment=environment,
            task_data=task_data,
            task_text_data=task_text_data,
            task_kenlm_path=task_kenlm_path,
            fairseq_root=fairseq_root,
            prefix=prefix,
            config_dir=config_dir,
            config_name=config_name,
            configs=conf,
        )

        # Register task output (if applicable)
        tk.register_output(f"{prefix}", job.out_dir)


def py():
    gan("w2vu")
    # gan("w2vu2")
