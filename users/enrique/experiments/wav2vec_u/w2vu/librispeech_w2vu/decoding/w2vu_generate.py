from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.w2vu_generate_job import FairseqGenerateWav2VecUJob
from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import get_fairseq_root

from sisyphus import tk
import logging

import os

from typing import Optional, Type


def generate_with_kaldi(
    environment: Optional[tk.Path],
    fairseq_root: tk.Path,
    checkpoint_path: tk.Path,
    features_to_decode: tk.Path,
    config_name: str,
    gen_subset: Optional[str],
):

    # fairseq_root = get_fairseq_root(
    #     python_env=environment,
    #     fairseq_root=fairseq_root,
    #     identifier="kaldi",
    # )

    # Training configuration parameters
    prefix = "w2vu_generate"

    checkpoints = []
    if os.path.isdir(checkpoint_path.get_path()):
        for dirpath, dirnames, filenames in os.walk(checkpoint_path.get_path()):
            for filename in filenames:
                if filename == "checkpoint_best.pt":
                    full_checkpoint_path = os.path.join(dirpath, filename)

                    checkpoints.append(tk.Path(full_checkpoint_path))

    elif os.path.isfile(checkpoint_path.get_path()):
        checkpoints.append(checkpoint_path)

    checkpoints = checkpoints[:2]
    for each_checkpoint in checkpoints:
        job = FairseqGenerateWav2VecUJob(
            environment=environment,
            fairseq_root=fairseq_root,
            task_data=features_to_decode,
            checkpoint_path=each_checkpoint,
            config_name=config_name,
            gen_subset=gen_subset,
        )
        tk.register_output(f"{prefix}", job.results_path)

    logging.info(f"Decodingwith Kaldi, {len(checkpoints)} checkpoints found")


def py():
    generate_with_kaldi(
        environment=tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"),
        fairseq_root=tk.Path(
            "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/EXTERNAL_SOFTWARE/fairseq_w2vu/fairseq"
        ),
        # Checkpoint can either be a path to a specific checkpoint or a directory containing checkpoints
        checkpoint_path=tk.Path(
            # "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_GAN/FairseqHydraTrainWav2VecUJob.Pr0GfeauHSJo/work/multirun/2025-05-09/13-53-01/0/checkpoint_best.pt"
            "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/training/wav2vec_u_GAN"
        ),
        features_to_decode=tk.Path(
            "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUFeaturizeAudioJob.TyJVh2DlIY8F/output/audio_features/precompute_pca512_cls128_mean"
        ),
        config_name="kike_kaldi_pruned_2",
        # config_name="viterbi"
        gen_subset="valid",
    )
