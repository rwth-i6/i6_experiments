from i6_experiments.users.enrique.experiments.wav2vec_u.w2vu.librispeech_w2vu.decoding.w2vu_generate import (
    generate_with_kaldi,
)
from sisyphus import tk


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
            "/work/smt4/zeineldeen/enrique.leon.lozano/setups-data/ubuntu_22_setups/fairseq_2025_06_02/work/i6_experiments/users/enrique/jobs/fairseq/wav2vec/audio_preprocessing/Wav2VecUFeaturizeAudioJob.I1ZVH3T2YXKP/output/audio_features/precompute_pca512_cls128_mean"
        ),
        config_name="kike_kaldi_pruned_2",
        # config_name="viterbi",
        gen_subset="train",
    )
