from i6_experiments.users.enrique.jobs.wav2vec_u_audio_preprocessing import (
    Wav2VecUDeleteSilencesInAudioJob,
    Wav2VecUFeaturizeAudioJob,
)
import logging
from i6_experiments.users.enrique.jobs.wav2vec_data_utils import get_rvad_root, get_fairseq_root
from sisyphus import tk

# fairseq_root = tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq")
audio_dir = tk.Path("/u/corpora/speech/LibriSpeech/LibriSpeech/train-other-960")

# Highly recommend using this environment
environment = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")

w2v2_model_path = tk.Path(
    "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/models/w2v2/wav2vec_vox_new.pt"
)


def prepare_audio():

    # Create the job instance with all configurations
    job = Wav2VecUDeleteSilencesInAudioJob(
        environment=environment,
        # fairseq_root=get_fairseq_root(
        #     fairseq_python_exe="/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3/bin/python"
        # ),
        audio_dir=audio_dir,
        valid_percent=0.01,
        extension="flac",
        rvad_root=get_rvad_root(),
        w2v2_model_path=w2v2_model_path,
        concurrent=16,
        # initial_manifests_dir=tk.Path("/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/LibriSpeech/w2vu_processed_audio/remove_audio_silence/with_silence/"),
    )

    # Register task output (if applicable)
    tk.register_output(f"test_output", job.out_manifest_and_vads)

    logging.info("LirbiSpeech w2vu data will be processed")


def featurize_audio():
    # Create the job instance with all configurations
    job = Wav2VecUFeaturizeAudioJob(
        environment=environment,
        fairseq_root=get_fairseq_root(
            fairseq_python_exe=tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3/bin/python"),
            fairseq_root=tk.Path(
                "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
            ),
        ),
        layer=14,
        w2v2_model_path=w2v2_model_path,
        input_audio_manifests=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_experiments/users/enrique/jobs/wav2vec_u_audio_preprocessing/Wav2VecUDeleteSilencesInAudioJob.XHzXkr2xOvHI/output/preprocessed_manifest"
        ),
        # input_audio_manifests=tk.Path(
        #     "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/very_small_audio_test/output/processed_data/with_silence"
        # ),
        concurrent=16,
    )

    # Register task output (if applicable)
    tk.register_output(f"test_output", job.out_features)

    logging.info("LirbiSpeech w2vu data will be processed")


def py():
    # prepare_audio()
    featurize_audio()
