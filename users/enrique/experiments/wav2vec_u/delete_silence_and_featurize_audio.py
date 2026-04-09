###########################################################
# Minimal Audio Processing Script
###########################################################
import os

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import (
    process_audio,
)
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_rvad_root,
    get_fairseq_root,
)
from i6_core.tools.download import DownloadJob
from sisyphus import tk


def run_audio_processing():
    ###########################################################
    # General config
    ###########################################################
    audio_dir = "/u/corpora/speech/LibriSpeech/LibriSpeech/dev-clean"
    audio_alias = "processed_audio/dev-clean"
    
    audio_ext = "flac"
    
    w2v2model = "large_60kh"  # options: "base", "large_960h", "large_60kh"
    feature_extraction_layer = 14   # Layer to extract features
    concurrent_jobs = 8             # Number of parallel processes to run for featurization and silecence deletion, 8 or more is recommended for more full 960h LibriSpeech (otherwise could run OOM)


    ###########################################################
    # Environment and Fairseq root
    ###########################################################
    environment = "/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"
    fairseq_root = get_fairseq_root(
        python_env=tk.Path(environment),
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
    )

    # Wav2Vec2 model choice
    if w2v2model == "large_960h":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
            target_filename="wav2vec2_large_960h_no_finetune.pt",
        ).out_file
    elif w2v2model == "large_60kh":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt",
            target_filename="wav2vec_60kh_no_finetune.pt",
        ).out_file
    elif w2v2model == "base":
        w2v2_model_path = DownloadJob(
            "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
            target_filename="wav2vec_small.pt",
        ).out_file
    else:
        raise ValueError(f"Unknown model: {w2v2model}")

    ###########################################################
    # Run processing
    ###########################################################
    environment = tk.Path(environment)
    audio_dir = tk.Path(audio_dir)

    delete_silences_job, featurize_audio_job = process_audio(
        env=environment,
        fairseq_root=fairseq_root,
        audio_dir=audio_dir,
        valid_percent=None,
        ext=audio_ext,
        rvad_root=get_rvad_root(),
        concurrent=concurrent_jobs,
        layer=feature_extraction_layer,
        model_path=w2v2_model_path,
        alias_prefix=audio_alias,
        alias_delete=f"delete_silences/{w2v2model}/layer_{feature_extraction_layer}",
        alias_feat=f"featurize_audio/{w2v2model}/layer_{feature_extraction_layer}",
        max_n_audios_per_manifest=None,
        name_the_manifests_just_train_and_valid=False,
    )

    # Optional: Register outputs if needed
    tk.register_output(f"{audio_alias}/features", featurize_audio_job.out_features)


def py():
    run_audio_processing()
