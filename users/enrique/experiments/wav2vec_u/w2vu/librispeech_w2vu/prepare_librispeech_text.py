from recipe.i6_experiments.users.enrique.jobs.wav2vec_data_utils import PrepareWav2VecTextDataJob, get_fairseq_root
from recipe.i6_experiments.users.enrique.experiments.exp2025_03_19_fairseq.w2vu.librispeech_w2vu.default_tools import (
    KENLM_BINARY_PATH,
)
from sisyphus import tk


def main():

    fairseq_python_env = tk.Path("/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3")

    fairseq_root = get_fairseq_root(
        python_env=fairseq_python_env,
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
        identifier="kaldi",
    )

    language = "en"

    tts_engine = "G2P"

    fasttext_model = tk.Path(
        "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/i6_core/tools/download/DownloadJob.Zx9QK2LMKdWk/output/lid.176.bin"
    )

    text_file_path = tk.Path(
        "/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt"
    )

    # # Test file
    # text_file_path = tk.Path(
    #     "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/data/text_raw/BNCCorpus.txt"
    # )

    sil_prob = 0.25

    vocab_size = 1000

    job = PrepareWav2VecTextDataJob(
        fairseq_root=fairseq_root,
        language=language,
        text_file_path=text_file_path,
        kenlm_root=KENLM_BINARY_PATH,
        tts_engine=tts_engine,
        fasttext_model=fasttext_model,
        sil_prob=sil_prob,
        fairseq_python_env=fairseq_python_env,
        vocab_size=vocab_size,
    )

    tk.register_output("text", job.out_text_dir)


def py():
    main()
