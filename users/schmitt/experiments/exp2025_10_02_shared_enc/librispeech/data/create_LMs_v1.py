###########################################################
# Imports
###########################################################
import os


from i6_core.tools.download import DownloadJob

from i6_experiments.users.enrique.jobs.fairseq.wav2vec.audio_preprocessing import (
    Wav2VecUDeleteSilencesInAudioJob,
    Wav2VecUFeaturizeAudioJob,
    process_audio,
)
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_u_GAN import FairseqHydraTrainWav2VecUJob
from i6_experiments.users.enrique.experiments.wav2vec_u.default_tools import KENLM_BINARY_PATH

from recipe.i6_experiments.users.enrique.jobs.fairseq.wav2vec.w2vu_generate_job import FairseqGenerateWav2VecUJob
import logging
from i6_experiments.users.enrique.jobs.fairseq.wav2vec.wav2vec_data_utils import (
    get_rvad_root,
    get_fairseq_root,
    PrepareWav2VecTextDataJob,
    calculate_all_configs,
)
from i6_core.returnn.search import ReturnnComputeWERJob

from sisyphus import tk


def run_meta_experiments():
    environment = "/work/smt4/zeineldeen/enrique.leon.lozano/py_envs/fairseq_env_v3"
    fairseq_root = get_fairseq_root(
        python_env=tk.Path(environment),
        fairseq_root=tk.Path(
            "/u/enrique.leon.lozano/setups/ubuntu_22_setups/fairseq_2025_03_11/work/Fairseq/fairseq_w2vu/fairseq"
        ),
    )

    # Text configuration
    language = "en"  # Language of the text data
    tts_engine = "G2P"  # Text-to-speech engine to use for text normalization
    text_file_paths = [ # ("text_path", "alias")
        #("/work/asr4/schmitt/sisyphus_work_dirs/2025_03_21_wav2vec-u/i6_experiments/users/schmitt/experiments/exp2025_03_21_wav2vec_u/text/NormalizeLBSLMDataJob.7MP3Io7yzilY/output/corpus.norm.txt", "librispeech_lm_corpus_minus_librivox"),
        ("/u/schmitt/experiments/2025_10_02_unsup_asr_shared_enc/work/i6_core/text/processing/HeadJob.9TUStSHn1pXM/output/out","schmitt")
        ]
    text_file_paths = [(tk.Path(p), alias) for p, alias in text_file_paths]

    sil_prob = 0.25
    fasttext_model = DownloadJob(
        url="https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin", target_filename="lid.176.bin"
    ).out_file

    #lm_prunings = [[0,0,1,4], [0,0,2,5], [0,0,3,6]] # affects only the word-lm
    lm_prunings = [[0,0,1,4]] # affects only the word-lm
    #lm_prunings = [[0,0,0,0]] # affects only the word-lm

    ################################################################
    ########### text data and LM ############
    ################################################################
    environment = tk.Path(environment)

    prepare_text_job_decoding = []
    for text_file_path, alias in text_file_paths:
        alias = f"wav2vec_u_text_data_{alias}"
        for prun in lm_prunings:
            job = PrepareWav2VecTextDataJob(
                fairseq_root=fairseq_root,
                language=language,
                text_file_path=text_file_path,
                kenlm_root=KENLM_BINARY_PATH,
                tts_engine=tts_engine,
                fasttext_model=fasttext_model,
                sil_prob=sil_prob,
                fairseq_python_env=environment,
                lm_pruning=prun,
                #three_gram_lm=True, # Only temporal, usually use 4gram
            )
            tk.register_output(os.path.join(alias, f"3gram_LM"), job.out_text_dir)
            job.add_alias(os.path.join(alias, f"text_data_pruning_{'_'.join(map(str, prun))}"))
            prepare_text_job_decoding.append(job)
            

def py():
    run_meta_experiments()
