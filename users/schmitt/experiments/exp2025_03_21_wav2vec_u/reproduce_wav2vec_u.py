import os

from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.download import DownloadJob

from i6_experiments.users.schmitt.audio.preprocessing import RemoveSilenceFromAudioJob
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data


def run():
  fairseq_repo = CloneGitRepositoryJob(
    "https://github.com/facebookresearch/fairseq",
    checkout_folder_name="fairseq",
  )
  rvad_repo = CloneGitRepositoryJob(
    "https://github.com/zhenghuatan/rVADfast",
    commit="0ed4c12",
    checkout_folder_name="rvad",
  )

  libri_light_large_path = "/u/corpora/speech/LibriLight/large"
  libri_light_large_subfolders = os.listdir(libri_light_large_path)
  libri_light_large_subfolders_paths = [
    tk.Path(os.path.join(libri_light_large_path, subfolder)) for subfolder in libri_light_large_subfolders
  ]
  # split the list of subfolders into 4 lists
  n = 500
  libri_light_large_subfolders_paths_split = [
    libri_light_large_subfolders_paths[i::n] for i in range(n)
  ]

  remove_sil_job = RemoveSilenceFromAudioJob(
    fairseq_root=fairseq_repo.out_repository,
    rvad_root=rvad_repo.out_repository,
    concurrent_audio_dirs=libri_light_large_subfolders_paths_split,
    topmost_folder_name="LibriLight",
    audio_ext="flac",
    time=72,
  )
  tk.register_output("data/LibriLight/wo_silence", remove_sil_job.out_dir)

  librispeech_lm_corpus_job = DownloadJob(
    url="https://openslr.elda.org/resources/11/librispeech-lm-corpus.tgz"
  )
  tk.register_output("data/LibriSpeech/lm_corpus", librispeech_lm_corpus_job.out_file)

  librispeech_lm_norm_data = get_librispeech_normalized_lm_data()
  tk.register_output("data/LibriSpeech/lm_norm_data", librispeech_lm_norm_data)
