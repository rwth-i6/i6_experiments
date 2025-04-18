import os

from sisyphus import tk

from i6_core.tools.git import CloneGitRepositoryJob
from i6_core.tools.download import DownloadJob

from i6_experiments.users.schmitt.audio.preprocessing import RemoveSilenceFromAudioJob
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

from .text import DownloadGutenbergBooksJob, RemoveLibriLightFromLibrispeechLMCorpusJob, GetLibriLightBookTitlesJob


def run():
  prepare_libri_light_data()
  get_filtered_librispeech_lm_data()


def prepare_libri_light_data():
  # TODO: add explicit hashes for reproducibility
  fairseq_repo = CloneGitRepositoryJob(
    "https://github.com/facebookresearch/fairseq",
    checkout_folder_name="fairseq",
  )
  rvad_repo = CloneGitRepositoryJob(
    "https://github.com/zhenghuatan/rVADfast",
    commit="0ed4c12",
    checkout_folder_name="rvad",
  )

  # subfolders of '/u/corpora/speech/LibriLight/large' with very large amounts of audio files
  large_subfolders_num_elements = [
    ("3157", 44740),
    ("3645", 50785),
    ("4078", 49948),
    ("2607", 21108),
    ("5985", 18655),
    ("10289", 15204),
    ("10244", 20726),
    ("1401", 27932),
    ("3681", 22821),
    ("8098", 10142),
    ("6454", 76245),
    ("8713", 48285),
    ("7756", 56224),
    ("10801", 57857),
    ("2156", 42238)
  ]

  # the tasks for these jobs were interrupted (because of the large subfolders above)
  # therefore, we run a separate job for these subfolders
  separate_subfolders = [
    "7286",
    "13146",
    "11969",
    "12775",
    "4680",
    "5403",
    "9246",
    "17",
    "7079",
    "599",
    "1893",
    "11693",
    "10150",
    "1653",
    "8465",
    "9562",
    "11987",
    "7898",
    "2368",
    "13241",
    "5489",
    "11111",
    "8690",
    "10722",
    "9139",
    "13044",
    "4511",
    "460",
    "6140",
    "7030",
    "1574",
    "13117",
    "12026",
    "6112",
    "4859",
    "11995",
    "6221",
    "12244",
    "1877",
    "11953",
    "4156",
    "3557",
    "10345",
    "10420",
    "6760",
    "1847",
    "3927",
    "10360",
    "7891",
    "9096",
    "5294",
    "4036",
    "9503",
    "3887",
    "11044",
    "12882",
    "8659",
    "1195",
    "12685",
    "6039",
    "10411",
    "204",
    "11283",
    "9812",
    "398",
    "12704",
    "3160",
    "11133",
    "3914",
    "56",
    "10964",
    "3742",
    "7448",
    "5555",
    "10466",
    "9063",
    "4134",
    "2170",
    "12327",
    "7807",
    "1759",
    "6424",
    "1377",
    "12192",
    "4818",
    "3888",
    "5944",
    "3487",
    "5372",
    "3790",
    "6479",
    "2430",
    "4948",
    "11998",
    "12431",
    "4411",
    "1640",
    "2951",
    "11364",
    "8678",
    "12269",
    "3517",
    "7489",
    "9836",
    "9405",
    "9147",
    "11532",
    "4183",
    "10731",
    "6575",
    "11001",
    "9076",
    "1162",
    "4725",
    "6538",
    "453",
    "9429",
    "11274",
    "6497",
    "2447",
    "5293",
    "12927",
    "5126",
    "3708",
    "10651",
    "12202",
    "9197",
    "3693",
    "10557",
    "10079",
    "8873",
    "12384",
    "322",
    "5321",
    "6219",
    "4084",
    "7938",
    "6925",
    "888",
    "6142",
    "9863",
    "5102",
    "9462",
    "3150",
    "12397",
    "2159",
    "6546",
    "5616",
    "8857",
    "11604",
    "9679",
    "10068",
    "2623",
    "7553"
  ]

  libri_light_large_path = "/u/corpora/speech/LibriLight/large"
  libri_light_large_subfolders_part1 = [
    subfolder for subfolder in os.listdir(libri_light_large_path) if subfolder not in separate_subfolders
  ]
  libri_light_large_subfolders_part2 = [
    subfolder for subfolder in separate_subfolders if subfolder not in map(lambda x: x[0], large_subfolders_num_elements)
  ]

  for alias, subfolders in [
    ("part1", libri_light_large_subfolders_part1),
    ("part2", libri_light_large_subfolders_part2)
  ]:
    libri_light_large_subfolders = [
      tk.Path(os.path.join(libri_light_large_path, subfolder)) for subfolder in subfolders
    ]
    # split the list of subfolders into n lists
    if alias == "part1":
      n = 500
    else:
      n = 10

    libri_light_large_subfolders_paths_split = [
      libri_light_large_subfolders[i::n] for i in range(n)
    ]
    remove_sil_job = RemoveSilenceFromAudioJob(
      fairseq_root=fairseq_repo.out_repository,
      rvad_root=rvad_repo.out_repository,
      concurrent_audio_dirs=libri_light_large_subfolders_paths_split,
      topmost_folder_name="LibriLight",
      audio_ext="flac",
      time=72,
    )
    remove_sil_job.add_alias(f"data/LibriLight/wo_silence/{alias}")
    tk.register_output(remove_sil_job.get_one_alias(), remove_sil_job.out_dir)

  for large_subfolder, num_elements in large_subfolders_num_elements:
    large_subfolder_path = os.path.join(libri_light_large_path, large_subfolder)
    subsubfolder_list = [
      tk.Path(os.path.join(large_subfolder_path, subsubfolder)) for subsubfolder in os.listdir(large_subfolder_path)
    ]
    # split the list of subfolders into n lists
    if num_elements > 50_000:
      n = 30
    elif num_elements > 20_000:
      n = 20
    else:
      n = 10
    subsubfolder_list_split = [
      subsubfolder_list[i::n] for i in range(n)
    ]
    remove_sil_job = RemoveSilenceFromAudioJob(
      fairseq_root=fairseq_repo.out_repository,
      rvad_root=rvad_repo.out_repository,
      concurrent_audio_dirs=subsubfolder_list_split,
      topmost_folder_name="LibriLight",
      audio_ext="flac",
      time=72,
    )
    remove_sil_job.add_alias(f"data/LibriLight/wo_silence/{large_subfolder}")
    tk.register_output(remove_sil_job.get_one_alias(), remove_sil_job.out_dir)

    return


def get_filtered_librispeech_lm_data():
  librispeech_lm_corpus_job = DownloadJob(
    url="https://openslr.elda.org/resources/11/librispeech-lm-corpus.tgz"
  )
  tk.register_output("data/LibriSpeech/lm_corpus", librispeech_lm_corpus_job.out_file)

  librispeech_lm_norm_data = get_librispeech_normalized_lm_data()
  tk.register_output("data/LibriSpeech/lm_norm_data", librispeech_lm_norm_data)

  wav2letter_repo = CloneGitRepositoryJob(
    "https://github.com/flashlight/wav2letter",
    commit="e5a4b62d87f15fde6a963d9ac174c8db8eb67fbc",
    checkout_folder_name="wav2letter",
  )

  mosesdecoder_repo = CloneGitRepositoryJob(
    "https://github.com/moses-smt/mosesdecoder",
    commit="fd06cdf026dd9e0396db56a7d93c2f6b446a1e02",
    checkout_folder_name="mosesdecoder",
    clone_submodules=True
  )

  gutenberg_books_job = DownloadGutenbergBooksJob(
    wav2letter_root=wav2letter_repo.out_repository,
    concurrent=200,
    wav2letter_python_exe=tk.Path("/work/asr4/schmitt/venvs/wav2letter_lm_corpus/bin/python3"),
    mosesdecoder_root=mosesdecoder_repo.out_repository,
  )
  tk.register_output("data/gutenberg_books", gutenberg_books_job.out_dir)

  get_libri_light_book_titles_job = GetLibriLightBookTitlesJob(
    libri_light_data_root=tk.Path("/u/corpora/speech/LibriLight"),
  )
  tk.register_output("data/LibriLight/book_titles", get_libri_light_book_titles_job.out_file)

  filter_librispeech_job = RemoveLibriLightFromLibrispeechLMCorpusJob(
    wav2letter_root=wav2letter_repo.out_repository,
    concurrent=200,
    wav2letter_python_exe=tk.Path("/work/asr4/schmitt/venvs/wav2letter_lm_corpus/bin/python3"),
    mosesdecoder_root=mosesdecoder_repo.out_repository,
    librispeech_lm_corpus=librispeech_lm_corpus_job.out_file,
  )
  tk.register_output("data/LibriSpeech/lm_corpus_filtered", filter_librispeech_job.out_dir)
