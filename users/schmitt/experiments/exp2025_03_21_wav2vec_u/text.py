import os
import tempfile
import shutil
import subprocess as sp
from typing import Sequence
import requests
import re

from sisyphus import tk, Task, gs


class DownloadGutenbergBooksJob(tk.Job):
  def __init__(
          self,
          wav2letter_root: tk.Path,
          mosesdecoder_root: tk.Path,
          wav2letter_python_exe: tk.Path,
          concurrent: int = 200,
  ):
    """
    Download the Gutenberg books from the Gutenberg website and process them.
    Args:
      wav2letter_root:
      wav2letter_python_exe:
        a python environment with the wav2letter dependencies
        (https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019/raw_lm_corpus)
      concurrent:
    """

    self.wav2letter_root = wav2letter_root
    self.mosesdecoder_root = mosesdecoder_root
    self.concurrent = concurrent
    self.wav2letter_python_exe = wav2letter_python_exe

    self.out_dir = self.output_path("gutenberg_books", directory=True, cached=True)

  def tasks(self):
    yield Task(
      "download_harvest_files",
      resume="download_harvest_files",
      mini_task=True,
    )
    yield Task(
      "download_zip_files",
      resume="download_zip_files",
      rqmt={"time": 2, "cpu": 2, "gpu": 0, "mem": 8},
      args=range(1, self.concurrent + 1)
    )
    yield Task("cleanup", resume="cleanup", mini_task=True)
    yield Task("process_text", resume="process_text", mini_task=True)
    yield Task("preprocess_metadata", resume="preprocess_metadata", mini_task=True)

  def download_harvest_files(self):
    job_work_dir = os.getcwd()
    with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
      print("using temp-dir: %s" % tmp_dir)

      # download data from robot/harvest endpoint
      harvest_url = "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
      download_cmd = f"wget -m -H -nd --no-verbose '{harvest_url}'"
      sp.check_call(download_cmd, shell=True, cwd=tmp_dir)

      # move zip files to job work dir, so that they don't have to be downloaded again if job crashes
      # also write the names of the harvest files to a file
      with open(os.path.join(job_work_dir, "harvest_files.txt"), "w") as harvest_file:
        for file in os.listdir(tmp_dir):
          if file.startswith("harvest"):
            # move the zip files to the output directory
            shutil.move(os.path.join(tmp_dir, file), os.path.join(job_work_dir, file))
            harvest_file.write(file + "\n")

  def download_zip_files(self, task_id: int):
    job_work_dir = os.getcwd()
    harvest_files = [file for file in os.listdir(job_work_dir) if file.startswith("harvest")]
    harvest_files = sorted(harvest_files)
    harvest_files = harvest_files[task_id - 1::self.concurrent]

    print(f"Processing {len(harvest_files)} harvest files: \n", '\n'.join([str(file) for file in harvest_files]))

    with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
      print("using temp-dir: %s" % tmp_dir)

      for file in harvest_files:
        shutil.copy(os.path.join(job_work_dir, file), os.path.join(tmp_dir, file))

      grep_pattern = "http://aleph\\.gutenberg\\.org[^\\\"]+\\.zip"
      get_urls_cmd = f'grep -hEo "{grep_pattern}" harvest* | sort -u'
      url_proc = sp.Popen(get_urls_cmd, shell=True, cwd=tmp_dir, stdout=sp.PIPE)
      urls = url_proc.stdout.read().decode("utf-8").splitlines()

      for url in urls:
        # first, try original url
        download_cmd = f"wget --no-clobber --no-parent --no-verbose '{url}'"
        try:
          sp.check_call(download_cmd, shell=True, cwd=tmp_dir)
        except sp.CalledProcessError:
          pass
        try:
          # if the original url does not work, try the modified url
          modified_url = re.sub(r'(http://.*/)([\d\-]+\.zip)', r'\1old/\2', url)
          print(f"{url} does not work. Trying {modified_url}")
          download_cmd = f"wget --no-clobber --no-parent --no-verbose '{modified_url}'"
          sp.check_call(download_cmd, shell=True, cwd=tmp_dir)
        except sp.CalledProcessError:
          print(f"Failed to download {url} and {modified_url}. Skipping.")

      # move zip files to job work dir, so that they don't have to be downloaded again if job crashes
      for file in os.listdir(tmp_dir):
        if file.endswith(".zip"):
          # move the zip files to the output directory
          shutil.move(os.path.join(tmp_dir, file), os.path.join(job_work_dir, file))

  def cleanup(self):
    for file in os.listdir():
      # remove harvest files
      if file.startswith(("harvest", "robots.")):
        os.remove(file)
      # remove duplicates
      elif file.endswith(("-0.zip", "-8.zip")):
        os.remove(file)

      # remove zip files with non-standard names (i.e. digits and dashes)
      if file.endswith(".zip") and not re.match(r"^([\d\-]+)\.zip$", file):
        os.remove(file)

    unzip_cmd = "unzip '*.zip'"   # && rm *.zip"
    sp.check_call(unzip_cmd, shell=True)

    # remove non-txt data
    rm_non_txt = "ls | grep -v '\.txt' | xargs -r rm -rf"
    sp.check_call(rm_non_txt, shell=True)

  def process_text(self):
    rm_headers_cmd = (
      f"{self.wav2letter_python_exe} {self.wav2letter_root}/recipes/sota/2019/raw_lm_corpus/process_raw_text.py "
      f"--indir {os.getcwd()} "
    )
    sp.check_call(rm_headers_cmd, shell=True)

    body_texts_folder = os.path.join(os.getcwd(), "body_texts")
    os.makedirs(body_texts_folder, exist_ok=True)
    for file in os.listdir():
      if file.endswith(".body.txt"):
        shutil.move(file, os.path.join(body_texts_folder, file))

  def preprocess_metadata(self):
    body_texts_dir = os.path.join(os.getcwd(), "body_texts")
    title_id_map_path = os.path.join(os.getcwd(), "title_id_map.out")
    ids_path = os.path.join(os.getcwd(), "title_id_map.ids.out")
    titles_path = os.path.join(os.getcwd(), "title_id_map.titles.out")
    normalize_title_script_path = (
      f"{self.wav2letter_root}/recipes/sota/2019/lm_corpus_and_PL_generation/normalize_title.sh"
    )
    normalized_titles_path = os.path.join(os.getcwd(), "title_id_map.titles.out.norm")
    title_id_map_norm_sort_by_id_path = os.path.join(os.getcwd(), "title_id_map.sort_by_id.norm.table")
    title_id_map_norm_sort_by_title_path = os.path.join(os.getcwd(), "title_id_map.sort_by_title.norm.table")
    sorted_ids_path = os.path.join(os.getcwd(), "title_id_map.sort_by_id.ids.norm.table")
    gutenberg_title_splits_dir_path = os.path.join(os.getcwd(), "gutenberg_title_splits")

    with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
      print("using temp-dir: %s" % tmp_dir)

      manifest_path = os.path.join(tmp_dir, "manifest.out")
      title_id_map_temp_path = os.path.join(tmp_dir, "title_id_map.out")
      cache_path = os.path.join(tmp_dir, "cache")

      create_id_list_cmd = (
        f'find . -type f | sed "s/\.body.txt//g" | sed "s/\.\///g" > {manifest_path}'
      )
      sp.check_call(create_id_list_cmd, shell=True, cwd=body_texts_dir)

      download_metadata_cmd = (
        "wget http://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2 && tar -xjf rdf-files.tar.bz2"
      )
      sp.check_call(download_metadata_cmd, shell=True, cwd=tmp_dir)

      create_title_id_map_cmd = (
        f"{self.wav2letter_python_exe} {self.wav2letter_root}/recipes/sota/2019/raw_lm_corpus/get_titles.py "
        f"--infile {manifest_path} --outfile {title_id_map_temp_path} --cachepath {cache_path} "
      )
      sp.check_call(create_title_id_map_cmd, shell=True, cwd=tmp_dir)

      shutil.move(title_id_map_temp_path, title_id_map_path)

    separate_ids_and_titles_cmd = (
      f"awk -F '|' '{{print $1}}' {title_id_map_path} > {ids_path} && "
      f"awk -F '|' '{{print $2}}' {title_id_map_path} > {titles_path}"
    )
    sp.check_call(separate_ids_and_titles_cmd, shell=True, cwd=body_texts_dir)

    normalize_titles_cmd = (
      f"{normalize_title_script_path} {self.mosesdecoder_root.get_path()} {titles_path} {normalized_titles_path}"
    )
    sp.check_call(normalize_titles_cmd, shell=True, cwd=os.path.dirname(normalize_title_script_path))

    recombine_title_id_map_cmds = [
      # table sorted by id
      f"paste -d '|' {ids_path} {normalized_titles_path} | sort > {title_id_map_norm_sort_by_id_path}",
      # table sorted by title
      f"cat {title_id_map_norm_sort_by_id_path} | sort -k 2 -t '|' > {title_id_map_norm_sort_by_title_path}",
      # create sorted id list
      f"awk -F'|' '{{print $1}}' {title_id_map_norm_sort_by_id_path} > {sorted_ids_path}"
    ]
    for cmd in recombine_title_id_map_cmds:
      sp.check_call(cmd, shell=True, cwd=os.getcwd())

    os.makedirs(gutenberg_title_splits_dir_path, exist_ok=False)
    # heuristic given by git repo
    n_lines_per_file = "$(( ($(less ../title_id_map.titles.out.norm | wc -l) + $(nproc) - 1) / $(nproc) ))"
    create_split_title_list_cmd = (
      f"split --lines={n_lines_per_file} {normalized_titles_path}"
    )
    sp.check_call(create_split_title_list_cmd, shell=True, cwd=gutenberg_title_splits_dir_path)


  @classmethod
  def hash(cls, kwargs):
    if kwargs["wav2letter_python_exe"].get_path() == "/work/asr4/schmitt/venvs/wav2letter_lm_corpus/bin/python3":
      kwargs.pop("wav2letter_python_exe")

    # remove for now because hash would break otherwise
    kwargs.pop("mosesdecoder_root")

    return super().hash(kwargs)


# TODO: continue working on this and then on RemoveLibriLightFromLibrispeechLMCorpusJob
class GetLibriLightBookTitlesJob(tk.Job):
  def __init__(
          self,
          libri_light_data_root: tk.Path,
  ):
    """
    Download the Gutenberg books from the Gutenberg website and process them.
    Args:
      wav2letter_root:
      wav2letter_python_exe:
        a python environment with the wav2letter dependencies
        (https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019/raw_lm_corpus)
      concurrent:
    """

    self.libri_light_data_root = libri_light_data_root

    self.out_file = self.output_path("book_titles.txt")

  def tasks(self):
    yield Task("run", resume="run", mini_task=True)

  def run(self):
    small_folder = os.path.join(self.libri_light_data_root, "small")
    medium_folder = os.path.join(self.libri_light_data_root, "medium")
    large_folder = os.path.join(self.libri_light_data_root, "large")
    json_metadata_path = os.path.join(os.getcwd(), "json.metadata.manifest")

    find_json_files_cmd = (
      f"find {small_folder} {medium_folder} {large_folder} -name '*.json' > {json_metadata_path}"
    )
    sp.check_call(find_json_files_cmd, shell=True)


class RemoveLibriLightFromLibrispeechLMCorpusJob(tk.Job):
  def __init__(
          self,
          wav2letter_root: tk.Path,
          mosesdecoder_root: tk.Path,
          wav2letter_python_exe: tk.Path,
          librispeech_lm_corpus: tk.Path,
          concurrent: int = 200,
  ):
    """
    Download the Gutenberg books from the Gutenberg website and process them.
    Args:
      wav2letter_root:
      wav2letter_python_exe:
        a python environment with the wav2letter dependencies
        (https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019/raw_lm_corpus)
      concurrent:
    """

    self.wav2letter_root = wav2letter_root
    self.mosesdecoder_root = mosesdecoder_root
    self.concurrent = concurrent
    self.wav2letter_python_exe = wav2letter_python_exe
    self.librispeech_lm_corpus = librispeech_lm_corpus

    self.out_dir = self.output_path("filtered_corpus", directory=True, cached=True)

  def tasks(self):
    yield Task("find_overlapping_ids", resume="find_overlapping_ids", mini_task=True)

  def find_overlapping_ids(self):
    books_txt_path = os.path.join("librispeech-lm-corpus", "BOOKS.TXT")
    extract_books_txt_cmd = (
      f"tar -zxvf {self.librispeech_lm_corpus} {books_txt_path}"
    )
    # sp.check_call(extract_books_txt_cmd, shell=True)
    books_txt_path = os.path.join(os.getcwd(), books_txt_path)

    lbs_gutenberg_ids_path = os.path.join(os.getcwd(), "librispeech_lm_corpus.gutenberg.ids.lst")
    extract_gutenberg_ids_cmd = (
      f"awk -F'|' '{{print $1}}' {books_txt_path} | grep -Fv ';' | awk '{{$1=$1;print}}' | "
      f"sort | uniq > {lbs_gutenberg_ids_path}"
    )
    sp.check_call(extract_gutenberg_ids_cmd, shell=True)

