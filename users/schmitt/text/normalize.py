from sisyphus import tk, Task

import subprocess as sp
import shutil


class NormalizeLBSLMDataJob(tk.Job):
  def __init__(
          self,
          wav2letter_root: tk.Path,
          wav2letter_python_exe: tk.Path,
          librispeech_lm_corpus: tk.Path,
  ):
    self.wav2letter_root = wav2letter_root
    self.wav2letter_python_exe = wav2letter_python_exe
    self.librispeech_lm_corpus = librispeech_lm_corpus

    self.out_corpus_norm = self.output_path("corpus.norm.txt", cached=True)

  def tasks(self):
    yield Task("run", resume="run", rqmt={"time": 30, "cpu": 8, "gpu": 0, "mem": 8})

  def run(self):
    normalize_cmd = (
      f"which python3"
    )
    sp.check_call(normalize_cmd, shell=True)

    clone_moses_cmd = (
      "git clone --recursive https://github.com/moses-smt/mosesdecoder.git && cd mosesdecoder && git checkout fd06cdf026dd9e0396db56a7d93c2f6b446a1e02 && cd .."
    )
    sp.check_call(clone_moses_cmd, shell=True)

    copy_scripts_cmd = (
      f"cp {self.wav2letter_root}/recipes/sota/2019/lm_corpus_and_PL_generation/* . && chmod u+x *.sh"
    )
    sp.check_call(copy_scripts_cmd, shell=True)

    normalize_cmd = (
      f"which python3"
    )
    sp.check_call(normalize_cmd, shell=True)

    normalize_cmd = (
      f"./normalize.sh {self.librispeech_lm_corpus.get_path()}"
    )
    sp.check_call(normalize_cmd, shell=True)

    shutil.move("librispeech_lm_corpus_raw_without_librivox.txt.norm", self.out_corpus_norm.get_path())
