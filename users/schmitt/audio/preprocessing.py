import os
import tempfile
import shutil
import subprocess as sp
from typing import Sequence

from sisyphus import tk, Task, gs


class RemoveSilenceFromAudioJob(tk.Job):
  def __init__(
          self,
          fairseq_root: tk.Path,
          rvad_root: tk.Path,
          concurrent_audio_dirs: Sequence[Sequence[tk.Path]],
          topmost_folder_name: str,
          audio_ext: str,
          time: float,
  ):
    """

    """
    assert audio_ext in ("wav", "flac")

    self.fairseq_root = fairseq_root
    self.rvad_root = rvad_root
    self.n_concurrent = len(concurrent_audio_dirs)
    self.concurrent_audio_dirs = concurrent_audio_dirs
    self.audio_ext = audio_ext
    self.topmost_folder_name = topmost_folder_name

    self.out_dir = self.output_path("audio_wo_silence", directory=True, cached=True)
    self.rqmt = {
      "time": time,
      "cpu": 4,
      "gpu": 0,
      "mem": 8,
    }

  def tasks(self):
    yield Task("run", resume="run", rqmt=self.rqmt, args=range(1, self.n_concurrent + 1))

  def run(self, task_id):
    with tempfile.TemporaryDirectory(prefix=gs.TMP_PREFIX) as tmp_dir:
      print("using temp-dir: %s" % tmp_dir)

      audio_dirs = self.concurrent_audio_dirs[task_id - 1]
      print("Processing {len(audio_dirs)} audio directories: \n", '\n'.join([str(audio_dir) for audio_dir in audio_dirs]))
      for audio_dir_ in audio_dirs:
        audio_dir = audio_dir_.get_path()

        # find the topmost folder position in the current audio_dir
        base_path_split = audio_dir.split("/")
        idx = 0
        for i, path in enumerate(base_path_split):
          if path == self.topmost_folder_name:
            idx = i
            break
        # create the output directory (self.out_dir.get_path()/topmost_folder_name/...)
        output_dir_ = "/".join(base_path_split[idx:])
        output_dir_ = os.path.join(self.out_dir.get_path(), output_dir_)
        if os.path.exists(output_dir_):
          print(f"Output directory {output_dir_} already exists. Removing it.")
          shutil.rmtree(output_dir_)
        os.makedirs(output_dir_, exist_ok=False)

        # these two files are created by the fairseq scripts
        audio_path_file = os.path.join(tmp_dir, "train.tsv")  # manifest file
        vads_file = os.path.join(tmp_dir, "train.vads")  # silence parts file

        # create wav2vec manifest file
        extract_audio_paths_cmd = [
          "python3",
          f"{self.fairseq_root}/examples/wav2vec/wav2vec_manifest.py",
          audio_dir,
          "--ext",
          self.audio_ext,
          "--dest",
          ".",
          "--valid-percent",
          "0",
        ]
        sp.check_call(extract_audio_paths_cmd, cwd=tmp_dir)

        # run vads in order to find the silence parts
        vads_cmd = (
          f"python3 {self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/vads.py "
          f"-r {self.rvad_root.get_path()} < {audio_path_file} > {vads_file}"
        )
        sp.check_call(vads_cmd, cwd=tmp_dir, shell=True)

        # use a fairseq script to remove the silence parts
        remove_sil_cmd = [
          "python3",
          f"{self.fairseq_root.get_path()}/examples/wav2vec/unsupervised/scripts/remove_silence.py",
          "--tsv",
          audio_path_file,
          "--vads",
          vads_file,
          "--out",
          ".",
        ]
        sp.check_call(remove_sil_cmd, cwd=tmp_dir)

        # in the manifest file, find the folders which are created by the fairseq script
        sub_paths = set()
        with open(audio_path_file, "r") as f:
          for line in f:
            # the first line is the given audio_dir as an absolute path
            if line.startswith("/"):
              continue
            # all other lines are the paths below audio_dir leading to the individual audio files
            # and a tab-separated label
            path, _ = line.split("\t")
            # sub_path is the path to the folder containing the audio file
            sub_path, _ = os.path.split(path)
            # store the sub_paths in a set to avoid duplicates
            sub_paths.add(sub_path)

        for sub_path in sub_paths:
          # the fairseq script only writes to out_dir, so we need to create subsub_paths by ourselves
          subsub_path, out_dir = os.path.split(sub_path)

          if subsub_path != "":
            # extend the output_dir by the subsub_path
            output_dir = os.path.join(output_dir_, subsub_path)
            os.makedirs(output_dir, exist_ok=True)
          else:
            output_dir = output_dir_
          # move the created folders to the correct out_dir in order to keep the same structure as the input dirs
          shutil.move(
            f"{tmp_dir}/{out_dir}",
            f"{output_dir}",
          )
        # remove the temporary files
        os.remove(audio_path_file)
        os.remove(vads_file)
