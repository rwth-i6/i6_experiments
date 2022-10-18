
"""
Recipes to prepare LibriSpeech as a dataset for RETURNN.
As another reference, also see here:
https://github.com/rwth-i6/returnn-experiments/blob/master/2018-asr-attention/librispeech/full-setup-attention/
"""

from sisyphus import *
from .common import OggZipDataset
import os
import tempfile
import typing


Parts = [
  "dev-clean", "dev-other",
  "test-clean", "test-other",
  "train-clean-100", "train-clean-360", "train-other-500"]


class DownloadLibrispeechJob(Job):
  ExistingPath = None  # type: typing.Optional[str]
  _BaseUrl = "http://www.openslr.org/resources/12/"
  FileExt = ".tar.gz"

  def __init__(self):
    self.existing_path = self.ExistingPath
    self.output_dir = self.output_path("downloads", directory=True)

  @classmethod
  def hash(cls, parsed_args):
    return "static"  # always the same, independent from args

  def _download_file(self, filename):
    """
    :param str filename:
    """
    self.sh("wget -c %s%s" % (self._BaseUrl, filename))
    assert os.path.exists(filename)

  def run(self):
    os.chdir(self.output_dir.get_path())
    for part in Parts:
      filename = part + self.FileExt
      print("download:", filename)
      if self.existing_path and os.path.exists("%s/%s" % (self.existing_path, filename)):
        os.symlink("%s/%s" % (self.existing_path, filename), filename)
      else:
        self._download_file(filename)
      assert os.path.exists(filename)
    self._download_file("md5sum.txt")
    self.sh("md5sum -c --ignore-missing md5sum.txt")

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 10}, mini_task=True)


class ConvertLibrispeechAudioToZipJob(Job):
  """
  Extract the downloaded tar.gz files,
  convert to OGG,
  and then zip it again.
  """

  ExistingPath = None  # type: typing.Optional[str]

  def __init__(self, download_dir):
    """
    :param Path download_dir:
    """
    self.download_dir = download_dir
    self.existing_path = self.ExistingPath
    self.output_dir = self.output_path("ogg-zips", directory=True)

  @classmethod
  def hash(cls, parsed_args):
    return "static"  # always the same, independent from args

  def run(self):
    my_dir = os.getcwd()
    for part in Parts:
      zip_filename = "%s.zip" % part
      if self.existing_path and os.path.exists("%s/%s" % (self.existing_path, zip_filename)):
        os.symlink("%s/%s" % (self.existing_path, zip_filename), "%s/%s" % (self.output_dir.get_path(), zip_filename))
      else:
        with tempfile.TemporaryDirectory("librispeech-%s" % part) as temp_dir:
          os.chdir(temp_dir)
          tar_filename = "%s/%s.tar.gz" % (self.download_dir, part)
          self.sh("tar xf %s" % tar_filename)
          self.sh('find LibriSpeech -name "*.flac" -exec ffmpeg -i "{}" "{}.ogg" ";" -exec rm -f "{}" ";"')
          self.sh('zip -qdgds 500m -0 %s.zip -r LibriSpeech' % part)
          assert os.path.exists(zip_filename)
          self.sh("mv %s %s/" % (zip_filename, self.output_dir.get_path()))
          os.chdir(my_dir)
      assert os.path.exists("%s/%s" % (self.output_dir.get_path(), zip_filename))

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 20})


def get_dataset_zip_dir():
  """
  :rtype: Path
  """
  download = DownloadLibrispeechJob()
  oggzip = ConvertLibrispeechAudioToZipJob(download_dir=download.output_dir)
  return oggzip.output_dir


class CreateLibrispeechTxtJob(Job):
  """
  Create separate txt files to be used with :class:`returnn.OggZipDataset`.
  Example:
    https://github.com/rwth-i6/returnn-experiments/blob/master/2019-asr-e2e-trafo-vs-lstm/tedlium2/full-setup/03_convert_to_ogg.py
  """

  def __init__(self, dataset_dir):
    """
    :param Path dataset_dir: e.g. via :class:`ConvertLibrispeechAudioToZipJob`. contains the zip files
    """
    self.dataset_dir = dataset_dir
    self.output_dir = self.output_path("txt", directory=True)

  @classmethod
  def hash(cls, parsed_args):
    return "static"  # always the same, independent from args

  def run(self):
    import os
    from zipfile import ZipFile, ZipInfo
    import subprocess
    import tempfile
    for part in Parts:
      dest_meta_filename = "%s.txt" % part
      dest_meta_file = open(dest_meta_filename, "w")
      dest_meta_file.write("[\n")
      zip_filename = "%s/%s.zip" % (self.dataset_dir, part)
      assert os.path.exists(zip_filename)
      zip_file = ZipFile(zip_filename)
      assert zip_file.filelist
      assert zip_file.filelist[0].filename.startswith("LibriSpeech/")
      count_lines = 0
      for info in zip_file.filelist:
        assert isinstance(info, ZipInfo)
        path = info.filename.split("/")
        assert path[0] == "LibriSpeech", "does not expect %r (%r)" % (info, info.filename)
        if path[1].startswith(part):
          subdir = path[1]  # e.g. "train-clean-100"
          assert subdir == part
          if path[-1].endswith(".trans.txt"):
            print("read", part, path[-1])
            for line in zip_file.read(info).decode("utf8").splitlines():
              seq_name, txt = line.split(" ", 1)  # seq_name is e.g. "19-198-0000"
              count_lines += 1
              ogg_filename = "%s/%s.flac.ogg" % ("/".join(path[:-1]), seq_name)
              ogg_bytes = zip_file.read(ogg_filename)
              assert len(ogg_bytes) > 0
              # ffprobe does not work correctly on piped input. That is why we have to store it in a temp file.
              with tempfile.NamedTemporaryFile(suffix=".ogg") as temp_file:
                temp_file.write(ogg_bytes)
                temp_file.flush()
                duration_str = subprocess.check_output(
                  ["ffprobe", temp_file.name,
                   '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'compact'],
                  stderr=subprocess.STDOUT).decode("utf8").strip()
                duration_str = duration_str.split("=")[-1]  # e.g. "format|duration=10.028000"
              assert float(duration_str) > 0  # just a check
              dest_meta_file.write(
                "{'text': %r, 'file': %r, 'duration': %s},\n" % (txt, ogg_filename, duration_str))
      dest_meta_file.write("]\n")
      dest_meta_file.close()
      self.sh("gzip %s" % dest_meta_filename)
      assert os.path.exists("%s.gz" % dest_meta_filename)
      self.sh("mv %s.gz %s/" % (dest_meta_filename, self.output_dir.get_path()))
      assert os.path.exists("%s/%s.txt.gz" % (self.output_dir.get_path(), part))

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 10}, mini_task=True)


def get_dataset_txt_dir():
  """
  :rtype: Path
  """
  zip_dir = get_dataset_zip_dir()
  librispeech_txt = CreateLibrispeechTxtJob(dataset_dir=zip_dir)
  librispeech_txt_dir = librispeech_txt.output_dir
  return librispeech_txt_dir


def get_ogg_zip_dataset(part):
  """
  :param str part: e.g. "train", "dev", "clean", or more specific like "train-clean-100", etc.
  :rtype: OggZipDataset
  """
  selected_parts = [part_ for part_ in Parts if part_.startswith(part)]
  assert len(selected_parts) > 0, "invalid part %r. parts: %r" % (part, Parts)
  if len(selected_parts) == 1:
    assert selected_parts[0] == part, "invalid part %r. parts: %r" % (part, Parts)
  zip_dir = get_dataset_zip_dir()
  txt_dir = get_dataset_txt_dir()
  paths = []
  for part_ in selected_parts:
    paths += [
      "%s/%s" % (zip_dir.get_path(), "%s.zip" % part_),
      "%s/%s" % (txt_dir.get_path(), "%s.txt.gz" % part_)]
  return OggZipDataset(
    name=part, dependent_paths=[zip_dir, txt_dir], path=paths,
    other_opts=dict(zip_audio_files_have_name_as_prefix=False))


def get_train_transcriptions_txt():
  """
  :rtype: Path
  """
  dataset = get_ogg_zip_dataset("train")
  from .common import ogg_zip_dataset_to_txt
  return ogg_zip_dataset_to_txt(dataset)


class OldToNewTranscriptions(Job):
  def __init__(self, input_txt: Path):
    super(OldToNewTranscriptions, self).__init__()
    self.input_txt = input_txt
    self.output_txt = self.output_path("output.txt.gz")

  def run(self):
    import re
    from recipe.utils import generic_open
    input_transcriptions = eval(generic_open(self.input_txt.get_path()).read())
    print("Create file:", self.output_txt.get_path())
    count = 0
    with generic_open(self.output_txt.get_path(), "w") as out:
      out.write("{\n")
      for seq_tag, txt in sorted(input_transcriptions.items()):
        # old seq tag 'dev-other-116-288045-0000'
        # new seq tag 'LibriSpeech/dev-other/3663/172528/3663-172528-0000.flac.ogg'
        assert isinstance(txt, str)
        m = re.match(r"([a-z\-]+)-([0-9]+)-([0-9]+)-([0-9]+)", seq_tag)
        seq_tag = f"LibriSpeech/{m.group(1)}/{m.group(2)}/{m.group(3)}/{m.group(2)}-{m.group(3)}-{m.group(4)}.flac.ogg"
        out.write("%r: %r,\n" % (seq_tag, txt))
        count += 1
      out.write("}\n")
    print("Num seqs:", count)

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 10}, mini_task=True)
