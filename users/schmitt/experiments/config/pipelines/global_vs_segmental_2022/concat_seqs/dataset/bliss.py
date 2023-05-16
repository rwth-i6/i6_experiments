"""
Take a Bliss XML (Sprint/RASR corpus file),
convert it to OggZipDataset or whatever we need.
"""

from sisyphus import *
import os
from .common import OggZipDataset


class BlissToOggZipDatasetJob(Job):
  """
  Basically wraps returnn/tools/bliss-to-ogg-zip.py.
  """
  ExistingCache = {}  # (bliss,segments) -> ogg-zip

  def __init__(self, name, bliss, segments=None):
    """
    :param str name: hash depends only on this, nothing else
    :param Path bliss:
    :param Path|None segments:
    """
    self.bliss_xml = bliss
    self.segments_file = segments
    if (bliss, segments) in self.ExistingCache:
      self._have_existing = True
      self.output_ogg_zip_path = self.ExistingCache[(bliss, segments)]
    else:
      self._have_existing = False
      self.output_ogg_zip_path = self.output_path("ogg.zip")
    self.output_ogg_zip = OggZipDataset(name=name, path=self.output_ogg_zip_path)

  @classmethod
  def hash(cls, parsed_args):
    return parsed_args["name"]

  def run(self):
    import subprocess
    args = [
      "%s/returnn/tools/bliss-to-ogg-zip.py" % tk.gs.BASE_DIR,
      self.bliss_xml.get_path()]
    if self.segments_file:
      args += ["--subset_segment_file", self.segments_file.get_path()]
    args += ["--output", self.output_ogg_zip_path.get_path()]
    print("$ %s" % " ".join(args))
    subprocess.check_call(args)
    assert os.path.exists(self.output_ogg_zip_path.get_path())

  def tasks(self):
    if self._have_existing:
      return  # nothing to do
    yield Task('run', rqmt={'mem': 2, 'time': 20})


def bliss_to_ogg_zip(name, bliss, segments=None):
  """
  :param str name: hash depends only on this, nothing else
  :param Path bliss:
  :param Path|None segments:
  :rtype: OggZipDataset
  """
  return BlissToOggZipDatasetJob(name=name, bliss=bliss, segments=segments).output_ogg_zip


class BlissToTxtJob(Job):
  """
  Alternative to :func:`common.ogg_zip_dataset_to_txt`.
  """
  def __init__(self, bliss):
    """
    :param Path bliss:
    """
    self.bliss = bliss
    self.output_txt = self.output_path("output.txt.gz")

  def run(self):
    # noinspection PyProtectedMember
    from returnn.LmDataset import _iter_bliss as iter_bliss
    import gzip

    f = gzip.open(self.output_txt.get_path(), "wt")

    def callback(orth):
      """
      :param str orth:
      """
      f.write("%s\n" % orth)

    iter_bliss(filename=self.bliss.get_path(), callback=callback)
    f.close()

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 1}, mini_task=True)


def bliss_to_txt(bliss):
  """
  :param Path bliss:
  :rtype: Path
  """
  return BlissToTxtJob(bliss=bliss).output_txt
