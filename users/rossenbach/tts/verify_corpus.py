from sisyphus import *

from i6_core.lib import corpus

class VerifyCorpus(Job):
  """
  verifies the audio files of a bliss corpus by loading it with the soundfile library
  """

  def __init__(self, bliss_corpus):
    self.bliss_corpus = bliss_corpus

    self.out = self.output_path("errors.log")

  def tasks(self):
    yield Task('run', mini_task=True)

  def run(self):
    import soundfile

    c = corpus.Corpus()
    c.load(tk.uncached_path(self.bliss_corpus))

    out_file = open(tk.uncached_path(self.out), "wt")

    success = True

    for r in c.all_recordings():
      try:
        temp = soundfile.read(open(r.audio, "rb"))
      except:
        print("error in file %s" % r.audio)
        out_file.write("error in file %s\n" % r.audio)
        success = False

    assert success, "there was an error, please see error.log"
