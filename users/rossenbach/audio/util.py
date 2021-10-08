import soundfile
from i6_core.lib import corpus

def run_duration_recover(source_corpus, target_corpus):
  """
  iterates over a single segment bliss corpus and uses the soundfile library to get the actuall recording length

  :param source_corpus:
  :param target_corpus:
  :return:
  """
  c = corpus.Corpus()
  c.load(source_corpus)

  for r in c.all_recordings():
    assert len(r.segments) == 1, "needs to be a single segment recording"
    old_duration = r.segments[0].end
    data, sample_rate = soundfile.read(open(r.audio, "rb"))
    new_duration = len(data) / sample_rate
    print("%s: %f vs. %f" % (r.segments[0].name, old_duration, new_duration))
    r.segments[0].end = new_duration

  c.dump(target_corpus)
