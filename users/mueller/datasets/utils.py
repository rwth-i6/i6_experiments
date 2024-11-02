import itertools

from typing import Dict

from i6_core.lib import corpus
from sisyphus import Job, Task as SisTask, tk
from i6_core.util import uopen

class CorpusReplaceOrthFromPyDictJob(Job):
    """
    Merge HDF pseude labels back into a bliss corpus
    """

    def __init__(self, bliss_corpus, recog_words_file, segment_file=None):
        """
        :param Path bliss_corpus: Bliss corpus
        :param Path recog_words_file: a recog_words file
        :param Path|None segment_file: only replace the segments as specified in the segment file
        """
        self.bliss_corpus = bliss_corpus
        self.recog_words_file = recog_words_file
        self.segment_file = segment_file

        self.out_corpus = self.output_path("corpus.xml.gz")

    def tasks(self):
        yield SisTask("run", mini_task=True)

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        if self.segment_file:
            with uopen(self.segment_file.get_path(), "rt") as f:
                segments_whitelist = set(l.strip() for l in f.readlines() if len(l.strip()) > 0)
            segment_iterator = filter(lambda s: s.fullname() in segments_whitelist, c.segments())
        else:
            segment_iterator = c.segments()
            
        d = eval(uopen(self.recog_words_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "only search output file with dict format is supported"
        
        # TODO look at returnn/search.py SearchWordsDummyTimesToCTMJob

        for segment, line in itertools.zip_longest(segment_iterator, d):
            assert segment is not None, "there were more text file lines than segments"
            assert line is not None, "there were less text file lines than segments"
            assert len(line) > 0
            segment.orth = line.strip()

        c.dump(self.out_corpus.get_path())
        
def get_ogg_zip_dict_pseude_labels(bliss_corpus_dict: Dict[str, tk.Path]) -> Dict[str, tk.Path]:
    from i6_core.returnn.oggzip import BlissToOggZipJob
    import os

    ogg_zip_dict = {}
    for name, bliss_corpus in bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            no_conversion=True,
            returnn_python_exe=None,
            returnn_root=None,
        )
        ogg_zip_job.add_alias(os.path.join("datasets", "LibriSpeech-PseudoLabels", "%s_ogg_zip_job" % name.replace('-', '_')))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict