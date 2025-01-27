import os

from typing import Dict, Tuple, Union, Any, Optional, Sequence

from i6_core.lib import corpus
from sisyphus import Job, Task as SisTask, tk
from i6_core.util import uopen

class CorpusReplaceOrthFromPyDictJob(Job):
    """
    Merge HDF pseudo labels back into a bliss corpus
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
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

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
        assert isinstance(d, dict), "Has to be a dict containing the path to the search output file"
        
        assert c.fullname() in d["path"], "Corpus not in search output"
        
        d = eval(uopen(d["path"][c.fullname()], "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "only search output file with dict format is supported"
        
        j = 0
        for segment in segment_iterator:
            assert segment.fullname() in d, f"Segment {segment.fullname()} not in search output"
            line = d[segment.fullname()]
            if len(line) == 0:
                assert segment.recording is not None, f"Segment {segment.fullname()} has no recording"
                assert len(segment.recording.segments) == 1, f"Recording {segment.recording.fullname()} has more than one segment ({segment.recording.segments})"
                print(f"Segment {segment.fullname()} has empty pseudo label. It should be {segment.orth}")
                c.remove_recording(segment.recording)
                j += 1
            else:
                segment.orth = line.strip()
        n = len(c.recordings)
        m = len(d)
        assert m == n + j, f"Number of segments in corpus ({n+j}) does not match number of segments in search output ({m})"
        
        print(f"Number of segments with empty pseudo label: {j} out of {m}, Percentage: {j/m}")
        c.dump(self.out_corpus.get_path())
        
def get_ogg_zip_dict_pseudo_labels(bliss_corpus_dict: Dict[str, tk.Path]) -> Dict[str, tk.Path]:
    from i6_core.returnn.oggzip import BlissToOggZipJob
    import os

    ogg_zip_dict = {}
    for name, bliss_corpus in bliss_corpus_dict.items():
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus,
            no_audio=True,
            returnn_python_exe=None,
            returnn_root=None,
        )
        ogg_zip_job.add_alias(os.path.join("datasets", "LibriSpeech-PseudoLabels", "%s_ogg_zip_job" % name.replace('-', '_')))
        ogg_zip_dict[name] = ogg_zip_job.out_ogg_zip

    return ogg_zip_dict

class MetaDataset():
    """
    Represents :class:`MetaDataset` in RETURNN

    Only allows the MetaDataset to be used with an explicit control dataset.
    """

    def __init__(self,
                 data_map: Dict[str, Tuple[str, str]],
                 datasets: Dict[str, Dict],
                 seq_order_control_dataset: str,
                 other_opts: Optional[Dict[str, Any]] = None):
        """
        :param data_map:
        :param datasets:
        :param seq_order_control_dataset:
        :param dict other_opts:
        """
        self.data_map = data_map
        self.datasets = datasets
        assert seq_order_control_dataset in datasets
        self.seq_order_control_dataset = seq_order_control_dataset
        if other_opts is None:
            other_opts = {}
        self.other_opts = other_opts

    def as_returnn_opts(self):
        d = {
            'class': 'MetaDataset',
            'data_map': self.data_map,
            'datasets': self.datasets,
            'seq_order_control_dataset': self.seq_order_control_dataset
        }
        d.update(self.other_opts)
        return d