import os
import gzip
import numpy as np

from typing import Dict, Tuple, Union, Any, Optional, Sequence

from i6_core.lib import corpus
from sisyphus import tools as sis_tools
from sisyphus import Job, Task as SisTask, tk
from i6_core.util import uopen
from i6_core.lib.hdf import get_returnn_simple_hdf_writer

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
        self.scores_file = None

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
        # if self.return_scores:
        #     SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
        #     out_hdf = SimpleHDFWriter(filename=self.scores_file.get_path(), dim=None)
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
                if isinstance(line, list):
                    lines = []
                    for e in line:
                        new_str = e[1].strip()
                        if new_str:
                            if new_str in lines:
                                lines.append("")
                                # raise ValueError(f"Duplicate pseudo label {new_str} in segment {segment.fullname()}")
                            else:
                                lines.append(new_str)
                        else:
                            print(f"Empty pseudo label in segment {segment.fullname()}")
                            lines.append("")
                    line = " ZZZZZ ".join(lines)
                    # out_hdf.insert_batch(
                    #     inputs=np.array(scores, dtype=np.float32).reshape(1, -1),
                    #     seq_len=[len(scores)],
                    #     seq_tag=[segment.fullname()],
                    # )
                    segment.orth = line
                else:
                    segment.orth = line.strip()
        n = len(c.recordings)
        m = len(d)
        assert m == n + j, f"Number of segments in corpus ({n+j}) does not match number of segments in search output ({m})"
        
        # if self.return_scores:
        #     out_hdf.close()
        
        print(f"Number of segments with empty pseudo label: {j} out of {m}, Percentage: {j/m}")
        c.dump(self.out_corpus.get_path())

class GetAlignmentTargets(Job):
    def __init__(self, corpus_name: str, recog_words_file: tk.Path, vocab_file: tk.Path):
        """
        :param Path bliss_corpus: Bliss corpus
        """
        self.corpus_name = corpus_name
        self.recog_words_file = recog_words_file
        self.out_file = self.output_path("align_targets.hdf")
        self.vocab_file = vocab_file

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        d = eval(uopen(self.recog_words_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "Has to be a dict containing the path to the search output file"
        
        assert self.corpus_name in d["path"], "Corpus not in search output"
        
        if d["path"][self.corpus_name].endswith(".hdf"):
            os.symlink(d["path"][self.corpus_name], self.out_file.get_path())
            return
        
        d = eval(uopen(d["path"][self.corpus_name], "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "only search output file with dict format is supported"
        
        vocab = eval(uopen(self.vocab_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(vocab, dict), "Has to be a dict containing the vocab!"
        assert len(vocab) == 185
        vocab["<blank>"] = 184
        
        SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
        out_hdf = SimpleHDFWriter(filename=self.out_file.get_path(), dim=None)
        for k, v in d.items():
            assert isinstance(v, list)
            # assert len(v) > 0
            assert len(v) == 1
            v = v[0]
            l = v[1].strip().split(" ")
            l = [vocab[e] for e in l]
            out_hdf.insert_batch(
                inputs=np.array(l, dtype=np.int32).reshape(1, -1),
                seq_len=[len(l)],
                seq_tag=[k],
            )
        out_hdf.close()

class GetScoresDummy(Job):
    """
    Creates a dummy with scores for corpus without pseudo labels
    """

    def __init__(self, bliss_corpus: tk.Path, pseudo_nbest: int):
        """
        :param Path bliss_corpus: Bliss corpus
        """
        self.bliss_corpus = bliss_corpus
        self.pseudo_nbest = pseudo_nbest
        self.scores_file = self.output_path("dummy_scores.hdf")

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())
        n = len(list(c.segments()))
        scores = [0.0] + [-1e30] * (self.pseudo_nbest - 1)
        scores = [scores] * n
        scores = np.array(scores, dtype=np.float32)
        tags = [segment.fullname() for segment in c.segments()]
        
        assert scores.shape == (n, self.pseudo_nbest)
        
        SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
        out_hdf = SimpleHDFWriter(filename=self.scores_file.get_path(), dim=None)
        out_hdf.insert_batch(
            inputs=scores,
            seq_len=[self.pseudo_nbest] * n,
            seq_tag=tags,
        )
        out_hdf.close()
        
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