import os
import gzip
import numpy as np

from typing import Dict, Tuple, Union, Any, Optional, Sequence

from i6_core.lib import corpus
from sisyphus import tools as sis_tools
from sisyphus import Job, Task as SisTask, tk
from i6_core.util import uopen
from i6_core.lib.hdf import get_returnn_simple_hdf_writer

from returnn.datasets.util.vocabulary import BytePairEncoding

from returnn_common.datasets_old_2022_10.interface import VocabConfig

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
                if isinstance(line, list):
                    assert len(line) == 1
                    line = line[0][1]
                    assert isinstance(line, str)
                segment.orth = line.strip()
        n = len(c.recordings)
        m = len(d)
        assert m == n + j, f"Number of segments in corpus ({n+j}) does not match number of segments in search output ({m})"
        
        print(f"Number of segments with empty pseudo label: {j} out of {m}, Percentage: {j/m}")
        c.dump(self.out_corpus.get_path())
        
class CorpusToHDF(Job):
    def __init__(self, bliss_corpus, vocab_config: VocabConfig, nbest: int):
        self.bliss_corpus = bliss_corpus
        self.out_file = self.output_path("bliss_targets.hdf")
        self.vocab_config = vocab_config
        self.nbest = nbest

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        segment_iterator = c.segments()
        
        bpe = BytePairEncoding(**self.vocab_config.get_opts().copy())
        
        SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
        out_hdf = SimpleHDFWriter(filename=self.out_file.get_path(), dim=None, ndim=2)
        for segment in segment_iterator:
            l = segment.orth.strip()
            key = segment.fullname()
            
            lines = []
            lengths = []
            if l:
                l = bpe.get_seq(l)
                lines.append(l)
                lengths.append(len(l))
            else:
                print(f"Empty pseudo label in {key}, {l}")
                lines.append([])
                lengths.append(0)
            
            for _ in range(self.nbest - 1):
                lines.append([])
                lengths.append(0)
            
            max_len = max(lengths)
            lines = [l + [self.vocab_config.get_eos_idx()] * (max_len - len(l)) for l in lines]
            lines = np.array(lines, dtype=np.int32)
            lines = np.expand_dims(lines, axis=0)
            assert lines.shape[1] == self.nbest
            lengths = np.array(lengths, dtype=np.int32)
            lengths = np.expand_dims(lengths, axis=0)
            assert lengths.shape[1] == self.nbest
            
            out_hdf.insert_batch(
                inputs=lines,
                seq_len={0: [lines.shape[1]], 1: [lines.shape[2]]},
                seq_tag=[key],
                extra={"lengths": lengths}
            )
            
        out_hdf.close()

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
        
class TargetsToHDF(Job):
    def __init__(self, corpus_name: str, recog_words_file: tk.Path, vocab_file: tk.Path, nbest: int, vocab_config: VocabConfig = None):
        """
        :param Path bliss_corpus: Bliss corpus
        """
        self.corpus_name = corpus_name
        self.recog_words_file = recog_words_file
        self.out_file = self.output_path("targets.hdf")
        self.vocab_file = vocab_file
        self.nbest = nbest
        self.vocab_config = vocab_config

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        d = eval(uopen(self.recog_words_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "Has to be a dict containing the path to the search output file"
        
        assert self.corpus_name in d["path"], "Corpus not in search output"
        
        d = eval(uopen(d["path"][self.corpus_name], "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict), "only search output file with dict format is supported"
        
        if self.vocab_config is not None:
            vocab = BytePairEncoding(**self.vocab_config.get_opts().copy())
        else:
            vocab = eval(uopen(self.vocab_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(vocab, dict), "Has to be a dict containing the vocab!"
            assert len(vocab) == 185
            vocab["<blank>"] = 184
        
        SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
        out_hdf = SimpleHDFWriter(filename=self.out_file.get_path(), dim=None, ndim=2)
        for k, v in d.items():
            assert isinstance(v, list)
            assert len(v) == self.nbest
            lines = []
            lengths = []
            for line in v:
                if self.vocab_config is not None:
                    l = line[1].strip()
                    if l and l != "":
                        l = vocab.get_seq(l)
                        if l in lines:
                            lines.append([])
                            lengths.append(0)
                        else:
                            lines.append(l)
                            lengths.append(len(l))
                    else:
                        print(f"Empty pseudo label in {v}")
                        lines.append([])
                        lengths.append(0)
                else:
                    l = line[1].strip().split(" ")
                    if l and l != [""]:
                        l = [vocab[e] for e in l]
                        if l in lines:
                            raise ValueError(f"Duplicate pseudo label {l} in {v}")
                            # lines.append([])
                        else:
                            lines.append(l)
                            lengths.append(len(l))
                    else:
                        # raise ValueError(f"Empty pseudo label in {v}")
                        print(f"Empty pseudo label in {v}")
                        lines.append([])
                        lengths.append(0)
            max_len = max(lengths)
            assert max_len != 0, f"Max length is 0 in {v}"
            if self.vocab_config is not None:
                eos_idx = self.vocab_config.get_eos_idx()
            else:
                eos_idx = vocab["</s>"]
            lines = [l + [eos_idx] * (max_len - len(l)) for l in lines]
            lines = np.array(lines, dtype=np.int32)
            lines = np.expand_dims(lines, axis=0)
            assert lines.shape[1] == self.nbest
            lengths = np.array(lengths, dtype=np.int32)
            lengths = np.expand_dims(lengths, axis=0)
            assert lengths.shape[1] == self.nbest
            out_hdf.insert_batch(
                inputs=lines,
                seq_len={0: [lines.shape[1]], 1: [lines.shape[2]]},
                seq_tag=[k],
                extra={"lengths": lengths}
            )
            
        out_hdf.close()
        
    @classmethod
    def hash(cls, parsed_args) -> str:
        # Extend the default hash() function.
        d = parsed_args.copy()
        if not d["vocab_config"]:
            d.pop("vocab_config")
        
        return sis_tools.sis_hash(d)
        
class DummyHDF(Job):
    def __init__(self, bliss_corpus, nbest: int):
        self.bliss_corpus = bliss_corpus
        self.out_file = self.output_path("bliss_dummy.hdf")
        self.nbest = nbest

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        c = corpus.Corpus()
        c.load(self.bliss_corpus.get_path())

        segment_iterator = c.segments()
        
        SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
        out_hdf = SimpleHDFWriter(filename=self.out_file.get_path(), dim=None)
        for segment in segment_iterator:
            key = segment.fullname()
            
            out_hdf.insert_batch(
                inputs=np.zeros((1,1), dtype=np.float32),
                seq_len={0: [1]},
                seq_tag=[key],
                extra={"lengths": np.full((1, self.nbest), -1, dtype=np.int32)}
            )
            
        out_hdf.close()
        
class ScoresHDF(Job):
    def __init__(self, corpus_name: str | None, bliss_corpus: tk.Path | None, recog_words_file: tk.Path | None, nbest: int):
        """
        :param Path bliss_corpus: Bliss corpus
        """
        self.corpus_name = corpus_name
        self.bliss_corpus = bliss_corpus
        self.recog_words_file = recog_words_file
        self.out_file = self.output_path("scores.hdf")
        self.nbest = nbest

    def tasks(self):
        yield SisTask("run", rqmt={"cpu": 4, "mem": 8, "time": 4})

    def run(self):
        if self.corpus_name is not None:
            d = eval(uopen(self.recog_words_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(d, dict), "Has to be a dict containing the path to the search output file"
            
            assert self.corpus_name in d["path"], "Corpus not in search output"
            d = eval(uopen(d["path"][self.corpus_name], "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(d, dict), "only search output file with dict format is supported"
            
            SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
            out_hdf = SimpleHDFWriter(filename=self.out_file.get_path(), dim=None)
            for k, v in d.items():
                assert isinstance(v, list)
                assert len(v) == self.nbest
                scores = [line[0] for line in v]
                scores = np.array(scores, dtype=np.float32)
                scores = np.expand_dims(scores, axis=0)
                assert scores.shape[1] == self.nbest
                out_hdf.insert_batch(
                    inputs=scores,
                    seq_len={0: [self.nbest]},
                    seq_tag=[k],
                )
                
            out_hdf.close()
        else:
            assert self.bliss_corpus is not None
            c = corpus.Corpus()
            c.load(self.bliss_corpus.get_path())

            segment_iterator = c.segments()
            
            SimpleHDFWriter = get_returnn_simple_hdf_writer(None)
            out_hdf = SimpleHDFWriter(filename=self.out_file.get_path(), dim=None)
            for segment in segment_iterator:
                key = segment.fullname()
                
                out_hdf.insert_batch(
                    inputs=np.full((1,1), float("inf"), dtype=np.float32),
                    seq_len={0: [1]},
                    seq_tag=[key],
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