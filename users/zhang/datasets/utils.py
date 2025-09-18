import os

from typing import Dict, Tuple, Union, Any, Optional, Sequence

import re
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

def raw_text_spm_to_word(text: str) -> str:
    return text.replace(" ", "").replace ("▁", " ")

def raw_text_bpe_to_word(text: str) -> str:
    return text.replace("@@ ", "")

import re
from decimal import Decimal

def extract_record_id(key: str) -> str:
    """
    Returns a *string* record ID usable for grouping:
      - LibriSpeech-style: parent dir '672-122797-0033' -> '122797'
      - Corpus-style: parent dir 'es_US_cc_AP_Finance_100_20220907_channel1' -> the whole string
      - Time-range style: parent dir 'zm_website_29e5dff2' -> the whole string
    """
    parts = key.split('/')
    if len(parts) < 2:
        raise ValueError(f"Invalid key format: {key}")

    parent = parts[-2]  # identifier before the final segment

    # LibriSpeech-style pattern like '672-122797-0033'
    if re.fullmatch(r"\d+-\d+-\d+", parent):
        _, mid, _ = parent.split('-')
        return mid  # return as string

    # Generic/corpus-style: record id is the whole parent identifier
    return parent


def extract_sequence_num(key: str) -> int:
    """
    Returns the numeric sequence index for ordering within a record:
      - Corpus-style: final path segment (e.g., '/1') -> 1
      - LibriSpeech-style: last hyphen group in parent (e.g., '...-0033') -> 33
      - Fallback: last number within parent
    NOTE: Time-range tags (idx-start-end) are not handled here; they use a different sort path.
    """
    parts = key.split('/')
    if not parts:
        raise ValueError(f"Invalid key format: {key}")

    last = parts[-1]
    if re.fullmatch(r"\d+", last):
        return int(last)

    if len(parts) >= 2:
        parent = parts[-2]
        if re.fullmatch(r"\d+-\d+-\d+", parent):
            seq = parent.split('-')[-1]
            return int(seq)  # '0033' -> 33

        # Generic fallback: take the last number in parent if present
        nums = re.findall(r"\d+", parent)
        if nums:
            return int(nums[-1])

    raise ValueError(f"Cannot extract sequence number from key: {key}")


# ---- New: support for time-range style tails like '212-2434.110-2446.810' or '2434.110-2446.810' ----

_TIME_TAIL_RE = re.compile(r"(?:(\d+)-)?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\Z")

def _parse_time_range_tail(tail: str):
    """
    Parse tails of the form:
        idx-start-end      e.g. '212-2434.110-2446.810'
        start-end          e.g. '2434.110-2446.810'
    Returns (start: Decimal, end: Decimal, idx: int) or None if not a time-range tail.
    """
    m = _TIME_TAIL_RE.fullmatch(tail)
    if not m:
        return None
    idx_str, start_str, end_str = m.groups()
    start = Decimal(start_str)
    end = Decimal(end_str)
    idx = int(idx_str) if idx_str is not None else -1
    return (start, end, idx)


def sort_key_for_segment(key: str):
    """
    Unified sort key for all supported formats.

    For time-range tags (idx-start-end or start-end in the final path component):
        (record_id, start, end, idx, 0)

    For classic numeric-sequence tags:
        (record_id, seq_as_decimal, seq_as_decimal, -1, 1)

    Using the same tuple shape keeps Python's sort stable and comparable.
    """
    rec = extract_record_id(key)
    parts = key.split('/')
    tail = parts[-1] if parts else ''

    parsed = _parse_time_range_tail(tail)
    if parsed is not None:
        start, end, idx = parsed
        return (rec, start, end, idx, 0)

    # Fallback to your existing numeric sequence logic
    seq = Decimal(extract_sequence_num(key))
    return (rec, seq, seq, -1, 1)


def sort_dict_by_record(data: dict) -> dict:
    """
    Groups by record ID and orders within each record:
      - time-range tags: by start, then end, then idx
      - others: by numeric sequence
    Output is a dict whose iteration order reflects the sorted order.
    """
    sorted_keys = sorted(data.keys(), key=sort_key_for_segment)
    return {k: data[k] for k in sorted_keys}

class TextDictToDummyRecogOutJob(Job):
    """
    Convert pure text dict to Recogout: seq_tag: {[dummyscore, seq]}
    """

    def __init__(self, text_dict: tk.Path, *, spm: tk.Path = None, enable_unk: bool = False ,output_gzip: bool = True):
        """
        :param text_dict
        :param output_gzip: gzip the output
        """
        self.text_dict = text_dict
        self.spm = spm
        self.enable_unk = enable_unk
        self.dummy_rec_out = self.output_path("dummy_rec_out.py" + (".gz" if output_gzip else ""))
        self.rqmt = {"time": 1, "cpu": 1, "mem": 3}

    def tasks(self):
        """task"""
        yield SisTask("run", mini_task=True)

    def run(self):
        """run"""
        import i6_core.util as util
        text_d = eval(util.uopen(self.text_dict, "rt").read(),
                       {"nan": float("nan"), "inf": float("inf")})  # {seq_tag:[(score, hyp)]}
        spm = None
        if self.spm is not None:
            import sentencepiece
            spm = sentencepiece.SentencePieceProcessor(model_file=self.spm.get_path())
            if self.enable_unk:
                spm.set_encode_extra_options("unk")
                
        with util.uopen(self.dummy_rec_out, "wt") as out:
            out.write("{\n")
            for seq_tag, seq in text_d.items():
                out.write(f"{seq_tag!r}: [\n")
                if spm is not None:
                    pieces = spm.encode(seq.rstrip("\n"), out_type=str)
                    seq = " ".join(pieces)
                out.write(f"({0.0!r}, {seq!r}),\n")
                out.write("],\n")
            out.write("}\n")

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