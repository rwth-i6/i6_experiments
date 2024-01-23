"""
Calc latency
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, List, Dict
import argparse
import gzip
import re
from decimal import Decimal
from xml.etree import ElementTree
from collections import OrderedDict
from returnn.datasets.generating import Vocabulary
from returnn.datasets.hdf import HDFDataset
from returnn.sprint.cache import open_file_archive, FileArchiveBundle, FileArchive
from returnn.util import better_exchook


ET = ElementTree


@dataclass
class Deps:
    """deps"""

    phone_alignments: Union[FileArchiveBundle, FileArchive]
    phone_alignment_sec_per_frame: Decimal
    lexicon: Lexicon
    corpus: Dict[str, BlissItem]
    chunk_labels: HDFDataset
    eoc_idx: int
    chunk_bpe_vocab: Vocabulary
    chunk_left_padding: Decimal
    chunk_stride: Decimal
    chunk_size: Decimal


def uopen(path: str, *args, **kwargs):
    if path.endswith(".gz"):
        return gzip.open(path, *args, **kwargs)
    else:
        return open(path, *args, **kwargs)


class BlissItem:
    """
    Bliss item.
    """

    def __init__(self, segment_name, recording_filename, start_time, end_time, orth, speaker_name=None):
        """
        :param str segment_name:
        :param str recording_filename:
        :param Decimal start_time:
        :param Decimal end_time:
        :param str orth:
        :param str|None speaker_name:
        """
        self.segment_name = segment_name
        self.recording_filename = recording_filename
        self.start_time = start_time
        self.end_time = end_time
        self.orth = orth
        self.speaker_name = speaker_name

    def __repr__(self):
        keys = ["segment_name", "recording_filename", "start_time", "end_time", "orth", "speaker_name"]
        return "BlissItem(%s)" % ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys])

    @property
    def delta_time(self):
        """
        :rtype: float
        """
        return self.end_time - self.start_time


def iter_bliss(filename):
    """
    :param str filename:
    :return: yields BlissItem
    :rtype: list[BlissItem]
    """
    corpus_file = open(filename, "rb")
    if filename.endswith(".gz"):
        corpus_file = gzip.GzipFile(fileobj=corpus_file)

    parser = ElementTree.XMLParser(target=ElementTree.TreeBuilder(), encoding="utf-8")
    context = iter(ElementTree.iterparse(corpus_file, parser=parser, events=("start", "end")))
    _, root = next(context)  # get root element
    name_tree = [root.attrib["name"]]
    elem_tree = [root]
    count_tree = [0]
    recording_filename = None
    for event, elem in context:
        if elem.tag == "recording":
            recording_filename = elem.attrib["audio"] if event == "start" else None
        if event == "end" and elem.tag == "segment":
            elem_orth = elem.find("orth")
            orth_raw = elem_orth.text or ""  # should be unicode
            orth_split = orth_raw.split()
            orth = " ".join(orth_split)
            elem_speaker = elem.find("speaker")
            if elem_speaker is not None:
                speaker_name = elem_speaker.attrib["name"]
            else:
                speaker_name = None
            segment_name = "/".join(name_tree)
            yield BlissItem(
                segment_name=segment_name,
                recording_filename=recording_filename,
                start_time=Decimal(elem.attrib["start"]),
                end_time=Decimal(elem.attrib["end"]),
                orth=orth,
                speaker_name=speaker_name,
            )
            root.clear()  # free memory
        if event == "start":
            count_tree[-1] += 1
            count_tree.append(0)
            elem_tree += [elem]
            elem_name = elem.attrib.get("name", None)
            if elem_name is None:
                elem_name = str(count_tree[-2])
            assert isinstance(elem_name, str)
            name_tree += [elem_name]
        elif event == "end":
            assert elem_tree[-1] is elem
            elem_tree = elem_tree[:-1]
            name_tree = name_tree[:-1]
            count_tree = count_tree[:-1]


class Lexicon:
    """
    Represents a bliss lexicon, can be read from and written to .xml files
    """

    def __init__(self, file: Optional[str] = None):
        self.phonemes = OrderedDict()  # type: OrderedDict[str, str] # symbol => variation
        self.lemmata = []  # type: List[Lemma]
        self.orth_to_lemma = {}  # type: Dict[str, Lemma]
        self.special_phones = {}  # type: Dict[str, Lemma]
        if file:
            self.load(file)

    def add_phoneme(self, symbol, variation="context"):
        """
        :param str symbol: representation of one phoneme
        :param str variation: possible values: "context" or "none".
            Use none for context independent phonemes like silence and noise.
        """
        self.phonemes[symbol] = variation

    def remove_phoneme(self, symbol):
        """
        :param str symbol:
        """
        del self.phonemes[symbol]

    def add_lemma(self, lemma):
        """
        :param Lemma lemma:
        """
        assert isinstance(lemma, Lemma)
        self.lemmata.append(lemma)
        for orth in lemma.orth:
            self.orth_to_lemma[orth] = lemma
        if lemma.special:
            for phon in lemma.phon:
                self.special_phones[phon] = lemma

    def load(self, path):
        """
        :param str path: bliss lexicon .xml or .xml.gz file
        """
        with uopen(path, "rt") as f:
            root = ET.parse(f)

        for phoneme in root.findall(".//phoneme-inventory/phoneme"):
            symbol = phoneme.find(".//symbol").text.strip()
            variation_element = phoneme.find(".//variation")
            variation = "context"
            if variation_element is not None:
                variation = variation_element.text.strip()
            self.add_phoneme(symbol, variation)

        for lemma in root.findall(".//lemma"):
            l = Lemma.from_element(lemma)
            self.add_lemma(l)

    def to_xml(self):
        """
        :return: xml representation, can be used with `util.write_xml`
        :rtype: ET.Element
        """
        root = ET.Element("lexicon")

        pi = ET.SubElement(root, "phoneme-inventory")
        for symbol, variation in self.phonemes.items():
            p = ET.SubElement(pi, "phoneme")
            s = ET.SubElement(p, "symbol")
            s.text = symbol
            v = ET.SubElement(p, "variation")
            v.text = variation

        for l in self.lemmata:
            root.append(l.to_xml())

        return root


class Lemma:
    """
    Represents a lemma of a lexicon
    """

    def __init__(
        self,
        orth: Optional[List[str]] = None,
        phon: Optional[List[str]] = None,
        synt: Optional[List[str]] = None,
        eval: Optional[List[List[str]]] = None,
        special: Optional[str] = None,
    ):
        """
        :param orth: list of spellings used in the training data
        :param phon: list of pronunciation variants. Each str should
            contain a space separated string of phonemes from the phoneme-inventory.
        :param synt: list of LM tokens that form a single token sequence.
            This sequence is used as the language model representation.
        :param eval: list of output representations. Each
            sublist should contain one possible transcription (token sequence) of this lemma
            that is scored against the reference transcription.
        :param special: assigns special property to a lemma.
            Supported values: "silence", "unknown", "sentence-boundary",
            or "sentence-begin" / "sentence-end"
        """
        self.orth = [] if orth is None else orth
        self.phon = [] if phon is None else phon
        self.synt = synt
        self.eval = [] if eval is None else eval
        self.special = special
        if isinstance(synt, list):
            assert not (len(synt) > 0 and isinstance(synt[0], list)), (
                "providing list of list is no longer supported for the 'synt' parameter "
                "and can be safely changed into a single list"
            )

    def __repr__(self):
        return "Lemma(orth=%r, phon=%r, synt=%r, eval=%r, special=%r)" % (
            self.orth,
            self.phon,
            self.synt,
            self.eval,
            self.special,
        )

    def to_xml(self):
        """
        :return: xml representation
        :rtype:  ET.Element
        """
        attrib = {"special": self.special} if self.special is not None else {}
        res = ET.Element("lemma", attrib=attrib)
        for o in self.orth:
            el = ET.SubElement(res, "orth")
            el.text = o
        for p in self.phon:
            el = ET.SubElement(res, "phon")
            el.text = p
        if self.synt is not None:
            el = ET.SubElement(res, "synt")
            for token in self.synt:
                el2 = ET.SubElement(el, "tok")
                el2.text = token
        for e in self.eval:
            el = ET.SubElement(res, "eval")
            for t in e:
                el2 = ET.SubElement(el, "tok")
                el2.text = t
        return res

    @classmethod
    def from_element(cls, e):
        """
        :param ET.Element e:
        :rtype: Lemma
        """
        orth = []
        phon = []
        synt = []
        eval = []
        special = None
        if "special" in e.attrib:
            special = e.attrib["special"]
        for orth_element in e.findall(".//orth"):
            orth.append(orth_element.text.strip() if orth_element.text is not None else "")
        for phon_element in e.findall(".//phon"):
            phon.append(phon_element.text.strip() if phon_element.text is not None else "")
        for synt_element in e.findall(".//synt"):
            tokens = []
            for token_element in synt_element.findall(".//tok"):
                tokens.append(token_element.text.strip() if token_element.text is not None else "")
            synt.append(tokens)
        for eval_element in e.findall(".//eval"):
            tokens = []
            for token_element in eval_element.findall(".//tok"):
                tokens.append(token_element.text.strip() if token_element.text is not None else "")
            eval.append(tokens)
        synt = None if not synt else synt[0]
        return Lemma(orth, phon, synt, eval, special)


def handle_segment(deps: Deps, segment_name: str) -> List[Decimal]:
    """handle segment"""
    corpus_entry = deps.corpus[segment_name]
    words = corpus_entry.orth.split()
    phone_align_ends = get_phone_alignment_word_ends(deps, segment_name)
    chunk_ends = get_chunk_ends(deps, segment_name)
    assert len(phone_align_ends) == len(chunk_ends) == len(words)
    res = []
    for word, phone_align_end, chunk_end in zip(words, phone_align_ends, chunk_ends):
        print(f"{word}: {phone_align_end} vs {chunk_end}, latency: {chunk_end - phone_align_end}sec")
        res.append(chunk_end - phone_align_end)
    return res


def get_phone_alignment_word_ends(deps: Deps, segment_name: str) -> List[Decimal]:
    """handle segment"""
    phone_alignment = deps.phone_alignments.read(segment_name, "align")
    corpus_entry = deps.corpus[segment_name]
    words = corpus_entry.orth.split()
    allophones = deps.phone_alignments.get_allophones_list()
    next_time_idx = 0
    word_idx = 0
    cur_word_phones = []
    res = []
    for time_idx, allophone_idx, state, weight in phone_alignment:
        assert next_time_idx == time_idx
        next_time_idx += 1
        allophone = allophones[allophone_idx]  # like: "[SILENCE]{#+#}@i@f" or "W{HH+AH}"
        m = re.match(r"([a-zA-Z\[\]#]+){([a-zA-Z\[\]#]+)\+([a-zA-Z\[\]#]+)}(@i)?(@f)?", allophone)
        assert m
        center, left, right, is_initial, is_final = m.groups()
        if center in deps.lexicon.special_phones:
            lemma = deps.lexicon.special_phones[center]
            if "" in lemma.orth:  # e.g. silence
                continue  # skip silence or similar
        if (
            time_idx + 1 < len(phone_alignment)
            and phone_alignment[time_idx + 1][1] == allophone_idx
            and phone_alignment[time_idx + 1][2] >= state
        ):
            continue  # skip to the last frame for this phoneme
        cur_word_phones.append(center)

        if is_final:
            lemma = deps.lexicon.orth_to_lemma[words[word_idx]]
            phones_s = " ".join(cur_word_phones)
            print(f"end time {time_idx * deps.phone_alignment_sec_per_frame}sec:", lemma.orth[0], "/", phones_s)
            if phones_s not in lemma.phon:
                raise Exception(f"Phones {phones_s} not in lemma {lemma}?")
            res.append(time_idx * deps.phone_alignment_sec_per_frame)

            cur_word_phones.clear()
            word_idx += 1
    assert word_idx == len(words)
    return res


def get_chunk_ends(deps: Deps, segment_name: str) -> List[Decimal]:
    """
    Example:

    chunk_size_dim = SpatialDim("chunk-size", 25)
    input_chunk_size_dim = SpatialDim("input-chunk-size", 150)
    sliced_chunk_size_dim = SpatialDim("sliced-chunk-size", 20)

    "_input_chunked": {
        "class": "window",
        "from": "source",
        "out_spatial_dim": chunked_time_dim,
        "stride": 120,
        "window_dim": input_chunk_size_dim,
        "window_left": 0,
    },

    # audio_features is 16.000 Hz, i.e. 16.000 frames per sec, 16 frames per ms.
    layer /'source':  # 100 frames per sec, 0.01 sec per frame, 10 ms per frame
        [B,T|'⌈(-199+time:var:extern_data:audio_features+-200)/160⌉'[B],F|F'mel_filterbank:feature-dense'(80)] float32
    layer /'_input_chunked':  # 1.2 sec per frame
        [B,T|'⌈(-199+time:var:extern_data:audio_features+-200)/19200⌉'[B],
         'input-chunk-size'(150),F|F'mel_filterbank:feature-dense'(80)] float32
    """
    corpus_entry = deps.corpus[segment_name]
    words = corpus_entry.orth.split()
    bpe_labels = deps.chunk_labels.get_data_by_seq_tag(segment_name, "data")
    bpe_labels_s = deps.chunk_bpe_vocab.get_seq_labels(bpe_labels)
    print(bpe_labels)
    print(bpe_labels_s)
    chunk_idx = 0
    cur_chunk_end_pos = -deps.chunk_left_padding + deps.chunk_size
    cur_word = ""
    word_idx = 0
    res = []
    for label_idx in bpe_labels:
        if label_idx == deps.eoc_idx:
            chunk_idx += 1
            cur_chunk_end_pos += deps.chunk_stride
            continue
        assert word_idx < len(words), f"{bpe_labels_s!r} does not fit to {corpus_entry.orth!r}"
        label = deps.chunk_bpe_vocab.id_to_label(label_idx)
        if label.endswith("@@"):
            cur_word += label[:-2]
            continue
        cur_word += label
        assert (
            cur_word == words[word_idx]
        ), f"{cur_word!r} != {words[word_idx]!r} in {bpe_labels_s!r} != {corpus_entry.orth!r}"
        print(f"end time {cur_chunk_end_pos}sec:", cur_word)
        res.append(cur_chunk_end_pos)
        word_idx += 1
        cur_word = ""
    assert word_idx == len(words) and not cur_word
    return res


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--phone-alignments", required=True, help="From RASR")
    arg_parser.add_argument("--phone-alignment-sec-per-frame", type=Decimal, default=Decimal("0.01"))
    arg_parser.add_argument("--allophone-file", required=True, help="From RASR")
    arg_parser.add_argument("--lexicon", required=True, help="XML")
    arg_parser.add_argument("--corpus", required=True, help="Bliss XML")
    arg_parser.add_argument("--chunk-labels", required=True, help="HDF dataset")
    arg_parser.add_argument("--eoc-idx", default=0, type=int, help="End-of-chunk idx")
    arg_parser.add_argument("--chunk-bpe-vocab", required=True, help="BPE vocab dict")
    arg_parser.add_argument(
        "--chunk-left-padding", type=Decimal, required=True, help="window_left in window layer, in sec"
    )
    arg_parser.add_argument("--chunk-stride", type=Decimal, required=True, help="stride in window layer, in sec")
    arg_parser.add_argument("--chunk-size", type=Decimal, required=True, help="window_dim in window layer, in sec")
    arg_parser.add_argument("--segment", nargs="*")
    args = arg_parser.parse_args()

    phone_alignments = open_file_archive(args.phone_alignments)
    phone_alignments.set_allophones(args.allophone_file)

    lexicon = Lexicon(args.lexicon)

    dataset = HDFDataset([args.chunk_labels])
    dataset.initialize()
    dataset.init_seq_order(epoch=1)

    corpus = {}
    for item in iter_bliss(args.corpus):
        corpus[item.segment_name] = item

    bpe_vocab = Vocabulary(args.chunk_bpe_vocab, unknown_label=None)

    deps = Deps(
        phone_alignments=phone_alignments,
        phone_alignment_sec_per_frame=args.phone_alignment_sec_per_frame,
        lexicon=lexicon,
        corpus=corpus,
        chunk_labels=dataset,
        eoc_idx=args.eoc_idx,
        chunk_bpe_vocab=bpe_vocab,
        chunk_left_padding=args.chunk_left_padding,
        chunk_stride=args.chunk_stride,
        chunk_size=args.chunk_size,
    )

    outliers = []
    res = []
    for segment_name in args.segment or corpus:
        print(corpus[segment_name])
        seg_latencies = handle_segment(deps, segment_name)
        if max(seg_latencies) > args.chunk_left_padding + args.chunk_size:
            outliers.append((segment_name, max(seg_latencies)))
        res += seg_latencies
    print("outliers:")
    for segment_name, latency in outliers:
        print(f"  {segment_name}: {latency}sec")
    if not outliers:
        print("  (no outliers)")
    print(f"avg latency: {sum(res) / len(res)}sec")
    print(f"max latency: {max(res)}sec")
    res = sorted(res)
    print(f"median latency: {res[len(res) // 2]}sec")
    print(f"p90 latency: {res[int(len(res) * 0.9)]}sec")
    print(f"p95 latency: {res[int(len(res) * 0.95)]}sec")
    print(f"p99 latency: {res[int(len(res) * 0.99)]}sec")


if __name__ == "__main__":
    better_exchook.install()
    main()
