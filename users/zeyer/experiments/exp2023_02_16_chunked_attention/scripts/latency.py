"""
Calc latency
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import argparse
import subprocess
import re
import gzip
from decimal import Decimal
from xml.etree import ElementTree
from collections import OrderedDict
from returnn.datasets.hdf import HDFDataset
from returnn.sprint.cache import WordBoundaries


ET = ElementTree


@dataclass
class Deps:
    """deps"""

    sprint_archiver_bin: str
    sprint_phone_alignment: str
    sprint_allophone_file: str

    sprint_lexicon: Lexicon
    labels_with_eoc_hdf: HDFDataset


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


def get_sprint_allophone_seq(deps: Deps, segment_name: str) -> List[str]:
    """sprint"""
    cmd = [
        deps.sprint_archiver_bin,
        deps.sprint_phone_alignment,
        segment_name,
        "--mode",
        "show",
        "--type",
        "align",
        "--allophone-file",
        deps.sprint_allophone_file,
    ]
    # output looks like:
    """
    <?xml version="1.0" encoding="ISO-8859-1"?>                                                                             
    <sprint>
      time= 0       emission=       115     allophone=      [SILENCE]{#+#}@i@f      index=  115     state=  0               
      time= 1       emission=       115     allophone=      [SILENCE]{#+#}@i@f      index=  115     state=  0               
      time= 2       emission=       115     allophone=      [SILENCE]{#+#}@i@f      index=  115     state=  0               
      time= 3       emission=       115     allophone=      [SILENCE]{#+#}@i@f      index=  115     state=  0               
      time= 4       emission=       115     allophone=      [SILENCE]{#+#}@i@f      index=  115     state=  0               
      time= 5       emission=       18025   allophone=      HH{#+W}@i       index=  18025   state=  0        
      time= 6       emission=       18025   allophone=      HH{#+W}@i       index=  18025   state=  0                       
      time= 7       emission=       67126889        allophone=      HH{#+W}@i       index=  18025   state=  1               
      time= 8       emission=       67126889        allophone=      HH{#+W}@i       index=  18025   state=  1               
      time= 9       emission=       67126889        allophone=      HH{#+W}@i       index=  18025   state=  1               
      time= 10      emission=       134235753       allophone=      HH{#+W}@i       index=  18025   state=  2               
      time= 11      emission=       134235753       allophone=      HH{#+W}@i       index=  18025   state=  2
    ...
    """
    out = subprocess.check_output(cmd)
    time_idx = 0
    res = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith(b"time="):
            continue
        line = line.decode("utf8")
        m = re.match(
            r"time=\s*([0-9]+)\s+"
            r"emission=\s*([0-9]+)\s+"
            r"allophone=\s*(\S+)\s+"
            r"index=\s*([0-9]+)\s+"
            r"state=\s*([0-9]*)",
            line,
        )
        assert m, f"failed to parse line: {line}"
        t, emission, allophone, index, state = m.groups()
        assert int(t) == time_idx
        res += [allophone]
        time_idx += 1
    return res


def get_sprint_word_ends(deps: Deps, segment_name: str) -> List[int]:
    pass


def handle_segment(deps: Deps, segment_name: str):
    """handle segment"""
    pass


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--archiver-bin", required=True)
    arg_parser.add_argument("--phone-alignment", required=True)
    arg_parser.add_argument("--allophone-file", required=True)
    arg_parser.add_argument("--lexicon", required=True)
    arg_parser.add_argument("--corpus", required=True)
    args = arg_parser.parse_args()

    deps = Deps(
        sprint_archiver_bin=args.archiver_bin,
        sprint_phone_alignment=args.phone_alignment,
        sprint_allophone_file=args.allophone_file,
        sprint_lexicon=Lexicon(args.lexicon),
        labels_with_eoc_hdf=HDFDataset([args.corpus]),
    )

    for item in iter_bliss(args.corpus):
        print(item)
        print(get_sprint_allophone_seq(deps, item.segment_name))
        break


if __name__ == "__main__":
    main()
