"""
Parse job log file run with beam search and debug_out enabled
and collect statistics on ending and pruning.
"""

from __future__ import annotations
from typing import Optional, Tuple, TextIO
import os
from io import TextIOWrapper
import tarfile
import argparse
from decimal import Decimal
import gzip
import xml.etree.ElementTree as ElementTree
import sentencepiece as spm
from contextlib import contextmanager


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("jobdir", help="The directory containing the job files")
    arg_parser.add_argument("--spm", required=True, help="sentence piece model file")
    arg_parser.add_argument("--bliss", required=True, help="Bliss XML file for the corpus")
    arg_parser.add_argument("--beam", type=int, required=True, help="Beam size")
    args = arg_parser.parse_args()

    print(f"Open job log in {args.jobdir}")
    with open_job_log(args.jobdir) as (log, log_fn, job_dir):
        job_log = log.read().splitlines()

    print(f"Open job output in {job_dir}")
    job_output = eval(open(f"{job_dir}/output/output.py.gz").read())
    assert isinstance(job_output, dict)

    print("Open Bliss XML corpus")
    bliss = {seq.segment_name: seq for seq in iter_bliss(args.bliss)}

    print("Open SentencePiece model")
    sp = spm.SentencePieceProcessor(model_file=args.spm)

    seq_tag_iter = iter(job_output.keys())
    prev_step: Optional[int] = None
    cur_seqs = []
    total_num_seqs = 0
    total_num_seqs_finished = 0
    total_act_num_steps = 0
    total_num_steps = 0
    total_act_hyps = 0
    total_len_orth = 0
    for line in job_log:
        if line.startswith("DEBUG: "):
            line = line[len("DEBUG: ") :]
            content = eval(f"dict({line})")
            step = content["step"]
            act_beam_sizes = content["act_beam_sizes"]
            assert (
                isinstance(step, int)
                and isinstance(act_beam_sizes, list)
                and all(isinstance(x, int) for x in act_beam_sizes)
            )
            assert ((prev_step is not None) and (step == prev_step + 1)) or step == 0
            if step == 0:
                cur_seqs.clear()
                for i in range(len(act_beam_sizes)):
                    seq_tag = next(seq_tag_iter)
                    seq = bliss[seq_tag]
                    cur_seqs.append(seq)
                total_num_seqs += len(act_beam_sizes)
            assert len(act_beam_sizes) == len(cur_seqs)
            for i, (seq, size) in enumerate(zip(cur_seqs, act_beam_sizes)):
                if not seq:  # already finished before
                    continue
                if size == 0:  # finished now
                    total_num_seqs_finished += 1
                    orth = seq.orth
                    orth_pieces = sp.encode(orth, out_type=str)
                    print(
                        f"seq {seq.segment_name} finished: step {step}, orth len {len(orth_pieces)}, orth {orth}"
                    )
                    total_len_orth += len(orth_pieces)
                    cur_seqs[i] = None
                    continue
                total_num_steps += 1
                if step > 0:
                    total_act_num_steps += 1
                    total_act_hyps += size

    assert total_num_seqs == total_num_seqs_finished  # just a sanity check
    try:
        next(seq_tag_iter)
    except StopIteration:
        pass  # exactly what we expect, we reached the end
    else:
        raise Exception("Mismatch between num seqs in output and in log")

    print("Num seqs:", total_num_seqs)
    print("Num steps:", total_num_steps)
    print("Avg num act hyps / step:", total_act_hyps / total_act_num_steps * 100., "%")
    print("Avg end diff to orth:", (total_num_steps / total_len_orth - 1) * 100., "%")


@contextmanager
def open_job_log(job: str, task: str = "run", index: int = 1) -> Tuple[TextIO, str, str]:
    """
    :yield: log file, log file name, job directory
    """
    log_base_fn = f"log.{task}.{index}"
    if job.endswith(log_base_fn):
        log_fn = job
    else:
        log_fn = f"{job}/{log_base_fn}"
    job_dir = os.path.dirname(log_fn)
    if os.path.exists(log_fn):
        yield open(log_fn), log_fn, job_dir
        return
    tar_fn = f"{job}/finished.tar.gz"
    if os.path.exists(tar_fn):
        with tarfile.open(tar_fn) as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if f.name == log_base_fn:
                    f = tarf.extractfile(f)
                    yield TextIOWrapper(f), f"{tar_fn}:{log_base_fn}", job_dir
                    return
        raise Exception(f"Did not found {log_base_fn} in {tar_fn}")
    raise Exception(f"Did neither find {log_fn} nor {tar_fn}")


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


if __name__ == "__main__":
    main()
