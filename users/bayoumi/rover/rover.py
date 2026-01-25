#!/usr/bin/env bash

__all__ = ["RoverJob", "GetOptimalRoverJob", "ProcessRoverFileJob"]

from sisyphus import tk, Task

import sys
import argparse
import subprocess as sp
import json
import re
from pathlib import Path
from typing import List, Tuple, Iterator, Union

PERCENT_TOTAL_ERROR_RE = re.compile(
    r"^\s*Percent\s+Total\s+Error\s*=\s*.*?\(\s*(\d+)\s*\)\s*$",
    re.IGNORECASE | re.MULTILINE
)

class GetOptimalRoverJob(tk.Job):
    def __init__(self, ctms: List[Tuple[str, float, float, tk.Path]]):
        """
        Finds the optimal setting on dev set.

        :param List[Tuple[str, float, float, tk.Path]] ctms: A list of containers of a dev results along with its settings.
        """

        self.ctms = ctms
        self.out_report = self.output_path("report.txt")

    
    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):

        best_value = None
        best_meth = None
        best_alpha = None
        best_conf = None
        best_path = None

        for (meth, alpha, conf, path) in self.ctms:
            
            path = Path(str(path) + "/sclite.dtl")
            text = path.read_text(encoding="utf-8", errors="replace")

            m = PERCENT_TOTAL_ERROR_RE.search(text)

            value = int(m.group(1))

            if best_value is None or value < best_value:
                best_value = value
                best_meth = meth
                best_alpha = alpha
                best_conf = conf
                best_path = path

        data = {"meth": best_meth, "alpha": best_alpha, "conf": best_conf, "path": str(best_path)}
        with open(self.out_report, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

class RoverJob(tk.Job):
    def __init__(self, 
            name: str, 
            ctms: List[tk.Path], 
            config: Union[dict, tk.Path],
            rover_cmd: str = "/u/noureldin.bayoumi/work/thesis_work/rover/rover.sh",
            cpu: int = 1, 
            time: float = 1.0, 
            use_gpu: bool = False, 
            mem: float = 4.0):
        """
        Runs rover on a set of ctms with certain settings.

        :param str name: Name of the combination
        :param List[tk.Path] ctms: A list of ctms to be combined
        :param Union[dict, tk.Path] config: The settings rover runs for the combination
        :param str rover_cmd: The rover command containing the path to the shell script.
        """


        self.rover_cmd = rover_cmd
        self.ctms = ctms
        self.config = config
        self.setup = self.output_path(name + ".txt")
        self.out_ctm = self.output_path(name + ".ctm")
        self.out_putat = self.output_path(name + ".putat")

        self.cpu = cpu
        self.rqmt = {
            "time": time,
            "cpu": cpu,
            "gpu": 1 if use_gpu else 0,
            "mem": mem,
        }

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        with open(self.setup, "wt") as setup:
            for ctm in self.ctms:
                setup.write(str(ctm))
                setup.write("\n")

        if isinstance(self.config, tk.Path):
            with open(self.config, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        cmd = [self.rover_cmd, self.setup, str(self.config["alpha"]), str(self.config["meth"]), str(self.config["conf"]), self.out_ctm if self.config["meth"] != "putat" else self.out_putat]
        sp.check_call(cmd)

PUTATIVE_START_RE = re.compile(r'<putative_tag\s+file=([^ >]+)\s+chan=([^ >]+)\s*>')
PUTATIVE_END_RE   = re.compile(r'</putative_tag\s*>')
ATTRIB_RE = re.compile(
    r'<attrib\s+tag="([^"]+)"\s+t1="([^"]+)"\s+t2="([^"]+)"\s+conf="([^"]+)"\s+tag1="([^"]+)">'
)
INPUT_NAMES_RE = re.compile(r'<input\s+names="([^"]+)">')

def putat2ctm(file: tk.Path, output_path: tk.Path):

    with open(file, "r", encoding="utf-8", errors="replace") as fh, \
         open(output_path, "w", encoding="utf-8") as out:
        
        target = None
        prebuffer = []
        
        for line in fh:
            prebuffer.append(line)
            m = INPUT_NAMES_RE.search(line)
            if m:
                names = m.group(1).split()
                if names:
                    target = names[0]
                break
        
        all_lines_iter = iter(prebuffer + fh.readlines())

        in_block = False
        cur_file = None
        cur_chan = None
        block_attribs = []  # list of tuples (word, t1, t2, conf, tag1)

        def flush_block():
            nonlocal block_attribs, cur_file, cur_chan
            if not block_attribs:
                return
            
            target_entries = [a for a in block_attribs if a[4] == target]
            if not target_entries:
                block_attribs = []
                return  # Nothing to emit for this block
            
            word_t, t1_s, t2_s, conf_s, _tag1 = target_entries[0]
            
            try:
                t1 = float(t1_s)
                t2 = float(t2_s)
            except ValueError:
                t1, t2 = 0.0, 0.0
            dur = max(0.0, t2 - t1)

            sum_conf = 0.0
            for (w, _a1, _a2, c_s, _t) in block_attribs:
                if w == word_t:
                    try:
                        sum_conf += float(c_s)
                    except ValueError:
                        pass

            # CTM format: <utt/file> <chan> <start> <dur> <word> <conf>
            out.write(f"{cur_file} {cur_chan} {t1:.6f} {dur:.6f} {word_t} {sum_conf}\n")

            block_attribs = []

        for line in all_lines_iter:
            if not in_block:
                m = PUTATIVE_START_RE.search(line)
                if m:
                    cur_file = m.group(1)
                    cur_chan = m.group(2)
                    in_block = True
                    block_attribs = []
                continue

            if PUTATIVE_END_RE.search(line):
                flush_block()
                in_block = False
                cur_file = None
                cur_chan = None
                continue

            am = ATTRIB_RE.search(line)
            if am:
                word, t1, t2, conf, tag1 = am.groups()
                block_attribs.append((word, t1, t2, conf, tag1))

        if in_block:
            flush_block()

def round_ctm(input_file: tk.Path, output_file: tk.Path, op: Union[str, List[Tuple[str, float]]]):
    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                conf = float(parts[-1])
                for op_, value in op:
                    if op_ == "neg":
                        conf *= -value
                    if op_ == "div":
                        conf /= value
                    if op_ == "shift":
                        conf += value
                if conf > 1:
                    conf = 1
                parts[-1] = f"{conf:.4f}"
            except ValueError:
                pass
            fout.write(" ".join(parts) + "\n")

class ProcessRoverFileJob(tk.Job):
    def __init__(self, file: tk.Path, op: Union[str, List[Tuple[str, float]]]):
        """
        Processes a rover output either editing ctm confidences or converting .putat to .ctm

        :param tk.Path file: Path to the rover output
        :param Union[str, List[Tuple[str, float]]] op: A list of operations to be done on the rover output
        """

        self.file = file
        self.op = op
        self.out_dir = self.output_path("out.ctm")

    def task(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        if self.op == "putat2ctm":
            putat2ctm(self.file, self.out_dir)
        else:
            round_ctm(self.file, self.out_dir, self.op)
