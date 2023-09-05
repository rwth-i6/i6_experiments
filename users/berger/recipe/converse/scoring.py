import json
import re
import subprocess as sp
from typing import Dict, List

from sisyphus import Job, Task, Path


class CtmSegment:
    def __init__(self, name: str, words: List[str], start: float, end: float):
        self.name = name
        self.words = words
        self.start = start
        self.end = end

    @classmethod
    def from_lines(cls, lines: List[str]):
        assert len(lines) > 0
        assert lines[0].startswith(";;"), f"unexpected first line: {lines[0]}"
        name = lines[0].split()[1]
        words = [line.split()[-2] for line in lines[1:]]
        start, end = [float(time) for time in lines[0].split()[2].strip("()").split("-")]
        if end == float("inf"):
            start = float(lines[1].split()[2])
            end = float(lines[-1].split()[2]) + float(lines[-1].split()[3])
        return cls(name, words, start, end)


class CtmFile:
    def __init__(self, filename: Path):
        self.filename = filename
        self._segments = []
        self.load()

    def load(self):
        with open(self.filename.get(), "r") as f:
            content = f.readlines()

        line_buffer = []
        for line in content:
            if line.startswith(";; <"):  # header line
                continue
            elif line.startswith(";;"):  # new segment start
                if len(line_buffer) > 0:
                    self._segments.append(CtmSegment.from_lines(line_buffer))
                line_buffer = [line]
            else:  # regular content line
                line_buffer.append(line)

        if len(line_buffer) > 0:  # add last segment
            self._segments.append(CtmSegment.from_lines(line_buffer))

    def get_matched_channel_segments(self) -> Dict[str, Dict[int, CtmSegment]]:
        """
        Assuming that the ctm file contains segments for multiple channels and multiple segments per channel,
        get the matching segments based on the name which ends on _<channel>_<segment_id>, e.g.
        overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0_segment_5_1_2 (channel 1, segment 2)
        """
        matched_segments = {}
        for seg in self._segments:
            rec_name = seg.name.split("/")[-2]
            channel = int(rec_name.split("_")[-1])
            seg_base_name = seg.name.replace(rec_name, re.sub(f"_{channel}$", "", rec_name))
            if seg_base_name not in matched_segments:
                matched_segments[seg_base_name] = {}
            matched_segments[seg_base_name][channel] = seg
        return matched_segments

    def get_matched_channel_multi_segments(self) -> Dict[str, Dict[int, Dict[int, CtmSegment]]]:
        """
        Assuming that the ctm file contains segments for multiple channels and multiple segments per channel,
        get the matching segments based on the name which ends on _<channel>_<segment_id>, e.g.
        overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0_segment_5_1_2 (channel 1, segment 2)
        """
        matched_segments = {}
        for seg in self._segments:
            rec_name = seg.name.split("/")[-2]
            channel = int(rec_name.split("_")[-2])
            seg_id = int(rec_name.split("_")[-1])
            seg_base_name = seg.name.replace(rec_name, re.sub(f"_{channel}_{seg_id}$", "", rec_name))
            if seg_base_name not in matched_segments:
                matched_segments[seg_base_name] = {}
            if channel not in matched_segments[seg_base_name]:
                matched_segments[seg_base_name][channel] = {}
            matched_segments[seg_base_name][channel][seg_id] = seg
        return matched_segments


class MultiChannelCtmToStmJob(Job):
    """
    This job takes the ctm file produced by the LatticeToCtmJob which contains all channels with
    suffixes (e.g. "_0", "_1") and converts it to an stm file that can be scored with the MeetEval toolkit.
    """

    def __init__(
        self,
        ctm_file: Path,
    ):
        """
        :param ctm_file: input ctm file
        """
        self.ctm_file = ctm_file

        self.out_stm_file = self.output_path("hyp.stm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        ctm_file = CtmFile(self.ctm_file)
        segments = ctm_file.get_matched_channel_segments()

        with open(self.out_stm_file.get(), "w") as stm_file:
            for seg in segments:
                for channel in segments[seg]:
                    name = segments[seg][channel].name.replace(f"_{channel}/", "/")
                    stm_content = []
                    stm_line = (
                        f"{name.split('/')[-2]} 0 {channel} "
                        f"{segments[seg][channel].start} "
                        f"{segments[seg][channel].end}"
                    )
                    for word in segments[seg][channel].words:
                        stm_line += f" {word}"
                    stm_content.append(stm_line)
                    stm_file.write("\n".join(stm_content) + "\n")


class MultiChannelMultiSegmentCtmToStmJob(Job):
    """
    This job takes the ctm file produced by the LatticeToCtmJob which contains all channels and VAD segments with
    suffixes (e.g. "_0_0", "_1_4") and converts it to an stm file that can be scored with the MeetEval toolkit.
    """

    def __init__(
        self,
        ctm_file: Path,
    ):
        """
        :param ctm_file: input ctm file
        """
        self.ctm_file = ctm_file

        self.out_stm_file = self.output_path("hyp.stm")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        ctm_file = CtmFile(self.ctm_file)
        segments = ctm_file.get_matched_channel_multi_segments()

        with open(self.out_stm_file.get(), "w") as stm_file:
            for seg in sorted(segments):
                for channel in segments[seg]:
                    name = segments[seg][channel][0].name.replace(f"_{channel}_0", "")
                    stm_content = []
                    for seg_id in segments[seg][channel]:
                        stm_line = (
                            f"{name.split('/')[-2]} 0 {channel} "
                            f"{1000 * seg_id + segments[seg][channel][seg_id].start} "
                            f"{1000 * seg_id + segments[seg][channel][seg_id].end}"
                        )
                        for word in segments[seg][channel][seg_id].words:
                            stm_line += f" {word}"
                        stm_content.append(stm_line)
                    stm_file.write("\n".join(stm_content) + "\n")


class MeetEvalJob(Job):
    """
    Run the scorer from the MeetEval toolkit (https://github.com/fgnt/meeteval).

    Outputs:
        - out_average: json file with averaged results
        - out_per_reco: json file with results per recording
    """

    def __init__(
        self,
        ref: Path,
        hyp: Path,
        meet_eval_exe: Path,
    ):
        """
        :param ref: reference stm text file
        :param hyp: hypothesis ctm text files
        :param meet_eval_root: path to python exe with meet eval installed
        """
        self.ref = ref
        self.hyp = hyp
        self.meet_eval_exe = meet_eval_exe

        self.out_report_dir = self.output_path("reports")
        self.out_average = self.output_path("reports/orcwer_average.json")
        self.out_per_reco = self.output_path("reports/orcwer_per_reco.json")

        self.out_wer = self.output_var("wer")
        self.out_num_errors = self.output_var("num_errors")
        self.out_percent_substitution = self.output_var("percent_substitution")
        self.out_num_substitution = self.output_var("num_substitution")
        self.out_percent_deletions = self.output_var("percent_deletions")
        self.out_num_deletions = self.output_var("num_deletions")
        self.out_percent_insertions = self.output_var("percent_insertions")
        self.out_num_insertions = self.output_var("num_insertions")
        self.out_ref_words = self.output_var("ref_words")
        self.out_hyp_words = self.output_var("hyp_words")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import os

        os.mkdir(self.out_report_dir.get())

        args = [
            self.meet_eval_exe.get(),
            "-m",
            "meeteval.wer",
            "orcwer",
            "-h",
            self.hyp.get(),
            "-r",
            self.ref.get(),
            "--average-out",
            self.out_average.get(),
            "--per-reco-out",
            self.out_per_reco.get(),
        ]
        print("$ " + " ".join(args))
        sp.check_call(args)

        with open(self.out_average.get()) as f:
            result = json.load(f)
        self.out_wer.set(result["error_rate"] * 100)

        self.out_num_errors.set(result["errors"])
        self.out_percent_substitution.set(result["substitutions"] / result["length"] * 100)
        self.out_num_substitution.set(result["substitutions"])
        self.out_percent_deletions.set(result["deletions"] / result["length"] * 100)
        self.out_num_deletions.set(result["deletions"])
        self.out_percent_insertions.set(result["insertions"] / result["length"] * 100)
        self.out_num_insertions.set(result["insertions"])
        self.out_ref_words.set(result["length"])
        self.out_hyp_words.set(result["length"] + result["insertions"] - result["deletions"])
