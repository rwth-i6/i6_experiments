__all__ = ["AllophoneDetails", "Hyp", "VisualizeBestTraceJob"]

import logging
from dataclasses import dataclass
from enum import Enum
import gzip
import typing
from xml.etree import ElementTree as ET

from sisyphus import Job, Path, Task


@dataclass
class Hyp:
    l: int
    pr: float
    sc: float
    st: int

    @classmethod
    def from_node(cls, n: ET.Element) -> "Hyp":
        return cls(
            l=int(n.attrib["l"]),
            pr=float(n.attrib["pr"]),
            sc=float(n.attrib["sc"]),
            st=int(n.attrib["st"]),
        )


class AllophoneDetails(Enum):
    FULL = "full"
    CONCISE = "concise"
    MONOPHONE = "mono"

    def apply(self, a: str) -> str:
        if self == AllophoneDetails.FULL:
            return a
        elif self == AllophoneDetails.CONCISE:
            return a.split("}")[0] + "}"
        elif self == AllophoneDetails.MONOPHONE:
            return a.split("{")[0]
        else:
            assert False, "unknown details"


class VisualizeBestTraceJob(Job):
    def __init__(
        self,
        rasr_logs: typing.Union[Path, typing.List[Path], typing.Dict[typing.Any, Path]],
        segments_to_process: typing.List[str],
        state_tying: Path,
        num_tied_phonemes: int,
        x_steps_per_log_time_step: int,
        allophone_detail_level: AllophoneDetails,
    ):
        super().__init__()

        self.allophone_detail_level = allophone_detail_level
        self.num_tied_phonemes = num_tied_phonemes
        self.rasr_logs = (
            rasr_logs
            if isinstance(rasr_logs, list)
            else list(rasr_logs.values())
            if isinstance(rasr_logs, dict)
            else [rasr_logs]
        )
        self.segments_to_process = set(segments_to_process)
        self.state_tying = state_tying
        self.x_steps_per_log_time_step = x_steps_per_log_time_step

        self.out_print_files = {seg: self.output_path(f"segment.{i}.txt") for i, seg in enumerate(segments_to_process)}
        self.out_plot_files = {seg: self.output_path(f"segment.{i}.png") for i, seg in enumerate(segments_to_process)}

        self.rqmt = {"cpu": 1, "mem": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt, args=range(len(self.rasr_logs)))

    def run(self, index: int):
        import matplotlib.pyplot as plt

        logging.info(f"loading state tying {self.state_tying}")
        parsed_tying = VisualizeBestTraceJob.parse_state_tying(self.state_tying, self.allophone_detail_level)

        log_file = self.rasr_logs[index]
        logging.info(f"loading search log {log_file}")
        parsed_search_space = VisualizeBestTraceJob.parse_search_space(log_file)

        segments_done = set()

        for segment, search_space in parsed_search_space.items():
            if segment not in self.segments_to_process:
                logging.info(f"{segment} not in list to process, skipping")
                continue
            else:
                logging.info(f"processing {segment}")


            best_hyps = {t: min(hyps, key=lambda hyp: hyp.sc) for t, hyps in search_space.items()}

            keys = sorted(best_hyps.keys())
            scores = [best_hyps[k] for k in keys]
            pad_w = max((len(parsed_tying.get(hyp.l, "N/A")) for hyp in scores))

            with open(self.out_print_files[segment], "wt") as f:
                for hyp in scores:
                    allophone = parsed_tying.get(hyp.l, "N/A")
                    w = pad_w * self.x_steps_per_log_time_step + (self.x_steps_per_log_time_step - 1) * 2

                    f.write(f"{allophone.ljust(w)}||")

            plt.clf()
            plt.imshow([[hyp.l for hyp in scores]], vmin=0, vmax=self.num_tied_phonemes)
            plt.savefig(self.out_plot_files[segment])

            segments_done.add(segment)

        not_processed = self.segments_to_process - segments_done
        if len(not_processed) > 0:
            raise AttributeError(f"did not process all requested segments: {not_processed}")

    @classmethod
    def parse_search_space(
        cls, log_file: typing.Union[str, Path]
    ) -> typing.Dict[str, typing.Dict[int, typing.List[Hyp]]]:
        p = log_file.get_path() if isinstance(log_file, Path) else log_file

        if p.endswith(".gz"):
            with gzip.open(p, "rt") as f:
                xml = ET.parse(f)
        else:
            with open(p, "rt") as f:
                xml = ET.parse(f)

        root = xml.getroot()
        obj = {
            segment.attrib["name"]: {
                int(step.attrib["time"]): [Hyp.from_node(n) for n in step.findall("./hyp")]
                for step in segment.findall(".//step")
            }
            for segment in root.findall(".//segment")
        }

        return obj

    @classmethod
    def parse_state_tying(
        cls,
        path: typing.Union[str, Path],
        allophone_detail_level: AllophoneDetails,
    ) -> typing.Dict[int, str]:
        tying = {}

        with open(path, "rt") as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue

                allophone, label = line.strip().split()
                label = int(label)

                if label not in tying:
                    tying[int(label)] = allophone_detail_level.apply(allophone)

        return tying
