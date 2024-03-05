__all__ = ["AllophoneDetails", "Hyp", "TraceSource", "VisualizeBestTraceJob"]

import logging
from dataclasses import dataclass
from enum import Enum
import gzip
import numpy as np
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


@dataclass(frozen=True)
class TraceSource:
    name: str
    rasr_log: Path
    state_tying: Path
    num_tied_phonemes: int
    x_steps_per_time_step: int


class VisualizeBestTraceJob(Job):
    """This doesn't do anything helpful. The trace visualized here is mostly meaningless."""

    def __init__(
        self, sources: typing.List[TraceSource], segments: typing.List[str], allophone_detail_level: AllophoneDetails
    ):
        super().__init__()

        self.allophone_detail_level = allophone_detail_level
        self.sources = sources
        self.segments = segments

        self.out_print_files = {
            (seg, s.name): self.output_path(f"segment.{i}.{s.name}.txt")
            for i, seg in enumerate(segments)
            for s in sources
        }
        self.out_plot_files = {seg: self.output_path(f"segment.{i}.png") for i, seg in enumerate(segments)}

        self.rqmt = {"cpu": 1, "mem": 16, "time": 0.5}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import matplotlib.pyplot as plt

        tyings = [
            VisualizeBestTraceJob.parse_state_tying(src.state_tying, self.allophone_detail_level)
            for src in self.sources
        ]
        search_spaces = [VisualizeBestTraceJob.parse_search_space(src.rasr_log) for src in self.sources]

        segments_done = set()

        for segment in self.segments:
            scores_per_source = []

            for tying, search_space, source in zip(tyings, search_spaces, self.sources):
                hyps = search_space[segment]
                best_hyp = {t: min(hyps, key=lambda hyp: hyp.sc) for t, hyps in hyps.items()}

                keys = sorted(best_hyp.keys())
                best_hyps = [best_hyp[k] for k in keys]
                best_hyps_widened = [el for hyp in best_hyps for el in [hyp] * source.x_steps_per_time_step]

                pad_w = max((len(tying.get(hyp.l, "N/A")) for hyp in best_hyps)) * source.x_steps_per_time_step + (
                    source.x_steps_per_time_step - 1
                )
                label_per_hyp = [tying.get(hyp.l, "N/A").ljust(pad_w) for hyp in best_hyps]
                with open(self.out_print_files[(segment, source.name)], "wt") as file:
                    file.write("|".join(label_per_hyp))

                scores_per_source.append([float(hyp.l) / source.num_tied_phonemes for hyp in best_hyps_widened])

            min_len = min((len(a) for a in scores_per_source))
            lens_fixed = [a[:min_len] for a in scores_per_source]

            plt.clf()
            plt.imshow(lens_fixed, vmin=0, vmax=1.0, aspect="auto", interpolation="none")
            plt.savefig(self.out_plot_files[segment])

            segments_done.add(segment)

        not_processed = set(self.segments) - segments_done
        if len(not_processed) > 0:
            raise AttributeError(f"did not process all requested segments: {not_processed}")

    @classmethod
    def parse_search_space(
        cls, log_file: typing.Union[str, Path]
    ) -> typing.Dict[str, typing.Dict[int, typing.List[Hyp]]]:
        logging.info(f"loading log {log_file}")

        p = log_file.get_path() if isinstance(log_file, Path) else log_file

        if p.endswith(".gz"):
            with gzip.open(p, "rt") as f:
                xml = ET.parse(f)
        else:
            with open(p, "rt") as f:
                xml = ET.parse(f)

        root = xml.getroot()
        obj = {
            segment.attrib["full-name"]: {
                int(step.attrib["time"]): [Hyp.from_node(n) for n in step.findall("./hyp")]
                for step in segment.findall(".//step")
            }
            for segment in root.findall(".//segment")
        }

        return obj

    @classmethod
    def parse_state_tying(
        cls, path: typing.Union[str, Path], allophone_detail_level: AllophoneDetails
    ) -> typing.Dict[int, str]:
        logging.info(f"loading tying {path}")

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
