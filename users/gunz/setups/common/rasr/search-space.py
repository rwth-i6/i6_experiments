__all__ = ["AllophoneDetails", "Hyp", "PlotBestScore"]

from dataclasses import dataclass
from enum import Enum
import gzip
import math
import typing
from xml.etree import ElementTree as ET

from sisyphus import Job, Path


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

    def apply(self, a: str) -> str:
        if self == AllophoneDetails.FULL:
            return a
        elif self == AllophoneDetails.CONCISE:
            return a.split("}")[0] + "}"
        else:
            assert False, "unknown details"


PAD_TO = 20


class DrawBestScore(Job):
    def __init__(
        rasr_logs: typing.Union[Path, typing.List[Path], typing.Dict[typing.Any, Path]],
        state_tying: Path,
        x_steps_per_log_time_step: int = 1,
        allophone_detail_level: AllophoneDetails = AllophoneDetails.CONCISE,
    ):
        super().__init__()

        self.allophone_detail_level = allophone_detail_level
        self.rasr_logs = (
            rasr_logs
            if isinstance(rasr_logs, list)
            else list(rasr_logs.values())
            if isinstance(rasr_logs, dict)
            else [rasr_logs]
        )
        self.state_tying = state_tying
        self.x_steps_per_log_time_step = x_steps_per_log_time_step

        self.rqmt = {"cpu": 1, "mem": 4}

    def tasks(self):
        yield Task("run", args=self.rasr_logs)

    def run(self, log: Path):
        tying = self.parse_state_tying(self.state_tying)
        parsed = self.parse_search_space(log)

        for i, (segment, search_space) in enumerate(parsed.items()):
            best_hyps = {t: min(hyps, key=lambda hyp: math.exp(-hyp.sc)) for t, hyps in search_space.items()}

            keys = sorted(best_hyps.keys())
            scores = [best_hyps[k] for k in keys]

            with open(f"segment.{i}", "wt") as f:
                for hyp in scores:
                    allophone = tying[hyp.l]
                    w = PAD_TO * self.x_steps_per_log_time_step

                    f.write(f"{allophone.ljust(w)}|")

    def parse_search_space(self, log: Path) -> typing.Dict[str, typing.Dict[int, typing.List[Hyp]]]:
        if log.get_path().endswith(".gz"):
            with gzip.open(log, "rt") as f:
                xml = ET.parse(f)
        else:
            with open(log, "rt") as f:
                xml = ET.parse(f)

        root = xml.getroot()
        obj = {
            segment.attrib["name"]: {
                int(step.attrib["time"]): [Hyp.from_node(n) for n in step.findall("./hyp")]
                for step in segment.findall("./step")
            }
            for segment in root.findall(".//segment")
        }

        return obj

    def parse_state_tying(self) -> typing.Dict[int, str]:
        tying = {}

        with open(self.state_tying, "rt") as f:
            for line in f:
                if line.strip().startswith("#"):
                    continue

                allophone, label = line.strip().split()
                tying[int(label)] = self.allophone_detail_level.apply(allophone)

        return tying
