import glob
import numpy as np
import pickle
import os
import os.path as path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from sisyphus import tk, Job, Path, Task

from .phoneme_duration import compute_phoneme_durations
from .processor import AlignmentProcessor


DEFAULT_GROUPS = {
    "consonants": [
        "B",
        "CH",
        "D",
        "DH",
        "ER",
        "F",
        "G",
        "HH",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "V",
        "W",
        "Z",
        "ZH",
    ],
    "vowels": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "EY", "IH", "IY", "OW", "OY", "UH", "UW", "Y"],
}


class PlotPhonemeDurationsJob(Job):
    def __init__(
        self,
        alignment_bundle_path: Path,
        allophones_path: Path,
        time_step_s: float,
        stat_groups: Optional[Dict[str, List[str]]] = None,
        sil_allophone: str = "[SILENCE]",
        figsize: Tuple[int, int] = (20, 10),
    ):
        self.alignment_bundle_path = alignment_bundle_path
        self.allophones_path = allophones_path
        self.figsize = figsize
        self.sil_allophone = sil_allophone
        self.stat_groups = stat_groups or DEFAULT_GROUPS
        self.time_step_s = time_step_s

        self.out_plot_folder = self.output_path("plots", directory=True)
        self.out_means = self.output_var("means")
        self.out_vars = self.output_var("vars")

    def tasks(self) -> Iterator[Task]:
        with open(self.alignment_bundle_path, "rt") as bundle_file:
            archives = [a.strip() for a in bundle_file.readlines()]
        yield Task("compute_statistics", args=archives, rqmt={"cpu": 1, "mem": 1, "time": 10 / 60})
        yield Task("plot", rqmt={"cpu": 1, "mem": 8}, mini_task=True)
        yield Task("cleanup", mini_task=True)

    def compute_statistics(self, cache_file: str):
        durations = compute_phoneme_durations(cache_file=cache_file, allophones=self.allophones_path.get_path())
        with open(f"stats.{path.basename(cache_file)}.pk", "wb") as f:
            pickle.dump(durations, f)

    def plot(self):
        import matplotlib.pyplot as plt

        def unpickle(file_name: str) -> Any:
            with open(file_name, "rb") as f:
                return pickle.load(f)

        counts = glob.glob("stats.*.pk")
        loaded_counts: List[Dict[str, List[int]]] = [unpickle(f) for f in counts]
        all_phonemes: Set[str] = {ph for counts in loaded_counts for ph in counts.keys()}
        merged_counts: Dict[str, List[float]] = {
            k: [count * self.time_step_s for counts in loaded_counts for count in counts[k]]
            for k in sorted(all_phonemes)
        }

        ph_counts = {k: v for k, v in merged_counts.items() if k != self.sil_allophone}
        sil_counts = {k: v for k, v in merged_counts.items() if k == self.sil_allophone}
        to_plot = [
            (ph_counts, self.out_plot_folder.join_right("phonemes.pdf")),
            (sil_counts, self.out_plot_folder.join_right("sil.pdf")),
        ]

        for group, phonemes in self.stat_groups.items():
            joined_stats = {group: [count for ph in phonemes for count in merged_counts[ph]]}
            to_plot.append((joined_stats, self.out_plot_folder.join_right(f"{group}.pdf")))

        for counts, dest in to_plot:
            plt.clf()
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.boxplot(counts.values(), 0, "", whis=[5, 95])
            ax.set_xticklabels(counts.keys())
            ax.set_xlabel("Phoneme")
            ax.set_ylabel("Duration [s]")
            ax.set_ylim(bottom=0)
            fig.savefig(dest, bbox_inches="tight", transparent=True)

        means = {k: np.mean(v) for k, v in merged_counts.items()}
        means["PHONEMES"] = np.mean(
            [v for ph, counts in merged_counts.items() if ph != self.sil_allophone for v in counts]
        )

        variances = {k: np.var(v) for k, v in merged_counts.items()}
        variances["PHONEMES"] = np.var(
            [v for ph, counts in merged_counts.items() if ph != self.sil_allophone for v in counts]
        )

        self.out_means.set(means)
        self.out_vars.set(variances)

    def cleanup(self):
        files = glob.glob("stats.*.pk")
        for file in files:
            os.remove(file)


class PlotViterbiAlignmentsJob(Job):
    __sis_hash_exclude__ = {"font_size": 10, "show_title": True}

    def __init__(
        self,
        alignment_bundle_path: Path,
        allophones_path: Path,
        segments: Union[Path, tk.Variable, List[str]],
        show_labels: bool,
        show_title: bool = True,
        sil_allophone: str = "[SILENCE]",
        font_size: int = 10,
        monophone: bool = True,
    ):
        self.alignment_bundle_path = alignment_bundle_path
        self.allophones_path = allophones_path
        self.sil_allophone = sil_allophone
        self.monophone = monophone
        self.font_size = font_size
        self.show_labels = show_labels
        self.show_title = show_title
        self.segments = segments

        self.out_plot_folder = self.output_path("plots", directory=True)

        if isinstance(segments, list):
            self.out_plots = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments]
        else:
            self.out_plots = None

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 1, "time": 1, "mem": 4})

    def run(self):
        processor = AlignmentProcessor(
            alignment_bundle_path=self.alignment_bundle_path.get_path(),
            allophones_path=self.allophones_path.get_path(),
            sil_allophone=self.sil_allophone,
            monophone=self.monophone,
        )

        if isinstance(self.segments, tk.Variable):
            segments_to_plot = self.segments.get()
            assert isinstance(segments_to_plot, list)
            out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments_to_plot]
        elif isinstance(self.segments, Path):
            with open(self.segments, "rt") as segments_file:
                segments_to_plot = [s.strip() for s in segments_file.readlines()]
            out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments_to_plot]
        else:
            segments_to_plot = self.segments
            out_plot_files = self.out_plots

        for seg, out_path in zip(segments_to_plot, out_plot_files):
            fig, ax, *_ = processor.plot_segment(
                seg, font_size=self.font_size, show_labels=self.show_labels, show_title=self.show_title
            )
            if self.show_title:
                fig.savefig(out_path, transparent=True)
            else:
                fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
