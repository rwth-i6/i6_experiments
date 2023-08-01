import glob
import pickle
import os.path as path
from typing import Any, Dict, Iterator, List, Set, Union

from sisyphus import tk, Job, Path, Task

from .phoneme_duration import compute_phoneme_durations
from .processor import AlignmentProcessor


class PlotPhonemeDurationsJob(Job):
    def __init__(
        self, alignment_bundle_path: Path, allophones_path: Path, time_step_s: float, sil_allophone: str = "[SILENCE]"
    ):
        self.alignment_bundle_path = alignment_bundle_path
        self.allophones_path = allophones_path
        self.sil_allophone = sil_allophone
        self.time_step_s = time_step_s

        self.out_plot = self.output_path("plot.png")
        self.out_sil_plot = self.output_path("plot_sil.png")

    def tasks(self) -> Iterator[Task]:
        with open(self.alignment_bundle_path, "rt") as bundle_file:
            archives = [a.strip() for a in bundle_file.readlines()]
        yield Task("compute_statistics", args=archives, rqmt={"cpu": 1, "mem": 1, "time": 10 / 60})
        yield Task("plot", rqmt={"cpu": 1, "mem": 8})

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
            if k != self.sil_allophone
        }

        ph_counts = {k: v for k, v in merged_counts.items() if k != self.sil_allophone}
        sil_counts = {k: v for k, v in merged_counts.items() if k == self.sil_allophone}

        for counts, dest in [(ph_counts, self.out_plot), (sil_counts, self.out_sil_plot)]:
            plt.clf()
            fig, ax = plt.subplots()
            ax.boxplot(counts.values(), 0, "")
            ax.set_xticklabels(counts.keys())
            fig.savefig(dest)


class PlotViterbiAlignmentsJob(Job):
    def __init__(
        self,
        alignment_bundle_path: Path,
        allophones_path: Path,
        segments: Union[Path, tk.Variable, List[str]],
        show_labels: bool,
        sil_allophone: str = "[SILENCE]",
        monophone: bool = True,
    ):
        self.alignment_bundle_path = alignment_bundle_path
        self.allophones_path = allophones_path
        self.sil_allophone = sil_allophone
        self.monophone = monophone
        self.show_labels = show_labels
        self.segments = segments

        self.out_plot_folder = self.output_path("plots", directory=True)

        if isinstance(segments, list):
            self.out_plots = [self.output_path(f"plots/{s.replace('/', '_')}.png") for s in segments]
        else:
            self.out_plots = None

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
            out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.png") for s in segments_to_plot]
        elif isinstance(self.segments, Path):
            with open(self.segments, "rt") as segments_file:
                segments_to_plot = [s.strip() for s in segments_file.readlines()]
            out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.png") for s in segments_to_plot]
        else:
            segments_to_plot = self.segments
            out_plot_files = self.out_plots

        for seg, out_path in zip(segments_to_plot, out_plot_files):
            fig, ax, *_ = processor.plot_segment(seg, self.show_labels)
            fig.savefig(out_path)
