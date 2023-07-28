from typing import List, Union

from sisyphus import tk, Job, Path

from .processor import AlignmentProcessor


class PlotPhonemeDurationsJob(Job):
    def __init__(
        self,
        alignment_bundle_path: Path,
        allophones_path: Path,
        show_labels: bool,
        sil_allophone: str = "[SILENCE]",
        monophone: bool = True,
    ):
        self.alignment_bundle_path = alignment_bundle_path
        self.allophones_path = allophones_path
        self.sil_allophone = sil_allophone
        self.monophone = monophone
        self.show_labels = show_labels

        self.out_plot_folder = self.output_path("plots", directory=True)

    def run(self):
        raise NotImplementedError()

        processor = AlignmentProcessor(
            alignment_bundle_path=self.alignment_bundle_path.get_path(),
            allophones_path=self.allophones_path.get_path(),
            sil_allophone=self.sil_allophone,
            monophone=self.monophone,
        )


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
