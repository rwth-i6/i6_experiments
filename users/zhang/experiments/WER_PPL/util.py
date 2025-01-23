import glob
import numpy as np
import pickle
import os
import os.path as path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from sisyphus import tk, Job, Path, Task
import json

class WER_ppl_PlotAndSummaryJob(Job):
    # Summary the recog experiment with LMs on various ppl. Applying linear regression
    # Assumed input: A list of (ppl,WER) pairs
    # Output: A summaryfile with fitted equation, a plot image,
    __sis_hash_exclude__ = {"names": None}

    def __init__(
        self,
        names: List[str],
        results: List[Tuple[tk.Variable, tk.Path]],
        # Reserved for plot setting
    ):
        self.out_summary_json = self.output_path("summary.json")
        #self.out_plot_folder = self.output_path("plots", directory=True)
        self.names = names
        self.results = results

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True, rqmt={"cpu": 1, "time": 1, "mem": 4})

    def run(self):
        ppls = list()
        wers = list()
        for ppl_val, wer_path in self.results:
            ppls.append(ppl_val.get())
            with open(wer_path.get_path(), "r") as f:
                wers.append(json.load(f))
        res = dict(zip(self.names, zip(ppls,wers)))
        with open(self.out_summary_json.get_path(), "w") as f:
            f.write(json.dumps(res))
            f.write("\n")
        # from matplotlib import pyplot as plt
        #
        # processor = AlignmentProcessor(
        #     alignment_bundle_path=self.alignment_bundle_path.get_path(),
        #     allophones_path=self.allophones_path.get_path(),
        #     sil_allophone=self.sil_allophone,
        #     monophone=self.monophone,
        # )
        #
        # if isinstance(self.segments, tk.Variable):
        #     segments_to_plot = self.segments.get()
        #     assert isinstance(segments_to_plot, list)
        #     out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments_to_plot]
        # elif isinstance(self.segments, Path):
        #     with open(self.segments, "rt") as segments_file:
        #         segments_to_plot = [s.strip() for s in segments_file.readlines()]
        #     out_plot_files = [self.output_path(f"plots/{s.replace('/', '_')}.pdf") for s in segments_to_plot]
        # else:
        #     segments_to_plot = self.segments
        #     out_plot_files = self.out_plots
        #
        # plt.rc("font", family="serif")
        #
        # for seg, out_path in zip(segments_to_plot, out_plot_files):
        #     fig, ax, *_ = processor.plot_segment(
        #         seg, font_size=self.font_size, show_labels=self.show_labels, show_title=self.show_title
        #     )
        #     if self.show_title:
        #         fig.savefig(out_path, transparent=True)
        #     else:
        #         fig.savefig(out_path, transparent=True, bbox_inches="tight", pad_inches=0)
