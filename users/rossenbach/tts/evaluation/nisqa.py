from sisyphus import Job, Task, gs, tk

import numpy as np
import os
import subprocess
from typing import Optional

from i6_core.lib.corpus import Corpus


class NISQAMosPredictionJob(Job):
    """
    Perform MOS prediction with NISQA pretrained model on each recording of a bliss file
    """

    def __init__(self, bliss_corpus: tk.Path, nisqa_repo: tk.Path, nisqa_exe: Optional[tk.Path] = None):
        """

        """
        self.bliss_corpus = bliss_corpus
        self.nisqa_repo = nisqa_repo
        self.nisqa_exe = nisqa_exe or gs.SIS_COMMAND[0]

        self.output_dir = self.output_path("output_dir", directory=True)
        self.out_mos_average = self.output_var("mos_average")
        self.out_mos_min = self.output_var("mos_min")
        self.out_mos_max = self.output_var("mos_max")
        self.out_mos_std_dev = self.output_var("mos_std_dev")

        self.rqmt = {"mem": 8, "cpu": 8, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        c = Corpus()
        tmp_csv = open("in.csv", "wt")
        c.load(self.bliss_corpus.get_path())
        corpus_base_path = os.path.dirname(self.bliss_corpus.get_path())
        tmp_csv.write("filename,\n")
        for recording in c.all_recordings():
            if recording.audio.startswith("/"):
                tmp_csv.write(f"{recording.audio},\n")
            else:
                tmp_csv.write(f"{os.path.join(corpus_base_path, recording.audio)},\n")
        tmp_csv.close()

        in_file = os.path.realpath("in.csv")
        print("Infile: %s" % in_file)
        subprocess.check_call([
            self.nisqa_exe,
            "run_predict.py",
            "--mode", "predict_csv",
            "--pretrained_model", "weights/nisqa_tts.tar",
            "--csv_file", in_file,
            "--csv_deg", "filename",
            "--output_dir", self.output_dir.get_path(),
            "--num_workers", "0",
            "--bs", "10"
        ], cwd=self.nisqa_repo.get_path())

        mos_csv = open(os.path.join(self.output_dir, "NISQA_results.csv"), "rt")
        mos_values = []
        for i, line in enumerate(mos_csv.readlines()):
            if i == 0:
                # skip header line
                continue
            _, _, mos, _ = line.strip().split(",")
            mos_values.append(float(mos))

        self.out_mos_average.set(np.average(mos_values))
        self.out_mos_min.set(np.min(mos_values))
        self.out_mos_max.set(np.max(mos_values))
        self.out_mos_std_dev.set(np.std(mos_values))
