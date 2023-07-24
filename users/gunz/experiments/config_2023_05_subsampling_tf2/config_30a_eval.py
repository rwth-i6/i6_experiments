#!/work/asr3/raissi/shared_workspaces/gunz/miniconda3/envs/python3.10+tf2.12/bin/python3

import glob
import itertools
import os
import re
import sys

_, out_dir = sys.argv

dtl_files = glob.glob(os.path.join(out_dir, "*.wer", "sclite.dtl"))

for file in dtl_files:
    print(f"processing {file}")

    with open(file, "rt") as dtl:
        dtl_contents = " ".join(itertools.islice(dtl, 50))

        wer, total_errs = re.match(
            "Percent Total Error\s+=\s+(\d+\.\d+)%\s+\((\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()
        _, total_subst = re.match(
            "Percent Substitutions\s+=\s+(\d+\.\d+)%\s+\((\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()
        _, total_del = re.match("Percent Deletions\s+=\s+(\d+\.\d+)%\s+\((\d+)\)", dtl_contents, re.IGNORECASE).groups()
        _, total_insrt = re.match(
            "Percent Insertions\s+=\s+(\d+\.\d+)%\s+\((\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()

        tdpScale, spLoop, spFwd, spExit, silLoop, silFwd, silExit = re.match(
            "Beam\d+-Lm\d+\.\d+-Pron\d+\.\d+-prC\d+\.\d+-tdpScale-(\d+\.\d+)-spTdp-(\d+),(\d+),inf,(\d+)-silTdp-(\d+),(\d+),inf,(\d+)-tdpNWex",
            os.path.dirname(file),
        ).groups()

        print(f"{tdpScale};{spLoop};{spFwd};{spExit};{silLoop};{silFwd};{silExit};{wer};{total_errs};{total_subst};{total_del};{total_insrt}")
