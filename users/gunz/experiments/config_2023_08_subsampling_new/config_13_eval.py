#!/work/asr3/raissi/shared_workspaces/gunz/miniconda3/envs/python3.10+tf2.12/bin/python3

import glob
import itertools
import os
import re
import sys


def process(file: str):
    with open(file, "rt") as dtl:
        dtl_contents = " ".join(itertools.islice(dtl, 50))

        wer, total_errs = re.search(
            r"Percent Total Error\s+=\s+(\d+\.\d+)%\s+\(\s*(\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()
        _, total_subst = re.search(
            "Percent Substitution\s+=\s+(\d+\.\d+)%\s+\(\s*(\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()
        _, total_del = re.search(
            "Percent Deletions\s+=\s+(\d+\.\d+)%\s+\(\s*(\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()
        _, total_insrt = re.search(
            "Percent Insertions\s+=\s+(\d+\.\d+)%\s+\(\s*(\d+)\)", dtl_contents, re.IGNORECASE
        ).groups()

        tdpScale, spLoop, spFwd, spExit, silLoop, silFwd, silExit = re.search(
            "Beam\d+-Lm\d+\.\d+-Pron\d+\.\d+-prC\d+\.\d+-tdpScale-(\d+\.\d+)-spTdp-(\d+|\d+\.\d+),(\d+|\d+\.\d+),inf,(\d+|\d+\.\d+)-silTdp-(\d+|\d+\.\d+),(\d+|\d+\.\d+),inf,(\d+|\d+\.\d+)-tdpNWex",
            os.path.dirname(file),
        ).groups()

        print(
            f"{tdpScale};{spLoop};{spFwd};{spExit};{silLoop};{silFwd};{silExit};{wer};{total_errs};{total_subst};{total_del};{total_insrt}".replace(
                ".", ","
            )
        )


def main():
    _, out_dir = sys.argv

    dtl_files = glob.glob(os.path.join(out_dir, "*.wer", "sclite.dtl"))

    for file in dtl_files:
        try:
            process(file)
        except Exception as e:
            print(f"failed on {file}")
            raise e


if __name__ == "__main__":
    main()
