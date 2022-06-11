import argparse
import glob
import logging as log
import subprocess

log.basicConfig(level=log.INFO)

PY = '/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8'

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment", default=None)
parser.add_argument("-extra", "--extra-code", default=None)

args = parser.parse_args()

all_exps = [ x.replace("alias/conformer/", "") for x in glob.glob(f"alias/conformer/{args.experiment}/*") if not "recog_" in x ]

for ex in all_exps:
    log.info(f"Extracting: {ex}")
    _call = f"{PY} tools_linked/results.py -c {ex}".split(" ")

    if args.extra_code:
        _call += args.extra_code.split(" ")
        log.info(f"using extra: {args.extra_code}")

    out = ""
    try:
        out = subprocess.check_output(_call).decode('UTF-8')
    except Exception as e:
        log.info(f"Failed: {e}")

    log.debug(out)