# Makes one big results table 
import subprocess
import re
from typing import OrderedDict
import glob
import json
import logging as log
import os
import argparse
import csv

log.basicConfig(level=log.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--only', default=None) # Allow to ass multiple sep by comma
args = parser.parse_args()

RESULTS_PATH = "results2"


datasets = [
    "dev-other",
    "dev-clean",
    "test-other",
    "test-clean"
]

if args.only:
    datasets = [args.only]
    assert "dev-other" in datasets, "Need dev-other always!"


csv_columns = {
    "NAME" : [],

    "BEST epoch \nby dev-other WER" : [],

    **{f"WER ({_set})":[] for _set in datasets},

    "Train time (hours)\n(until this epoch)" : [],

    "num_params (M)" : [],
    "FULL CONFIG PATH" : []
}

all_experiments = [s.replace(f"{RESULTS_PATH}/", "") for s in glob.glob(f"{RESULTS_PATH}/*") ]
all_sub_experiments = {k : [] for k in all_experiments}

for i, ex in enumerate(all_experiments):
    all_sub_experiments[ex] = [s.split("/")[-1].replace(".json", "") for s in glob.glob(f"{RESULTS_PATH}/{ex}/*") ]

def get_best_epoch_dev_other(data, optim=False):
    set_data = data["dev-other"]
    wers = set_data["optim_wer_by_ep" if optim else "wer_by_ep"]
    log.debug(f"working with wers: {wers}")
    if isinstance(wers, str):
        return "No best ep"
    best_wer = 100
    best_ep = -1
    for x in wers:
        ep = int(x)
        wer = wers[x][-2] if optim else wers[x][0] #TODO: not sure about the no optim case
        wer = float(wer.replace("%", ""))
        if wer < best_wer:
            best_wer = wer
            best_ep = ep

    log.debug(f"Best ep {best_ep} (dev-other): {best_wer}")
    return best_ep

rows = []
rows.append(list(csv_columns.keys()))
for ex in all_experiments:
    row = [ex] + ["-"]*(len(csv_columns) -1)
    rows.append(row)
    log.debug(f"\nProcessing experiment '{ex}'\n")
    for sub_ex in all_sub_experiments[ex]:
        log.debug(f"\nsub ex '{sub_ex}'\n")
        try:
            with open(f'{RESULTS_PATH}/{ex}/{sub_ex}.json') as data_file:
                data = json.load(data_file)
        except Exception as e:
            log.debug(f"Failed parsing {sub_ex}.json, error: {e}")
            row = [sub_ex] +["parse error"] + ["-"]*(len(csv_columns) -2)
            rows.append(row)
            continue

        log.debug(data)

        if isinstance(data["dev-other"], str):
            log.debug(f"Skipping {sub_ex}, cause extraction error: {data['dev-other']}")
            row = [sub_ex] +["extract error"] + ["-"]*(len(csv_columns) -2)
            rows.append(row)
            continue

        use_optim = isinstance(data["dev-other"]["optim_wer_by_ep"], dict) and len(data["dev-other"]["optim_wer_by_ep"]) > 0
        
        best_ep_dev_other = get_best_epoch_dev_other(data, use_optim)
        log.debug(f"Best ep: {best_ep_dev_other}")

        if isinstance(best_ep_dev_other, str):
            log.debug(f"skipping {sub_ex} no good ep found")
            continue # Then we skip this

        def get_dataset_epoch(ep, _set):
            log.debug(f"looking for {ep} in {data[_set]}")
            # first try optimizes, then look for non optimzes values
            if str(best_ep_dev_other) in data[_set]["optim_wer_by_ep"]:
                log.debug("Using optimized eps")
                return float(data[_set]["optim_wer_by_ep"][str(best_ep_dev_other)][-2].replace("%",""))
            elif str(best_ep_dev_other) in data[_set]["wer_by_ep"]:
                return float(data[_set]["wer_by_ep"][str(best_ep_dev_other)][-2].replace("%",""))
            else:
                return f"no data for ep{ep}"

        config_path = data["config_path"]
        wers_per_set = [ get_dataset_epoch(best_ep_dev_other, _set) for _set in datasets ]
        log.debug(f"Found wers: {wers_per_set}")
        row = [
            data["name"], # name
            best_ep_dev_other,
            *wers_per_set,
            data["dev-other"]["time_p_sep"] * best_ep_dev_other,
            data["dev-other"]["num_params"], # params
            f"{os.getcwd()}/{config_path}" #config path
        ]
        log.debug(f"Writing row: {row}")
        rows.append(row)

with open('summary_new.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)