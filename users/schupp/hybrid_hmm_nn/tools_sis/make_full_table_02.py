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
import importlib

log.basicConfig(level=log.DEBUG)


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--only', default=None) # Allow to ass multiple sep by comma

parser.add_argument("-ocn", "--overwrite-config-name", default="returnn.config")
args = parser.parse_args()

RESULTS_PATH = "results2"
CONFIG_NAME = args.overwrite_config_name

datasets = [
    "dev-other",
    "dev-clean",
    "test-other",
    "test-clean"
]

if args.only:
    datasets = [args.only]
    assert "dev-other" in datasets, "Need dev-other always!"


# TODO: things missing here:
# - devother error relation
# - lm/am ratio

csv_columns = {
    "NAME" : [],

    "Amount epochs: ": [],

    "BEST sub epoch \nby dev-other WER" : [],

    **{f"WER ({_set})":[] for _set in datasets},

    "Train time (hours)\n(until this epoch)" : [],

    "Complete train time untill final epoch" : [],

    "num_params (M)" : [],

    "devother error relation" : [],

    "GPUs used" : [],

    "avarage epoch time" : [],

    "best lm/am ratio" : [],

    "FULL CONFIG PATH" : []
}

all_experiments = [s.replace(f"{RESULTS_PATH}/", "") for s in glob.glob(f"{RESULTS_PATH}/*") ]
all_sub_experiments = {k : [] for k in all_experiments}

for i, ex in enumerate(all_experiments):
    all_sub_experiments[ex] = [s.split("/")[-1].replace(".json", "") for s in glob.glob(f"{RESULTS_PATH}/{ex}/*") ]

def get_config_data(config_path):
    spec = importlib.util.spec_from_loader('config_data', loader=None)
    config_data = importlib.util.module_from_spec(spec)

    data = None
    with open(config_path, 'r') as file:
        data = file.read()

    exec(data, config_data.__dict__)

    split = -1
    if hasattr(config_data, "epoch_split"):
        split = config_data.epoch_split
    else:
        split = config_data.train["partitionEpoch"]
    return OrderedDict(
        num_epochs = config_data.num_epochs,
        epoch_split = split,
        full_epochs = config_data.num_epochs // split # TODO: check if these params are present
    )

# Parse the run.log.1 ( contains infos such as which gpus where used)
def parse_log(log_file_path):
    used_gpus = []
    time_switched_gpu = []

    time_hh_mm_ss = r'(?:[0-5]\d):(?:[0-5]\d):(?:[0-5]\d)'
    date_jj_mm_dd = r'(?:[0-5]\d)-(?:[0-5]\d)-(?:[0-5]\d)'

    data = None
    with open(log_file_path) as file:
        data = file.read()

    lines = data.split("\n")
    for line in lines:
        if "Created TensorFlow device" in line and "tensorflow/core/common_runtime/gpu/gpu_device.cc" in line:
            log.debug(line)
            exert = line[line.index("name:") + 5:].split(",")[0]
            if used_gpus and exert == used_gpus[-1]:
                continue # Only store gpu if it changed

            used_gpus.append(exert)

            time = re.findall(time_hh_mm_ss, line)[0]
            date = re.findall(date_jj_mm_dd, line)[0]
            time_switched_gpu.append(f'{".".join(date.split("-")[1:])}-{time}')

    return OrderedDict(
        used_gpus=used_gpus,
        time_switched_gpu=time_switched_gpu,
    )


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
        config_rel_path = f"alias/conformer/{config_path.split('/conformer/')[-1]}".replace("returnn.config", CONFIG_NAME) # Needed for pings old experiments

        train_log_path = config_rel_path.replace(f"output/{CONFIG_NAME}", "log.run.1")
        log_data = parse_log(train_log_path)

        log.debug(f"rel config path: {config_rel_path}")
        config_data = get_config_data(config_rel_path)
        wers_per_set = [ get_dataset_epoch(best_ep_dev_other, _set) for _set in datasets ]
        log.debug(f"Found wers: {wers_per_set}")

        row = [
            data["name"], # name
            config_data["full_epochs"],
            best_ep_dev_other,
            *wers_per_set,
            data["dev-other"]["time_p_sep"] * best_ep_dev_other,
            config_data["num_epochs"] * data["dev-other"]["time_p_sep"],
            data["dev-other"]["num_params"], # params
            "TODO: error relation!",
            "\n" + "\n".join([f'{time}: {gpu}' for gpu, time in zip(log_data["used_gpus"], log_data["time_switched_gpu"])]), # GPU by time
            data["dev-other"]["time_p_sep"],
            "TODO: best lm/am ratio",
            f"{os.getcwd()}/{config_path}" #config path
        ]
        log.debug(f"Writing row: {row}")
        rows.append(row)

with open('summary_new.csv', 'w') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)