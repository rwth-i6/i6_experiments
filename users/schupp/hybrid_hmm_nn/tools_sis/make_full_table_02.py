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

log.basicConfig(level=log.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--only', default=None) # Allow to ass multiple sep by comma
parser.add_argument('-a', '--append', action="store_true") # Append to csv instead of overwriting

parser.add_argument("-ocn", "--overwrite-config-name", default="returnn.config")
parser.add_argument("-ox", "--only-experiment", default=None)
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

    "finished" : [],

    "Amount epochs: ": [],

    "BEST sub epoch \nby dev-other WER" : [],

    **{f"WER ({_set})":[] for _set in datasets},

    "Train time (hours)\n(until this epoch)" : [],

    "Train time untill end (h)" : [],

    "Average time full epoch": [],

    "num params (M)" : [],

    "dev/devtrain CE (final ep)" : [],
    "dev/devtrain FER (final ep)" : [],
    "devtrain WER" : [],
    "dev/devtrain WER (final ep)" : [],

    "GPUs used" : [],

    "avarage epoch time" : [],

    "best lm/am ratio" : [],

    "FULL CONFIG PATH" : []
}

all_experiments = [s.replace(f"{RESULTS_PATH}/", "") for s in glob.glob(f"{RESULTS_PATH}/*") ]
if args.only_experiment:
    all_experiments = [args.only_experiment]

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

all_log_datas = {}
rows = []
rows.append(list(csv_columns.keys()))
for ex in all_experiments:
    row = [ex] + ["-"]*(len(csv_columns) -1)
    rows.append(row)
    log.debug(f"\nProcessing experiment '{ex}'\n")
    for sub_ex in all_sub_experiments[ex]:
        log.info(f"\nsub ex '{sub_ex}'\n")
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

        def maybe_get_error_score_by_ep(name, epoch):
            if data["dev-other"]["errors_per_ep"] and str(epoch) in data["dev-other"]["errors_per_ep"] and \
                name in data["dev-other"]["errors_per_ep"][str(epoch)]:
                return data["dev-other"]["errors_per_ep"][str(epoch)][name]
            else:
                return None

        def get_key(_set="dev", _for="score"):
            if not data["dev-other"]["errors_per_ep"]:
                return None # data non existent
            elif not "1" in data["dev-other"]["errors_per_ep"]:
                return None
            keys = list(data["dev-other"]["errors_per_ep"]["1"].keys())
            if f"{_set}_{_for}_output" in keys:
                return f"{_set}_{_for}_output"
            elif f"{_set}_{_for}" in keys:
                return f"{_set}_{_for}"

            #log.info(keys)
            #log.info(data["dev-other"]["errors_per_ep"])
            return None
            #assert False, "unknown key strucutre"

        

        #log.info(data["dev-other"]["errors_per_ep"])
        devtrain_score_key = get_key("devtrain", "score")
        log.info(devtrain_score_key)

        devtrain_score_error_final = [
            maybe_get_error_score_by_ep(devtrain_score_key, config_data["num_epochs"]),
            maybe_get_error_score_by_ep(get_key("devtrain", "error"), config_data["num_epochs"]),
        ]

        #log.info(devtrain_score_error_final)

        dev_score_error_final = [
            maybe_get_error_score_by_ep(get_key("dev", "score"), config_data["num_epochs"]),
            maybe_get_error_score_by_ep(get_key("dev", "error"), config_data["num_epochs"]),
        ]

        CE_error_ration = "no data"
        if devtrain_score_error_final[0] and dev_score_error_final[0]:
            CE_error_ration = devtrain_score_error_final[0]/dev_score_error_final[0]
            if CE_error_ration > 1:
                CE_error_ration = "extract error"
            else:
                CE_error_ration = round(CE_error_ration, 4)

        FER_error_ration = "no data"
        if devtrain_score_error_final[1] and dev_score_error_final[1]:
            FER_error_ration = devtrain_score_error_final[1]/dev_score_error_final[1]
            if FER_error_ration > 1:
                FER_error_ration = "extract error"
            else:
                FER_error_ration = round(FER_error_ration, 4)

        finished_train = "YES" if data["dev-other"]["finished_eps"] and int(data["dev-other"]["finished_eps"]) == int(config_data["num_epochs"]) else "NO"

        devtrain_WER = f"No data for ep {config_data['num_epochs']}"
        if "devtrain" in data and str(config_data["num_epochs"]) in data['devtrain']:
            log.info("Found devtrain WER")
            devtrain_WER = data['devtrain'][str(config_data["num_epochs"])]

        devWER_final = None
        if "dev-other" in data and str(config_data["num_epochs"]) in data['dev-other']['optim_wer_by_ep']:
            devWER_final = float(data['dev-other']["optim_wer_by_ep"][str(config_data["num_epochs"])][-2][:-1])

        dev_devtrain_WER_ration = f"No data for ep {config_data['num_epochs']}"
        if isinstance(devWER_final, float) and isinstance(devtrain_WER, float):
            dev_devtrain_WER_ration = devWER_final / devtrain_WER

        # Write the ex log_datas to another file that can be analyzes

        all_log_datas[f"{ex}/{sub_ex}"] = log_data

        row = [
            data["name"], # name
            finished_train,
            config_data["full_epochs"],
            best_ep_dev_other,
            *wers_per_set,
            data["dev-other"]["time_p_sep"] * best_ep_dev_other,
            config_data["num_epochs"] * data["dev-other"]["time_p_sep"],
            config_data["epoch_split"] * data["dev-other"]["time_p_sep"], # Time for a full epoch
            data["dev-other"]["num_params"], # params
            CE_error_ration,
            FER_error_ration,
            devtrain_WER,
            dev_devtrain_WER_ration,
            "\n" + "\n".join([f'{time}: {gpu}' for gpu, time in zip(log_data["used_gpus"], log_data["time_switched_gpu"])]), # GPU by time
            data["dev-other"]["time_p_sep"],
            "TODO: best lm/am ratio",
            f"{os.getcwd()}/{config_path}" #config path
        ]
        log.info(f"Writing row: {row}")
        rows.append(row)

with open('summary_new.csv', 'w' if not args.append else "a") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

old_log_data = {}
with open('log.datas.experiments.json', "r") as file:
    old_log_data = json.load(file)

old_log_data.update(all_log_datas)

with open('log.datas.experiments.json', "w") as file:
    json.dump(old_log_data, file, indent=1)