# Load all data: wers, error, lr's, configs

import subprocess
import re
from typing import OrderedDict
import glob
import json
import logging as log
import os
import argparse
import importlib
from numpy import nan # we need this in case there are any 'nan' in errors or anything ( for when we call eval() )

log.basicConfig(level=log.INFO)

# Allowed arguments: (WIP)
# --basepath conformer      -> the root path alias/*/
# --base-experiment         -> the path alias/--basepath/*
# --only dev-other          -> only one dataset
# --update-data-filter   "dev-other:filter-new"
# --filter-ex-name "dummy_job"         -> filters all these names out

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default=None)
parser.add_argument('-b', '--basepath', default='conformer')
parser.add_argument('-o', '--only', default=None)
parser.add_argument('-w', '--write', action="store_true")
parser.add_argument('-udf', '--update-data-filter', default=None)

args = parser.parse_args()

BASE = args.basepath

print(args)
# We allow also a '/' instead of ':'

args.config = args.config.replace("/", ":")
print(args.config)

data_set_prefixes = {
    "dev-other" : "", # TODO: other default?
}

if args.only:
    data_set_prefixes = { args.only : data_set_prefixes[args.only]}

if args.update_data_filter:
    log.debug(f"updating filter: {args.update_data_filter}")
    data, new_filt = args.update_data_filter.split(":")
    if data in data_set_prefixes:
        data_set_prefixes[data] = new_filt
        log.debug(f"updated: {data_set_prefixes}")

datasets = list(data_set_prefixes.keys())

log.debug(f"Datsets to consider: {datasets}")

PY="/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"

RESULTS_FOLDER = "results2/"


def get_summary_string_for_dataset(prefix, name, data):
    exe = "get_wer" if data_set_prefixes[data] == "" else "get_wer_for_set"
    extra = "" if data_set_prefixes[data] == "" else f" --set {data}"
    call = f"{PY} {exe}.py --prefix {prefix} --job_name {name} --print true{extra}"
    log.debug(call)
    _call = call.split(" ") # No spaces in names allowed

    try:
        out = subprocess.check_output(_call).decode('UTF-8')
    except Exception as e:
        return e # Just return the exeption handle it later
    return out

def parse_experiment_out_string(data_string):
    num_params = None
    avg_s_per_sep = None
    time_p_sep = None
    wer_by_ep = None
    optim_wer_by_ep = None
    best_ep_by_score = None
    errors_per_ep = None
    finished_eps = None

    float_or_int = r'[\d\.\d]+'

    def try_or_default(run, default): # Againt all rules but shoretest solution here
        try:
            return run()
        except Exception as e:
            log.debug(e)
            return default

    for line in data_string.split("\n"):
        if "num_params:" in line:
            num_params = try_or_default(lambda : float(re.findall(float_or_int, line)[0]), "not found")
        elif "average steps per subepoch" in line:
            avg_s_per_sep = try_or_default(lambda : float(re.findall(float_or_int, line)[0]), "not found")
        elif "training time per subepoch" in line and not time_p_sep:
            time_p_sep = try_or_default(lambda : float(re.findall(float_or_int, line)[0]), "not found")
        elif "recog:" in line:
            try:
                wer_by_ep = eval(line.replace("recog:", ""))
            except SyntaxError as e:
                wer_by_ep = "No wers found"
        elif "optimized am lm scales:" in line:
            try:
                optim_wer_by_ep = eval(line.replace("optimized am lm scales:", ""))
            except SyntaxError as e:
                optim_wer_by_ep = "No optim wers found"
        elif "best epoch:" in line:
            best_ep_by_score = try_or_default(lambda : int(re.findall(float_or_int, line)[0]), "not found")
        elif "errors:" in line:
            log.debug(line)
            try:
                errors_per_ep = eval(line.replace("errors:", ""))
            except SyntaxError as e:
                errors_per_ep = "No errors, sores found"
        elif "finished epoch:" in line:
            finished_eps = try_or_default(lambda : float(re.findall(float_or_int, line)[0]), "not found")

    return OrderedDict(
        num_params = num_params,
        avg_s_per_sep = avg_s_per_sep,
        time_p_sep = time_p_sep,
        wer_by_ep = wer_by_ep,
        optim_wer_by_ep = optim_wer_by_ep,
        best_ep_by_score = best_ep_by_score,
        errors_per_ep = errors_per_ep,
        finished_eps = finished_eps
    )


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

def print_summary(ex_data, config_data, log_data=None):
    log.debug(list(ex_data.keys()))

    info = []
    wers = "no data"
    if not isinstance(ex_data['wer_by_ep'], str):
        wers = ex_data['wer_by_ep']
    elif not isinstance(ex_data['optim_wer_by_ep'], str):
        wers = { k : ex_data["optim_wer_by_ep"][k][-2] for k in ex_data['optim_wer_by_ep'] }

    cur_error = "not found"
    if ex_data['finished_eps'] and int(ex_data['finished_eps']) in ex_data['errors_per_ep']:
        cur_error = ex_data['errors_per_ep'][int(ex_data['finished_eps'])]
        cur_error = {k: cur_error[k] for k in cur_error if k in ["train_error_output", "train_score_output"]}

    info += [
        f"progress: seps {ex_data['finished_eps']}/{config_data['num_epochs']}",
        f"time per sep: {ex_data['time_p_sep']}",
        f"current error: {cur_error}"
    ]

    if log_data:
        info += [
            "gpus used: \n" + "\n".join([f'{time}: {gpu}' for gpu, time in zip(log_data["used_gpus"], log_data["time_switched_gpu"])])
        ]

    info += [
        f"params (m): {ex_data['num_params']}",
        f"wers: {wers}"
    ]
        #f"time to cur ep: {ex_data['time_p_sep'] * ex_data['finished_eps']}" error prone, have better checks
    log.info("\n" + "\n".join(info))


def get_all_data_experiment(experiment_path, name):
    data_by_set = {}
    for data in datasets: 
        log.debug(f"Extracting: {data}")
        _exp_data = get_summary_string_for_dataset(experiment_path, name, data)
        if isinstance(_exp_data, Exception):
            log.debug(f"Extraction error {_exp_data}")
            data_by_set[data] = f"Extraction error {_exp_data}"
        else:
            data_by_set[data] = parse_experiment_out_string(_exp_data)
        #log.debug(data_by_set[data])
    return {
        "name" : name,
        "config_path" : f"{os.getcwd()}/{experiment_path}/{name}/train.job/output/returnn.config",
        **data_by_set
    }

ex_path, sub_experiment = args.config.split(":")

experiment = f"{BASE}/{ex_path}/"

ex_path = experiment.replace(f"{BASE}/","")

if not os.path.exists(f"{RESULTS_FOLDER}{ex_path}"):
    os.mkdir(f"{RESULTS_FOLDER}{ex_path}")

log.debug(f"Starting extraction of: {sub_experiment}")
ex_data = get_all_data_experiment(experiment, sub_experiment)

if args.write:
    with open(f"{RESULTS_FOLDER}{ex_path}/{sub_experiment}.json", "w") as file:
        json.dump(
            ex_data,
            file,
            indent=1
        )

config_path = f"alias/{BASE}/{ex_path}{sub_experiment}/train.job/output/returnn.config"
conf_data = get_config_data(config_path)

log_file_path = f"alias/{BASE}/{ex_path}{sub_experiment}/train.job/log.run.1" # Maybe should use work/returnn.log instead?
log_data = parse_log(log_file_path)
log.debug(f"Got log data: {log_data}")

log.debug(ex_data)
print_summary(ex_data["dev-other"], conf_data, log_data)

