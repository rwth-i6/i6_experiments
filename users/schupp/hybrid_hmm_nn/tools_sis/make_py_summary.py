# Load all data: wers, error, lr's, configs

import subprocess
import re
from typing import OrderedDict
import glob
import json
import logging as log
import os
import argparse

from black import filter_cached

log.basicConfig(level=log.DEBUG)

# Allowed arguments: (WIP)
# --basepath conformer      -> the root path alias/*/
# --base-experiment         -> the path alias/--basepath/*
# --only dev-other          -> only one dataset
# --update-data-filter   "dev-other:filter-new"
# --filter-ex-name "dummy_job"         -> filters all these names out

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--basepath', default='conformer')
parser.add_argument('-bx', '--base-experiment', default=None)
parser.add_argument('-o', '--only', default=None)
parser.add_argument('-fex', '--filter-ex-name', default=None)
parser.add_argument('-udf', '--update-data-filter', default=None)
args = parser.parse_args()

BASE = args.basepath

print(args)


data_set_prefixes = {
    "dev-other" : "_",
    "dev-clean" : "_dev-clean",
    "train-other" : "_train-other",
    "train-clean" : "_tain-clean"
}

if args.only:
    data_set_prefixes = { args.only : data_set_prefixes[args.only]}

if args.update_data_filter:
    data, new_filt = args.update_data_filter.split(":")
    if data in data_set_prefixes:
        data_set_prefixes[data] = new_filt

datasets = list(data_set_prefixes.keys())

log.debug(f"Datsets to consider: {datasets}")

all_existing_experiments = [s.replace("alias/", "") for s in glob.glob(f"alias/{BASE}/*") ]
if args.base_experiment:
    all_existing_experiments = [f"{BASE}/{args.base_experiment}"] # With this you can filter for a specific setup
sub_experiments = {k : [] for k in all_existing_experiments}

filter_check = lambda s : (not "recog_" in s)
if args.filter_ex_name:
    filters = ["recog_", *args.filter_ex_name.split(",") ]
    log.debug(f"Using updated filters: {filters}")
    filter_check = lambda s : all([x not in s for x in filters])

for i, k in enumerate(all_existing_experiments):
    sub_experiments[k] = [s.split("/")[-1] for s in glob.glob(f"alias/{all_existing_experiments[i]}/*") if filter_check(s) ]

log.debug(all_existing_experiments)


PY="/work/tools/asr/python/3.8.0_tf_2.3-v1-generic+cuda10.1/bin/python3.8"

RESULTS_FOLDER = "results2/"


def get_summary_string_for_dataset(prefix, name, data):
    exe = "get_wer" if data_set_prefixes[data] == "" else "get_wer_for_set"
    extra = "" if data_set_prefixes[data] == "" else f" --set {data}"
    call = f"{PY} {exe}.py --prefix {prefix} --job_name {name} --print true{extra}"
    log.debug(call)
    _call = call.split(" ") # No spaces in names allowed

    out = subprocess.check_output(_call).decode('UTF-8')
    return out

def parse_experiment_out_string(data_string):
    num_params = None
    avg_s_per_sep = None
    time_p_sep = None
    wer_by_ep = None
    optim_wer_by_ep = None
    best_ep_by_score = None
    errors_per_ep = None

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
            log.debug(line)
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
            try:
                errors_per_ep = eval(line.replace("errors:", ""))
            except SyntaxError as e:
                errors_per_ep = "No errors, sores found"

    return OrderedDict(
        num_params = num_params,
        avg_s_per_sep = avg_s_per_sep,
        time_p_sep = time_p_sep,
        wer_by_ep = wer_by_ep,
        optim_wer_by_ep = optim_wer_by_ep,
        best_ep_by_score = best_ep_by_score
    )


def get_all_data_experiment(experiment_path, name):
    data_by_set = {}
    for data in datasets: 
        log.debug(f"Extracting: {data}")
        data_by_set[data] = parse_experiment_out_string(get_summary_string_for_dataset(experiment_path, name, data))
        log.debug(data_by_set[data])
    return {
        "name" : name,
        "config_path" : f"alias/{experiment_path}/{name}/train.job/output/returnn.config",
        **data_by_set
    }

for experiment in all_existing_experiments:
    for sub_experiment in sub_experiments[experiment]:
        ex_path = experiment.replace(f"{BASE}/","")

        if not os.path.exists(f"{RESULTS_FOLDER}{ex_path}"):
            os.mkdir(f"{RESULTS_FOLDER}{ex_path}")

        log.debug(f"Starting extraction of: {sub_experiment}")
        with open(f"{RESULTS_FOLDER}{ex_path}/{sub_experiment}.json", "w") as file:
            json.dump(
                get_all_data_experiment(experiment, sub_experiment), 
                file,
                indent=1
            )