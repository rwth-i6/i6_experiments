# Load all data: wers, error, lr's, configs

import subprocess
import re
from typing import OrderedDict
import glob
import json
import logging as log
import os
log.basicConfig(level=log.DEBUG)


data_set_prefixes = {
    "dev-other" : "_",
    "dev-clean" : "_dev-clean",
    "train-other" : "_train-other",
    "train-clean" : "_tain-clean"
}

datasets = list(data_set_prefixes.keys())

all_existing_experiments = [s.replace("alias/", "") for s in glob.glob("alias/conformer/*") ]
all_existing_experiments = ["conformer/baseline"] # With this you can filter for a specific setup
sub_experiments = {k : [] for k in all_existing_experiments}
for i, k in enumerate(all_existing_experiments):
    sub_experiments[k] = [s.split("/")[-1] for s in glob.glob(f"alias/{all_existing_experiments[i]}/*") if not "recog_" in s ]
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

    for line in data_string.split("\n"):
        if "num_params:" in line:
            num_params = float(re.findall(float_or_int, line)[0])
        elif "average steps per subepoch" in line:
            avg_s_per_sep = float(re.findall(float_or_int, line)[0])
        elif "training time per subepoch" in line and not time_p_sep:
            time_p_sep = float(re.findall(float_or_int, line)[0])
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
            best_ep_by_score = int(re.findall(float_or_int, line)[0])
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
        ex_path = experiment.replace("conformer/","")

        if not os.path.exists(f"{RESULTS_FOLDER}{ex_path}"):
            os.mkdir(f"{RESULTS_FOLDER}{ex_path}")

        log.debug(f"Starting extraction of: {sub_experiment}")
        with open(f"{RESULTS_FOLDER}{ex_path}/{sub_experiment}.json", "w") as file:
            json.dump(
                get_all_data_experiment(experiment, sub_experiment), 
                file,
                indent=1
            )