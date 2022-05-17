import re
from typing import OrderedDict
import glob
import json
import logging as log
import os
import argparse
import csv
import numpy as np
import inspect
import importlib.util
import itertools

log.basicConfig(level=log.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('-c','--configs', action="append", help='Configs to compare')
parser.add_argument('-ax', '--all-of-experiment', default=None)

RESULTS_FOLDER = "results2"

# Configs shoule be specified as 'experiment:sub_experiment'

args = parser.parse_args()
configs = None
if args.all_of_experiment:
    configs = [f"{args.all_of_experiment}:{sub_ex}" for sub_ex in \
        [s.split("/")[-1].replace(".json", "") for s in \
            glob.glob(f"{RESULTS_FOLDER}/{args.all_of_experiment}/*")]]
else:
    configs = args.configs
log.debug(f"parsing configs: {configs}")

args_always_different = [
    "model", # Save path to the model
    "learning_rates_file" # Could be different wouldn't matter
]

def get_diffs_two_configs(c1, c2):

    args1 = [x for x in dir(c1) if "__" not in x]
    args2 = [x for x in dir(c2) if "__" not in x]

    args_only_c1 = np.setdiff1d(args1,args2)
    args_only_c2 = np.setdiff1d(args2,args1)

    common_args = [ x for x in np.intersect1d(args1, args2) if not x in args_always_different ]

    # Now find arguments that are different
    diff_args = []
    for x in common_args:

        attr1 = getattr(c1, x)
        attr2 = getattr(c2, x)


        if attr1 != attr2:
            # Special case if arg is a function, compare the source code
            if x == "network":
                log.debug("Detected difference in network")
                diff = f"(net)| TODO" # Handle this better
                diff_args.append(diff)
                continue

            if callable(attr1) and callable(attr2):
                s1 = attr1.__code__.co_code # Get source don't work here
                s2 = attr2.__code__.co_code
                if s1 != s2:
                    diff = f"(func){x}|__code__.co_code differs"
                    log.debug(f"found difference: {diff}")
                    diff_args.append(
                        diff
                    )
            else:
                diff = f"{x}|{attr1}<->{attr2}"
                log.debug(f"found difference: {diff}")
                diff_args.append(
                    diff
                )

    return OrderedDict(
        different_arg_values = diff_args,
        different_base_vars = [
            args_only_c1, # Vars only in c1 not in c2
            args_only_c2   # Vars only in c2 not in c1
        ]
    )

def get_config_data(config):
    ex, sub_ex = config.split(":")
    res_data = None

    # Load the results file, see make_py_summary.py
    with open(f"{RESULTS_FOLDER}/{ex}/{sub_ex}.json", "r") as f:
        res_data = json.load(f)

    # Load the config data return style via using exec
    spec = importlib.util.spec_from_loader('config_data', loader=None)
    config_data = importlib.util.module_from_spec(spec)

    data = None
    with open(res_data["config_path"], 'r') as file:
        data = file.read()

    exec(data, config_data.__dict__)

    return res_data["name"], config_data


config_datas = {}
for c in configs:
    name, data = get_config_data(c)
    config_datas[name] = data

# Now lets compare all configs

# Test *all* combinations of len 2
config_names = list(config_datas.keys()) # This is our order now!
combintaions = list(itertools.combinations(config_names, 2))

config_diffs = {}
for c in combintaions:
    log.debug(f"\ncomparing: \n{c[0]}\nvs\n{c[1]}\n")
    differences = get_diffs_two_configs(config_datas[c[0]], config_datas[c[1]])
    log.debug(differences)

    config_diffs[f"{c[0]:c[1]}"] = differences

# Now generate a table with all differences: