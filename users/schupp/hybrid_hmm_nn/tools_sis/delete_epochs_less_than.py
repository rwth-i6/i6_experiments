import os
import glob
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import json
import sis

import argparse
# Create the jinja2 environment.
current_directory = os.path.dirname(os.path.abspath(__file__))

env = Environment(loader=FileSystemLoader(current_directory))
parser = argparse.ArgumentParser(prog="Delete some epoch greather than X",description="Bla", epilog="BL", formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("--path", "-p", dest="path",type=str, help="full paht to train.job dir")
parser.add_argument("--less_than_ep", "-ep", dest="epoch",type=int, help="< shouold be deleted")

args = parser.parse_args()

all_epoch_files = glob.glob(args.path + "/output/models/epoch.*.*")

for file in all_epoch_files:
    ep_name = file.split("/")[-1]
    ep_num = int(ep_name.split(".")[1])
    if ep_num < args.epoch:
        print("DELETING: " + file)
        os.remove(file)
