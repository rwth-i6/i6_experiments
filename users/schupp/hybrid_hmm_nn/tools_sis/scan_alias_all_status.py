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

# What all do we want part of our overview?:

# 1 - Global tag name e.g.: conformer
# 2 - Subtag name

consider_out_dirs = ["conformer/best/"] # TODO: use should be asked which of these he wants updated

# Lets per default consider all output directories:
consider_out_dirs = glob.glob("alias/conformer/*")
consider_out_dirs = [x.replace("alias/", "") + "/" for x in consider_out_dirs]

parser = argparse.ArgumentParser(prog="LW Dev Helper",description="Helps working with this setup", epilog="BL", formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("--filter", "-f", dest="filt",type=str, help="Filter for some setup?")
parser.add_argument("--only-pid", "-pid", dest="only_pid", action="store_true", help="Filter for some setup?")
parser.add_argument("--only-name", "-name", dest="only_name", action="store_true", help="Filter only names")
parser.add_argument("--only-done", "-done", dest="only_done", action="store_true", help="Filter only dones")
parser.add_argument("--only-not-done", "-nd", dest="only_not_done", action="store_true", help="Filter only not done")

args = parser.parse_args()

if args.filt is not None:
    consider_out_dirs = [args.filt]

only_pid = False
if args.only_pid is not None:
    only_pid = args.only_pid

only_name = False
if args.only_name is not None:
    only_name = args.only_name

only_done = False
if not args.only_done is None:
    only_done = args.only_done

only_not_done = args.only_not_done
if not args.only_not_done is None:
    only_not_done = args.only_not_done

if not only_pid and not only_name:
    print()
    print("====================================================")
    print("= Checking setup status " + str(datetime.now()) + " =")
    print("====================================================")
    print("Considering: " + str(consider_out_dirs))


update = [True]
_dir_files = {}
_first_names = []
# Load this from alreay known
_file_data_map = {}


def filt(name): # Removes the dataset tag from experiment name
    for x in ["dev-other", "dev-clean", "test-other", "test-clean"]:
        if x in name:
            return name[-(len(x)+1)]
    else:
        return name


def get_epoch(path):
    # Try to read log file:
    models_path = path + "/train.job/output/models/"
    all_eps = sorted(glob.glob(models_path + "epoch.*.index"))
    error_path = path + "/train.job/*error*"
    done_path = path + "/train.job/finished.run.*"
    possible_error = glob.glob(error_path)
    possible_finished = glob.glob(done_path)
    if len(possible_error) != 0:
        return "ERROR"

    if len(all_eps) == 0:
        # check if the setup as an error
        return "NONE"

    ep = all_eps[-1].split("/")[-1].split(".")[1]

    if len(possible_finished) != 0:
        return ep + " DONE"
    return ep

def get_job_id(path):
    # Test
    submit_log = glob.glob(path + "/train.job/submit_log.run")
    if len(submit_log) == 0:
        return "NO_ID_FOUND"
    else: # TODO: this would be way more beautiful with some regrex
        out_f = ""
        with open(submit_log[-1], 'r') as f:
            out_f = f.read()

        out = out_f.splitlines()[-1]
        cut = out[out.index("engine_info"): len(out) -2]
        cut2 = cut[cut.index("("): cut.index(")")]
        pid = int(cut2[cut2.index("'") +1: len(cut2) - 1])
        return pid
        
error_list = []

i = 0
for drr in consider_out_dirs:
    if not only_pid and not only_name:
        print()
        print("=========================== " + drr + "=============================")
    _first_names.append(drr)
    if drr not in _dir_files:
        #print("DIR: " + drr)
        _dir_files[drr] = glob.glob("alias/" + drr + "*") # gets all them files in that dir

        # Now they need to be filtered
        _dir_files[drr] = [f for f in _dir_files[drr] if not ("recog_" in f) ]

        # scan in the files
        for file in _dir_files[drr]:

            filtered_name = filt(file)
            filtered_name = filtered_name.split("/")[-1]
            if filtered_name not in _file_data_map:
                if not only_pid and not only_name:
                    print(filtered_name, end="")
                try:
                    epoch = get_epoch(file)
                    pid = get_job_id(file)
                    if not only_pid and not only_name:
                        print(f" [{pid}]: ", end="")
                        print("Epoch: " + str(epoch))
                    elif only_pid:
                        if only_not_done:
                            if not "DONE" in epoch:
                                print(pid)
                        else:
                            if not only_done or "DONE" in epoch:
                                print(pid)
                    elif only_name:
                        if only_not_done:
                            if not "DONE" in epoch:
                                print(file)
                        else:
                            if not only_done or "DONE" in epoch:
                                print(file)

                    if epoch == "ERROR":
                        error_list.append(filtered_name)
                    _file_data_map[filtered_name] = "Nothin" #extract_results(drr[:-1], [filtered_name])
                except Exception as e:
                    _file_data_map[filtered_name] = str(e)
                    print(e)
                #_file_data_map[filtered_name] = "NOT_GENERATED"


        # remove unwanted indexes from files
        for tag in ["dev-other", "dev-clean", "test-other", "test-clean"]:
            if tag in drr:
                del _dir_files[tag] # this delete only the element?

    i += 1

if not only_pid and not only_name:
    print()
    print("=========================== " + "ERRORS" + " =============================")
    for x in error_list:
        print(x)

_dir_files_short_names = { name : sorted([ d.split("/")[-1] for d in _dir_files[name] ]) for name in _dir_files.keys()} # Short name descriptions
#print(_dir_files_short_names)

first_names = consider_out_dirs
second_names = _dir_files_short_names
content_map = _file_data_map

